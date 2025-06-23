import torch

# Set high precision for matrix multiplications to improve numerical stability
torch.set_float32_matmul_precision('high')
# Use GPU if available, otherwise CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class DDPMSampler(torch.nn.Module):
    """Sampler for Denoising Diffusion Probabilistic Models (DDPM)"""
    def __init__(self, model):
        super().__init__()
        self.model = model  # Underlying diffusion model
        self.n_steps = model.n_steps  # Total diffusion steps
        
        # Schedule parameters for diffusion process
        beta = torch.linspace(1e-4, 0.02, self.n_steps)  # Linear noise schedule
        self.register_buffer('beta', beta)  # Variance schedule
        self.register_buffer('alpha', 1. - self.beta)  # 1 - beta
        self.register_buffer('alpha_cumprod', torch.cumprod(self.alpha, 0))  # Product of alphas
        self.register_buffer('sqrt_alpha_cumprod', torch.sqrt(self.alpha_cumprod))  # √(ᾱ)
        self.register_buffer('sqrt_1m_alpha_cumprod', torch.sqrt(1. - self.alpha_cumprod))  # √(1-ᾱ)

    @torch.no_grad()
    def forward(self, prevs, subseq):
        """Generate intermediate frames using DDPM sampling
        Args:
            prevs: Tensor of previous frames [B, C, H, W]
            subseq: Tensor of subsequent frames [B, C, H, W]
        Returns:
            Generated intermediate frame [B, C, H, W]
        """
        # Ensure tensors are on correct device
        self.to(device)
        self.model.to(device)
        prevs = prevs.to(device)
        subseq = subseq.to(device)
        
        n = len(prevs)  # Batch size
        # Initialize with random noise
        x = torch.randn(n, 3, prevs.shape[2], prevs.shape[3], device=device)
        
        # Prepare model input based on architecture
        if self.model.use_3d:
            model_in = torch.stack([prevs, x, subseq], dim=2)  # 3D convolution input
        else:
            model_in = torch.cat([prevs, x, subseq], dim=1)  # 2D convolution input
        
        # Reverse diffusion process
        for i in reversed(range(self.n_steps)):
            t = torch.full((n,), i, device=device)  # Current timestep
            
            # Extract schedule parameters for current step
            alpha = self.alpha[t].view(-1, 1, 1, 1)
            sqrt_1m_alpha_cumprod = self.sqrt_1m_alpha_cumprod[t].view(-1, 1, 1, 1)
            beta_val = self.beta[t].view(-1, 1, 1, 1)
            
            # Predict noise component
            noise_hat = self.model(model_in, t)
            
            # Add noise except at final step
            if i > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            
            # Update x using reverse diffusion equation
            x = (x - (beta_val / sqrt_1m_alpha_cumprod) * noise_hat) / torch.sqrt(alpha) 
            x += torch.sqrt(beta_val) * noise
            
            # Update model input with new x
            if self.model.use_3d:
                model_in = torch.stack([prevs, x, subseq], dim=2)
            else:
                model_in = torch.cat([prevs, x, subseq], dim=1)
                
        return x

class DDIMSampler(DDPMSampler):
    """Sampler for Denoising Diffusion Implicit Models (DDIM)"""
    def __init__(self, model):
        super().__init__(model)
    
    @torch.no_grad()
    def forward(self, prevs, subseq, eta=0.0, num_steps=None):
        """Accelerated sampling using DDIM method
        Args:
            prevs: Previous frames [B, C, H, W]
            subseq: Subsequent frames [B, C, H, W]
            eta: (0-1) Controls stochasticity (0=deterministic)
            num_steps: Optional reduced steps for faster sampling
        Returns:
            Generated intermediate frame [B, C, H, W]
        """
        # Device setup
        self.to(device)
        self.model.to(device)
        prevs = prevs.to(device)
        subseq = subseq.to(device)
        
        n = len(prevs)  # Batch size
        # Initialize with noise
        x = torch.randn(n, 3, prevs.shape[2], prevs.shape[3], device=device)
        
        # Prepare model input
        if self.model.use_3d:
            model_in = torch.stack([prevs, x, subseq], dim=2)
        else:
            model_in = torch.cat([prevs, x, subseq], dim=1)
        
        # Create reduced timestep sequence
        num_steps = num_steps or self.n_steps
        step_indices = torch.linspace(0, self.n_steps-1, num_steps, device=device).int()
        steps = list(reversed(step_indices.tolist()))  # Reverse diffusion order
        
        for i, step in enumerate(steps):
            t = torch.full((n,), step, device=device)  # Current step
            
            # Determine previous step index
            prev_step = steps[i+1] if i < len(steps)-1 else -1
            
            # Get alpha cumulative products
            alpha_cumprod = self.alpha_cumprod[step].view(-1, 1, 1, 1)
            alpha_cumprod_prev = self.alpha_cumprod[prev_step].view(-1, 1, 1, 1) if prev_step >= 0 else torch.ones(1, device=device).view(-1, 1, 1, 1)
            
            # Predict noise
            noise_hat = self.model(model_in, t)
            
            # Estimate original image (x0) at current step
            x0_t = (x - torch.sqrt(1 - alpha_cumprod) * noise_hat) / torch.sqrt(alpha_cumprod)
            
            # Calculate variance term
            sigma_t = eta * torch.sqrt((1 - alpha_cumprod_prev) / (1 - alpha_cumprod)) * torch.sqrt(1 - alpha_cumprod / alpha_cumprod_prev)
            
            # Direction pointing to x_t
            dir_xt = torch.sqrt(1 - alpha_cumprod_prev - sigma_t**2) * noise_hat
            
            # Add stochastic noise
            noise = sigma_t * torch.randn_like(x) if i < len(steps)-1 else torch.zeros_like(x)
            
            # Update x using DDIM equation
            x = torch.sqrt(alpha_cumprod_prev) * x0_t + dir_xt + noise
            
            # Update model input
            if self.model.use_3d:
                model_in = torch.stack([prevs, x, subseq], dim=2)
            else:
                model_in = torch.cat([prevs, x, subseq], dim=1)
        
        return x