import torch

torch.set_float32_matmul_precision('high')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class DDPMSampler(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.n_steps = model.n_steps
        beta = torch.linspace(1e-4, 0.02, self.n_steps)
        self.register_buffer('beta', beta)
        self.register_buffer('alpha', 1. - self.beta)
        self.register_buffer('alpha_cumprod', torch.cumprod(self.alpha, 0))
        self.register_buffer('sqrt_alpha_cumprod', torch.sqrt(self.alpha_cumprod))
        self.register_buffer('sqrt_1m_alpha_cumprod', torch.sqrt(1. - self.alpha_cumprod))

    @torch.no_grad()
    def forward(self, prevs, subseq):
        self.to(device)
        self.model.to(device)
        prevs = prevs.to(device)
        subseq = subseq.to(device)
        n = len(prevs)
        x = torch.randn(n, 3, prevs.shape[2], prevs.shape[3], device=device)
        if self.model.use_3d:
            model_in = torch.stack([prevs, x, subseq], dim=2)
        else:
            model_in = torch.cat([prevs, x, subseq], dim=1)
        for i in reversed(range(self.n_steps)):
            t = torch.full((n,), i, device=device)
            alpha = self.alpha[t].view(-1, 1, 1, 1)
            sqrt_1m_alpha_cumprod = self.sqrt_1m_alpha_cumprod[t].view(-1, 1, 1, 1)
            beta = self.beta[t].view(-1, 1, 1, 1)
            noise_hat = self.model(model_in, t)
            if i > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            x = (x - (beta/sqrt_1m_alpha_cumprod) * noise_hat) / torch.sqrt(alpha) + torch.sqrt(beta) * noise
            if self.model.use_3d:
                model_in = torch.stack([prevs, x, subseq], dim=2)
            else:
                model_in = torch.cat([prevs, x, subseq], dim=1)
        return x

class DDIMSampler(DDPMSampler):
    def __init__(self, model):
        super().__init__(model)
    
    @torch.no_grad()
    def forward(self, prevs, subseq, eta=0.0, num_steps=None):
        self.to(device)
        self.model.to(device)
        prevs = prevs.to(device)
        subseq = subseq.to(device)
        n = len(prevs)
        x = torch.randn(n, 3, prevs.shape[2], prevs.shape[3], device=device)
        if self.model.use_3d:
            model_in = torch.stack([prevs, x, subseq], dim=2)
        else:
            model_in = torch.cat([prevs, x, subseq], dim=1)

        # Create reduced timestep sequence
        num_steps = num_steps or self.n_steps
        step_indices = torch.linspace(0, self.n_steps-1, num_steps, device=device).int()
        steps = list(reversed(step_indices.tolist()))

        for i, step in enumerate(steps):
            t = torch.full((n,), step, device=device)
            
            # Calculate previous timestep
            prev_step = steps[i+1] if i < len(steps)-1 else -1
            
            # Get alpha cumulative products
            alpha_cumprod = self.alpha_cumprod[step].view(-1, 1, 1, 1)
            alpha_cumprod_prev = self.alpha_cumprod[prev_step] if prev_step >= 0 else torch.ones(1, device=device)
            alpha_cumprod_prev = alpha_cumprod_prev.view(-1, 1, 1, 1)

            # Predict noise
            noise_hat = self.model(model_in, t)

            # Calculate x0 estimate
            x0_t = (x - torch.sqrt(1 - alpha_cumprod) * noise_hat) / torch.sqrt(alpha_cumprod)
            sigma_t = eta * torch.sqrt((1 - alpha_cumprod_prev) / (1 - alpha_cumprod)) * torch.sqrt(1 - alpha_cumprod / alpha_cumprod_prev)
            dir_xt = torch.sqrt(1 - alpha_cumprod_prev - sigma_t**2) * noise_hat

            noise = sigma_t * torch.randn_like(x) if i < len(steps)-1 else torch.zeros_like(x)

            # Update x
            x = torch.sqrt(alpha_cumprod_prev) * x0_t + dir_xt + noise
            if self.model.use_3d:
                model_in = torch.stack([prevs, x, subseq], dim=2)
            else:
                model_in = torch.cat([prevs, x, subseq], dim=1)
        
        return x