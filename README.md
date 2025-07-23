# 3D Denoising Diffusion Video Frame Interpolation (5DVFI)

<!-- [![Paper](https://img.shields.io/badge/arXiv-Paper-crimson.svg)](https://arxiv.org/abs/XXXX.XXXXX) -->
![GitHub](https://img.shields.io/github/license/Feebami/5DVFI)
[![Model Deployment](https://img.shields.io/badge/Deployment-HuggingFace-orange.svg)](https://huggingface.co/spaces/Feebami/DiffusionVFI)

This repository contains:
- Diffusion-based video frame interpolation using 3D U-Net architectures
- DDIM sampling for accelerated generation
- Quantitative evaluation metrics (PSNR, LPIPS, FID)
- Model training and evaluation pipelines

![Interpolation Demo](display/interp_demo.gif)

## Repository Structure

### Core Components
| File/Folder          | Description                                                                 |
|----------------------|-----------------------------------------------------------------------------|
| `base_model/`        | Core model implementations                                                  |
| --- `samplers.py`    | DDPM/DDIM sampling algorithms for frame generation                          |
| --- `train.py`       | Lightning module for training diffusion models                              |
| --- `unet.py`        | 2D and 3D U-Net architectures                                               |
| `config_search/`           | Test many configurations                                              |
| --- `configs/`            | Folder of configuration YAML files                                        |
| --- `eval_configs.py` | Evaluate a folder of trained configurations   |
| --- `train_configs.py`    | Train a folder of configurations  |
| `dataset/`           | Video dataset processing                                                    |
| --- `video_dataset.py`| Dataset loaders for UCF-101 and Vimeo-90K triplets                         |
| `interpolate/`       | Video frame interpolation scripts                                           |
| --- `interpolate_base.py` | Interpolate frames for a single video                                  |
| --- `interpolate_dir.py`  | Batch interpolation for video directories                              |
| `utility/`           | Evaluation metrics and utilities                                            |
| --- `utils.py`       | Video decoding, transformations, and metric calculations                    |
| --- `make_ucf_triplet.py` | Tripletize random subset of UCF-101 datset                             |
| `config.yaml` | Final model configuration |

### Evaluation Scripts
| Script                     | Description                                                           |
|----------------------------|-----------------------------------------------------------------------|
| `test_davis.py` | Tests models generated interpolations on the Davis dataset                  |


## Getting Started

### Installation

#### Clone repository
```
git clone https://github.com/Feebami/5DVFI.git
cd 5DVFI
```

#### Install dependancies

```
pip install -r requirements.txt
```

### Download Resources

1. **Model Weights**: [Download Weights](https://drive.google.com/file/d/1BG5yZnf5ZrAcSV21NPDgIFit-44oCNmH/view?usp=drive_link)
2. **Davis Data**: [Download test data](https://sites.google.com/view/xiangyuxu/qvi_nips19)

Place downloaded files in project directory

## Reproducing Results

### Evaluate Results

```
python test_davis.py
```

## Results

### Quantitative Evaluation (UCF-101)
| Model         | PSNR ↑        | LPIPS ↓   | FID ↓     | Parameters (M)  | RT (sec)  |
|---------------|---------------|-----------|-----------|-----------------|-----------|
| MCVD          | 18.946  | 0.247     | 32.246   | 27.3            | 52.55*    |
| [LDMVFI](https://github.com/danier97/LDMVFI) | 25.541  | 0.107     | 12.554    | 439.0           | 8.48*     |
| [MADiff](https://arxiv.org/abs/2404.13534) | 26.069  | 0.096 | 11.089    | 448.8           | 47.59^    |
| 5DVFI         | 21..481        | 0.239     | 45.322 | 56.0            | 1.174        |

*QUANTITATIVE COMPARISON OF 5DVFI AND THREE OTHER DIFFUSION-BASED VFI METHODS ON DAVIS DATASET. THE LAST TWO
COLUMNS SHOW THE NUMBER OF PARAMETERS AND RUNTIME NEEDED TO INTERPOLATE ONE 480P FRAME. 5DVFI RUNTIME WAS
BENCHMARKED WITH THE SAME NUMBER OF DDIM TIMESTEPS USED TO PRODUCE TESTING DATA (16) USING AN RTX 4070 GPU.*

*Assumed RTX 3090 GPU with 200 DDIM steps  
^Runtime needed to interpolate one frame of Middlebury testset using one V100 GPU.

### Sample Interpolations from UCF-101 testing data

| Original  | Frame Blending  | 5DVFI |
|-----------|-----------------|-------|
| ![original](display/original_examples.gif) | ![blended](display/blend_examples.gif) | ![diffused](display/diffused_examples.gif)  |

Every frame in these videos is an interpolation of real frames two apart in the frame sequence. The 5DVFI video examples are clips from samples used to calculate result metrics above. 

## Train Models

**Download Vimeo Triplet**:  
[Download](http://toflow.csail.mit.edu/)  
Put vimeo_triplet folder in working directory.

**Tripletize UCF-101 subset**  
1. Download full dataset
2. Remove files in UCF_test from full dataset 
3. Remove `v_Archery_g01_c01.avi` (this file wouldn't decode properly, so it was removed)
4. Run:
```
python -m utility.make_ucf_triplet
```
> **Warning:** This process takes a while to complete and produces 20.9GB of data.  

Generates:  
UCF-101_triplet/  
--- Video file name/  
------ 00001/  
--------- 0.jpg  
--------- 1.jpg  
--------- 2.jpg  
------ 00002/  
...

**Run training script with desired configuration**
```python
python -m base_model.train --config <path/to/config>
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

```bibtex
@misc{hamel20255dvfi,
  title     = {3D Denoising Diffusion Video Frame Interpolation},
  author    = {Chandon Hamel},
  school    = {Regis University},
  year      = {2025},
  address   = {Denver, CO, USA}
}
```
Reach out for questions or collaborations:
chandonrhamel@gmail.com

## Acknowledgements

[Vimeo 90K](http://toflow.csail.mit.edu/)  
[UCF-101](https://www.crcv.ucf.edu/data/UCF101.php)  
[LDMVFI](https://github.com/danier97/LDMVFI)  
[MADiff](https://arxiv.org/abs/2404.13534)  
[VIDIM](https://vidim-interpolation.github.io/)  
