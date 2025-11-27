# Shape2Mask: A weakly supervised instance segmentation method for aluminum alloy precipitations based on prior shape knowledge

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1.1+](https://img.shields.io/badge/pytorch-2.1+-red.svg)](https://pytorch.org/)
[![Torch-geometric 2.6.1+](https://img.shields.io/badge/geometric-2.6.1+-red.svg)](https://pytorch.org/)

This repository contains the implementation of our paper 
"*Shape2Mask: A weakly supervised instance segmentation method for aluminum alloy precipitations based on prior shape knowledge*", 
introducing a novel shape-based instance segmentation framework specifically designed for nanoscale precipitations in TEM images.

## Key Features
- **Shape-Parameterized Prediction**: Replaces traditional pixel-wise mask generation with a parameterized shape regression paradigm, enabling explicit incorporation of geometric prior knowledge and improved handling of blurry or low-contrast structures in TEM images.
- **Differentiable Prior Loss Constraints**: Introduces domain-informed loss functions that directly regularize morphological properties such as aspect ratio and edge straightness, effectively bridging domain knowledge and weakly supervised learning.
- **Hybrid Architecture with GNN and KAN**: Combines Graph Neural Networks for structured shape feature extraction and Kolmogorov-Arnold Networks for highly nonlinear parameter fitting, enhancing representational flexibility and regression accuracy under limited supervision.

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 2.1.1+
- Detectron2 follow [Detectron2 installation instructions](https://detectron2.readthedocs.io/en/latest/tutorials/install.html).
- torch-geometric==2.6.1 (for GNN)
- CUDA 11+ (for GPU acceleration)

### CUDA kernel for MSDeformAttn and Diff rasterizer
Shape2Mask also uses the deformable attention modules introduced in [Deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR) and the differentiable rasterizer introduced in [BoundaryFormer](https://github.com/mlpc-ucsd/BoundaryFormer). Please build them on your system:
``` shell
bash scripts/auto_build.sh
```
or 
``` shell
cd ./modeling/layers/deform_attn
sh ./make.sh
cd ./modeling/layers/diff_ras
python setup.py build install
```

### Quick Start
1. Clone the repository:
   ```bash
   git clone https://github.com/neulmc/Shape2mask.git
   cd Shape2mask
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   
### Dataset Preparation
#### Aluminum Alloy TEM Dataset
Download our dataset from Baidu Drive: https://pan.baidu.com/s/12Hz-_s6xJaG7485_FTrzPQ?pwd=biks. (extraction code: biks)

#### Dataset Structure:
```
dataset/
    ├── metal/                   # TEM image dataset
       ├── annotations/          # Annotations (full/Box supervision)
           ├── test.json         # MS COCO format
           ├── train.json
       ├── test/                 # Test dataset
           ├── test_img1.bmp     # Observed based on TEM
           ├── test_img2.bmp
           ...
       ├── train/                # Train dataset
           ├── train_img1.bmp           
           ├── train_img1.bmp
           ...
```
#### Important Notes:
- Image Volume and Resolution: 330 TEM images with 1024×1024 pixel resolution
- Material Conditions: Includes three heat treatment states with aging times of 1 hour, 3 hours, and 12 hours
- Instance Distribution: 14,190 precipitation instances (2,978 horizontal, 8,262 vertical, 2,950 longitudinal)
- Annotation Type: Bounding box annotations for weakly supervised learning; pixel-level masks for fully supervised setting
- Spatial Resolution: 1 pixel = 0.16 nanometers
- Usage Restrictions: Designed for academic research - commercial use prohibited

### Training Pipeline
#### 1. Box-supervison instance segmentation
Configure parameters in config file and run:
```bash
python train_net.py
```
Key configurable parameters in config file:
```
cfg_MLP_KAN = True        # KAN enable
cfg_DECODE_GAT = True     # GAT enable
cfg_MERGE = True          # Fusion enable
cfg_shape_layer = 2       # num of shape-KAN module
cfg_point_layer = 2       # num of vertex-KAN module
cfg_POINTS_SHAPE_WEIGHT = 0.1     # The weight of prior loss function
cfg_POINTS_RELA_WEIGHT = 0.1
```
#### 2. Full-supervison instance segmentation

```bash
python train_sup_net.py
```
#### 3. Evaluation
The evaluation runs automatically, generating:
- Performance metrics in {method_name}/log.txt
- Prediction visualizations in {method_name}/inference/filename.png 
(*when setting save_visualization = False*)


### Table.1 Comparison of instance segmentation performance under different supervision settings

| Method                | AP-Lon | AP-Ver | AP-Hor | AP Score | Supervision Type |
|:----------------------|:------:|:------:|:------:|:--------:|:----------------:|
| **Weakly Supervised** |        |        |        |          |                  |
| BoxInst               |   –    |  38.2  |   –    |   12.7   |       Box        |
| BoxSnake              |   –    |  54.8  |   –    |   18.3   |       Box        |
| Box2Mask              |  5.6   |  29.7  |  5.3   |   13.5   |       Box        |
| DiscoBox              |   –    |  27.6  |  0.1   |   9.2    |       Box        |
| BoxLevelset           |   –    |  32.4  |   –    |   10.8   |       Box        |
| *Shape2Mask (Ours)*   |  17.9  |  52.9  |  17.6  |   29.5   |       Box        |
| **Fully Supervised**  |        |        |        |          |                  |
| BlendMask             |  19.4  |  42.7  |  17.0  |   26.4   |     Full Mask    |
| CondInst              |  21.3  |  47.4  |  24.1  |   31.0   |     Full Mask    |
| Mask R-CNN            |  23.7  |  48.2  |  18.9  |   30.3   |     Full Mask    |
| SOLOv2                |  20.9  |  44.0  |  16.6  |   27.2   |     Full Mask    |
| *Shape2Mask (Ours)*   |  28.0  |  52.3  |  19.9  |   33.4   |     Full Mask    |

*Note: "–" indicates AP score < 0.1, considered negligible. All values are reported in percentage (%).*

### Final models
This is the pre-trained model and log file in our paper. We used this model for evaluation. You can download by:
https://pan.baidu.com/s/1PQ7V5WqK9SVzWNboAjJkDw?pwd=ky6u code: ky6u.


### Code Structure
```
Shape2Mask-github/
├── config            
│   ├── shape2point_R_50_FPN_1x.yaml      # Config file
│   └── sup_shape2point_R_50_FPN_1x.yaml       
├── modeling                              # Networks used in model
│   ├── roi_heads  
│       ├── roi_heads.py                  # Base class
│       ├── shape2point_head.py           # Main Shape2Mask module
│       └── ...      
│   ├── transformer.py                    # Self/Cross-attention    
│   ├── GAT.py                            # Graph attention networks 
│   ├── KANs.py                           # Kolmogorov-Arnold Networks
│   └── ...
├── scripts                               # Source code for building deformable attention...
├── datasets     
│   ├── metal/                            # TEM image dataset
│       ├── annotations/     
│       └── ...                        
├── train_net.py                          # Execute code (weakly supervision mode)
├── train_sup_net.py                      # Execute code    
├── cust_vis.py                           # Adaptation to TEM dataset and visualization
└── requirements.txt      # Required packages
```
### Note
Sincere thanks to the following libraries and excellent work for providing us with assistance in implementing the Shape2Mask encoding.

### References
[1] <a href="https://github.com/Yangr116/BoxSnake">BoxSnake: Polygonal Instance Segmentation with Box Supervision.</a>

[2] <a href="https://github.com/Blealtan/efficient-kan">An Efficient Implementation of Kolmogorov-Arnold Network.</a>

[3] <a href="https://github.com/fundamentalvision/Deformable-DETR">Deformable DETR: Deformable Transformers for End-to-End Object Detection.</a>

[4] <a href="https://github.com/mlpc-ucsd/BoundaryFormer">Instance Segmentation With Mask-Supervised Polygonal Boundary Transformers. </a>

[5] <a href="https://github.com/facebookresearch/detectron2">Detectron2: a platform for object detection, segmentation and other visual recognition tasks.</a>

