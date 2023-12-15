---
title: SDFStudio
date: 2023-12-11 20:35:28
tags:
  - 3DReconstruction
  - SurfaceReconstruction
categories: 3DReconstruction/Multi-view/Implicit Function
---

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20231211203625.png)

<!-- more -->

# 环境配置

### Create environment

SDFStudio requires `python >= 3.7`. We recommend using conda to manage dependencies. Make sure to install [Conda](https://docs.conda.io/en/latest/miniconda.html) before proceeding.

```bash
conda create --name sdfstudio -y python=3.8
conda activate sdfstudio
python -m pip install --upgrade pip
```

### Dependencies

Install pytorch with CUDA (this repo has been tested with CUDA 11.3) and [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn)

```bash
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
# 以下win10需要在vs的x64 Native Tools Command Prompt for VS 2019中进行编译和安装PyTorch extension
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

### Installing SDFStudio

```bash
git clone https://github.com/autonomousvision/sdfstudio.git
cd sdfstudio
pip install --upgrade pip setuptools
pip install -e .
# install tab completion
ns-install-cli
```

# 数据集

`ns-download-data sdfstudio` 如果网络无法连接，直接使用amazonaws地址下载

```python
ns-download-data sdfstudio --dataset-name DATASET_NAME

# pylint: disable=line-too-long
sdfstudio_downloads = {
    "sdfstudio-demo-data": "https://s3.eu-central-1.amazonaws.com/avg-projects/monosdf/data/sdfstudio-demo-data.tar",
    "dtu": "https://s3.eu-central-1.amazonaws.com/avg-projects/monosdf/data/DTU.tar",
    "replica": "https://s3.eu-central-1.amazonaws.com/avg-projects/monosdf/data/Replica.tar",
    "scannet": "https://s3.eu-central-1.amazonaws.com/avg-projects/monosdf/data/scannet.tar",
    "tanks-and-temple": "https://s3.eu-central-1.amazonaws.com/avg-projects/monosdf/data/tnt_advanced.tar",
    "tanks-and-temple-highres": "https://s3.eu-central-1.amazonaws.com/avg-projects/monosdf/data/highresTNT.tar",
    "heritage": "https://s3.eu-central-1.amazonaws.com/avg-projects/monosdf/data/Heritage-Recon.tar",
    "neural-rgbd-data": "http://kaldir.vc.in.tum.de/neural_rgbd/neural_rgbd_data.zip",
    "all": None,
}
```

其中RGBD数据需要转换为SDFStudio数据格式
```bash
# kitchen scene for example, replca the scene path to convert other scenes
python scripts/datasets/process_neuralrgbd_to_sdfstudio.py --input_path data/neural-rgbd-data/kitchen/ --output_path data/neural_rgbd/kitchen_sensor_depth --type sensor_depth
```

## 格式

```
└── scan65
  └── meta_data.json
  ├── pairs.txt
  ├── 000000_rgb.png
  ├── 000000_normal.npy
  ├── 000000_depth.npy
  ├── .....
```

`meta_data.json like:`

```json
{
  'camera_model': 'OPENCV', # camera model (currently only OpenCV is supported)
  'height': 384, # height of the images
  'width': 384, # width of the images
  'has_mono_prior': true, # use monocular cues or not
  'pairs': 'pairs.txt', # pairs file used for multi-view photometric consistency loss
  'worldtogt': [
      [1, 0, 0, 0], # world to gt transformation (useful for evauation)
      [0, 1, 0, 0],
      [0, 0, 1, 0],
      [0, 0, 0, 1],
    ],
  'scene_box': {
      'aabb': [
          [-1, -1, -1], # aabb for the bbox
          [1, 1, 1],
        ],
      'near': 0.5, # near plane for each image
      'far': 4.5, # far plane for each image
      'radius': 1.0, # radius of ROI region in scene
      'collider_type': 'near_far',
      # collider_type can be "near_far", "box", "sphere",
      # it indicates how do we determine the near and far for each ray
      # 1. near_far means we use the same near and far value for each ray
      # 2. box means we compute the intersection with the bounding box
      # 3. sphere means we compute the intersection with the sphere
    },
  'frames': [ # this contains information for each image
      {
        # note that all paths are relateive path
        # path of rgb image
        'rgb_path': '000000_rgb.png',
        # camera to world transform
        'camtoworld':
          [
            [
              0.9702627062797546,
              -0.014742869883775711,
              -0.2416049987077713,
              0.6601868867874146,
            ],
            [
              0.007479910273104906,
              0.9994929432868958,
              -0.03095100075006485,
              0.07803472131490707,
            ],
            [
              0.2419387847185135,
              0.028223417699337006,
              0.9698809385299683,
              -2.6397712230682373,
            ],
            [0.0, 0.0, 0.0, 1.0],
          ],
        # intrinsic of current image
        'intrinsics':
          [
            [925.5457763671875, -7.8512319305446e-05, 199.4256591796875, 0.0],
            [0.0, 922.6160278320312, 198.10269165039062, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
          ],
        # path of monocular depth prior
        'mono_depth_path': '000000_depth.npy',
        # path of monocular normal prior
        'mono_normal_path': '000000_normal.npy',
      },
      ...,
    ],
}
```

`pairs.txt like:` 
for multi-view photometric consistency loss

```txt
# ref image, source image 1, source image 2, ..., source image N, note source image are listed in ascending order, which means last image has largest score
000000.png 000032.png 000023.png 000028.png 000031.png 000029.png 000030.png 000024.png 000002.png 000015.png 000025.png ...
000001.png 000033.png 000003.png 000022.png 000016.png 000027.png 000023.png 000007.png 000011.png 000026.png 000024.png ...
...
```

## 自定义数据集

```bash
python scripts/datasets/process_scannet_to_sdfstudio.py --input_path /your_path/datasets/scannet/scene0050_00 --output_path data/custom/scannet_scene0050_00
```

如果需要，可以使用[omnidata:](https://github.com/EPFL-VILAB/omnidata)提取单目深度和法向用于监督
```bash
python scripts/datasets/extract_monocular_cues.py --task normal --img_path data/custom/scannet_scene0050_00/ --output_path data/custom/scannet_scene0050_00 --omnidata_path YOUR_OMNIDATA_PATH --pretrained_models PRETRAINED_MODELS

python scripts/datasets/extract_monocular_cues.py --task depth --img_path data/custom/scannet_scene0050_00/ --output_path data/custom/scannet_scene0050_00 --omnidata_path YOUR_OMNIDATA_PATH --pretrained_models PRETRAINED_MODELS
```

# Train

eg: sdfstudio-demo-data

```bash
# Train model on the dtu dataset scan65
ns-train neus-facto --pipeline.model.sdf-field.inside-outside False --vis viewer --experiment-name neus-facto-dtu65 sdfstudio-data --data data/sdfstudio-demo-data/dtu-scan65

# Or you could also train model on the Replica dataset room0 with monocular priors
ns-train neus-facto --pipeline.model.sdf-field.inside-outside True --pipeline.model.mono-depth-loss-mult 0.1 --pipeline.model.mono-normal-loss-mult 0.05 --vis viewer --experiment-name neus-facto-replica1 sdfstudio-data --data data/sdfstudio-demo-data/replica-room0 --include_mono_prior True

# resume checkpoint 根据模型参数恢复训练
ns-train neus-facto --trainer.load-dir {outputs/neus-facto-dtu65/neus-facto/XXX/sdfstudio_models} sdfstudio-data --data data/sdfstudio-demo-data/dtu-scan65 
```

## 图片(+mask)

```bash
ns-train unisurf --pipeline.model.sdf-field.inside-outside False sdfstudio-data --data data/sdfstudio-demo-data/dtu-scan65

ns-train volsdf --pipeline.model.sdf-field.inside-outside False sdfstudio-data --data data/sdfstudio-demo-data/dtu-scan65

ns-train neus --pipeline.model.sdf-field.inside-outside False sdfstudio-data --data data/sdfstudio-demo-data/dtu-scan65
```

## mono-prior

```bash
ns-train monosdf --pipeline.model.sdf-field.inside-outside True sdfstudio-data --data data/sdfstudio-demo-data/replica-room0 --include-mono-prior True

ns-train mono-unisurf --pipeline.model.sdf-field.inside-outside True sdfstudio-data --data data/sdfstudio-demo-data/replica-room0 --include-mono-prior True

ns-train mono-neus --pipeline.model.sdf-field.inside-outside True sdfstudio-data --data data/sdfstudio-demo-data/replica-room0 --include-mono-prior True
```

## Geo-Neus's pairs

```bash
ns-train geo-neus --pipeline.model.sdf-field.inside-outside False sdfstudio-data -data data/dtu/scan24 --load-pairs True

ns-train geo-unisurf --pipeline.model.sdf-field.inside-outside False sdfstudio-data -data data/dtu/scan24 --load-pairs True

ns-train geo-volsdf --pipeline.model.sdf-field.inside-outside False sdfstudio-data -data data/dtu/scan24 --load-pairs True
```

## Neus

```bash
ns-train neus-acc --pipeline.model.sdf-field.inside-outside False sdfstudio-data -data data/dtu/scan65

ns-train neus-facto --pipeline.model.sdf-field.inside-outside False sdfstudio-data -data data/dtu/scan65
```

## NeuralReconW

```bash
ns-train neusW --pipeline.model.sdf-field.inside-outside False heritage-data --data data/heritage/brandenburg_gate
```

## 其他命令

### Representations

MLP、HashGrid、Tri-plane

```bash
# Representations
## MLPs
ns-train volsdf --pipeline.model.sdf-field.use-grid-feature False sdfstudio-data --data YOUR_DATA

## Multi-res Feature Grids
ns-train volsdf --pipeline.model.sdf-field.use-grid-feature True --pipeline.model.sdf-field.encoding-type hash sdfstudio-data --data YOUR_DATA

## Tri-plane
ns-train volsdf --pipeline.model.sdf-field.use-grid-feature True  --pipeline.model.sdf-field.encoding-type tri-plane sdfstudio-data --data YOUR_DATA

## Geometry Initialization
ns-train volsdf  --pipeline.model.sdf-field.geometric-init True --pipeline.model.sdf-field.bias 0.5 --pipeline.model.sdf-field.inside-outside False
ns-train volsdf --pipeline.model.sdf-field.geometric-init True --pipeline.model.sdf-field.bias 0.8 --pipeline.model.sdf-field.inside-outside True

## Color Network
ns-train volsdf --pipeline.model.sdf-field.num-layers-color 2 --pipeline.model.sdf-field.hidden-dim-color 512
```

### Supervision

```bash
## RGB Loss 默认使用L1损失

## Mask Loss
--pipeline.model.fg-mask-loss-mult 0.001

## Eikonal Loss
--pipeline.model.eikonal-loss-mult 0.01

## Smoothness Loss
--pipeline.model.smooth-loss-multi 0.01

## Monocular Depth Consistency
--pipeline.model.mono-depth-loss-mult 0.1

## Monocular Normal Consistency
--pipeline.model.mono-normal-loss-mult 0.05

## Multi-view Photometric Consistency（Geo-NeuS.）
--pipeline.model.patch-size 11 --pipeline.model.patch-warp-loss-mult 0.1 --pipeline.model.topk 4

## Sensor Depth Loss（for RGBD data）
# truncation is set to 5cm with a rough scale value 0.3 (0.015 = 0.05 * 0.3)
--pipeline.model.sensor-depth-truncation 0.015 --pipeline.model.sensor-depth-l1-loss-mult 0.1 --pipeline.model.sensor-depth-freespace-loss-mult 10.0 --pipeline.model.sensor-depth-sdf-loss-mult 6000.0
```

### vis可视化跟踪

`--vis {viewer, tensorboard, wandb}`

# 提取和渲染mesh

```bash
ns-extract-mesh --load-config outputs/neus-facto-dtu65/neus-facto/XXX/config.yml --output-path meshes/neus-facto-dtu65.ply

ns-render-mesh --meshfile meshes/neus-facto-dtu65.ply --traj interpolate  --output-path renders/neus-facto-dtu65.mp4 sdfstudio-data --data data/sdfstudio-demo-data/dtu-scan65
```

