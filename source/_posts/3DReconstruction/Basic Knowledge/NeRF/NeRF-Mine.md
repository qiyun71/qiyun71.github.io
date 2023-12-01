---
title: 基于Instant-nsr-pl创建项目
date: 2023-07-06 21:17:54
tags:
  - NeRF
  - Neus
  - InstantNGP
categories: 3DReconstruction/Basic Knowledge/NeRF
---

本项目[yq010105/NeRF-Mine (github.com)](https://github.com/yq010105/NeRF-Mine)基于[Instant-nsr-pl](https://github.com/bennyguo/instant-nsr-pl)(NSR,NGP,PytorchLightning)代码构建

- 保留 omegaconf、nerfacc、Mip-nerf_loss，类似文件结构
- 去除 pytorch-lightning 框架，使用 pytorch

NeRF 主要部分：

- 神经网络结构 --> 训练出来模型，即 3D 模型的隐式表达
  - 网络类型一般为 MLP，相当于训练一个函数，输入采样点的位置，可以输出该点的信息(eg: density, sdf, color...)
- [采样方式](/3DReconstruction/Basic%20Knowledge/NeRF/NeRF/Sampling)：沿着光线进行采样获取采样点
- [位置编码](/3DReconstruction/Basic%20Knowledge/NeRF/NeRF/Encoding)：对采样点的位置 xyz 和方向 dir 进行编码，使得 MLP 的输入为高频的信息
- [数学相关](/3DReconstruction/Basic%20Knowledge/NeRF/NeRF/Math)：光线的生成、坐标变换、体渲染公式、BRDF……
- 体渲染函数：
  - NeRF：$\mathrm{C}(r)=\int_{\mathrm{t}_{\mathrm{n}}}^{\mathrm{t}_{\mathrm{f}}} \mathrm{T}(\mathrm{t}) \sigma(\mathrm{r}(\mathrm{t})) \mathrm{c}(\mathrm{r}(\mathrm{t}), \mathrm{d}) \mathrm{dt} =\sum_{i=1}^{N} T_{i}\left(1-\exp \left(-\sigma_{i} \delta_{i}\right)\right) \mathbf{c}_{i}$
    - 不透明度$\sigma$，累计透光率 --> 权重
    - 颜色值
  - Neus：$C(\mathbf{o},\mathbf{v})=\int_{0}^{+\infty}w(t)c(\mathbf{p}(t),\mathbf{v})\mathrm{d}t$
    - sdf, dirs, gradients, invs --> $\alpha$ --> 权重
    - 颜色值
  - NeRO：$\mathbf{c}(\omega_{0})=\mathbf{c}_{\mathrm{diffuse}}+\mathbf{c}_{\mathrm{specular}} =\int_{\Omega}(1-m)\frac{\mathbf{a}}{\pi}L(\omega_{i})(\omega_{i}\cdot\mathbf{n})d\omega_{i} + \int_{\Omega}\frac{DFG}{4(\omega_{i}\cdot\mathbf{n})(\omega_{0}\cdot\mathbf{n})}L(\omega_{i})(\omega_{i}\cdot\mathbf{n})d\omega_{i}$
    - 漫反射颜色：Light(直射光)，金属度 m、反照率 a
    - 镜面反射颜色：Light(直射光+间接光)，金属度 m、反照率 a、粗糙度$\rho$ ，碰撞概率 occ_prob，间接光碰撞 human 的 human_light
    - 详情见[NeRO Code](NeRO-code.md)
- 隐式模型导出(.stl、.obj、.ply 等)显式模型(Marching Cube)：利用 trimesh，torchmcubes，mcubes 等库
  - 根据 sdf 和 threshold，获取物体表面的 vertices 和 faces(如需还要生成 vertices 对应的 colors)。
  - 然后根据 vertices、faces 和 colors，由 trimesh 生成 mesh 并导出模型为 obj 等格式

Future：

- [ ] 消除颜色 or 纹理与几何的歧义，Neus(X-->MLP-->SDF)的方法会将物体的纹理建模到物体的几何中
- [x] 只关注前景物体的建模，可以结合 SAM 将图片中的 interest object 分割出来: **Rembg**分割后效果也不好

<!-- more -->

NeRF-Mine 文件结构：

- confs/ 配置文件
  - dtu.yaml
- encoder/ 编码方式
  - get_encoding.py
  - frequency.py
  - hashgrid.py
  - spherical.py
- process_data/ 处理数据集
  - dtu.py
- models/ 放一些网络的结构和网络的运行和方法
  - network.py 基本网络结构
  - neus.py neus 的网络结构
  - utils.py
- systems/ 训练的程序
  - neus.py 训练 neus 的程序
- utils/ 工具类函数
- run.py 主程序
- inputs/ 数据集
- outputs/ 输出和 log 文件
  - logs filepath: /root/tf-logs/name_in_conf/trial_name

```bash
# 训练
python run.py --config confs/neus-dtu.yaml --train
# 恢复训练
python run.py --config confs/neus-dtu.yaml --train --resume ckpt_path

# test to 生成mesh + video
python run.py --config confs/neus-dtu.yaml --test --resume ckpt_path
```

# Question2023.9.17

- mesh 精度：
  - 一个可以评价模型质量的指标，目前大部分方法都只能通过定性的观察来判断，而**定量的比较只能比较渲染图片，而不能比较模型**
  - 改进
    - x Method1，提高前景 occupy grid 的分辨率，虽然细节颜色更加正确，**但同时带来 eikonal 项损失难收敛的问题**
    - x Method2，将前景和背景的 loss 分开反向传播，loss_fg 只反向传播到 fg 的 MLP。由于无法准确预测光线/像素是背景还是前景，因此重建效果很差。
      - x bg 的 loss 只有 L1_rgb 的话，求出来的 loss 没有 grad_fn，无法反向传播，添加条件 if loss_bg.grad_fn is not None，效果不好
    - Method3，先用之前方法训练得到一个深度 mask
- mesh 颜色：
  - neus 方式逆变换采样训练出来的 color，会分布在整个空间中，因此虽然 render 出来的视频效果很好，但是 mesh 表面点的颜色会被稀释

> [How to reconstruct texture after generating mesh ? · Issue #48 · Totoro97/NeuS (github.com)](https://github.com/Totoro97/NeuS/issues/48) >[What can we do with our own trained model? · Issue #44 · bmild/nerf (github.com)](https://github.com/bmild/nerf/issues/44)

Neus: 表面点的颜色
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230917192416.png)

![00300000_88_158.gif](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/00300000_88_158.gif)

如果采用更快速的 NGP+Neus 的方法，由于使用了占空网格的方式采样，因此不会将表面点的颜色散射到空间背景中，这样在 extract mesh 的时候，使用简单的法向量模拟方向向量，即可得到很好的效果

Instant-nsr-pl 表面点颜色：
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230917192510.png)

# 实验

## 环境配置

[NeRF-Mine/README.md](https://github.com/qiyun71/NeRF-Mine/blob/main/README.md)

```bash
# conda 技巧
conda remove -n  需要删除的环境名 --all
conda env create --file xxx.yaml
conda env update --file xxx.yaml --prune
    ## --prune: uninstalls dependencies which were removed from `xxx.yml`

# conda 清华源
vi  ~/.condarc
修改为以下内容
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch-lts: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  simpleitk: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
```

## EXP

### exp1

```
python imgs2poses.py E:\3\Work\EXP\exp1\dataset\Miku_exp1
python gen_cameras.py E:\3\Work\EXP\exp1\dataset\Miku_exp1

# 训练
python run.py --config confs/neus-dtu_like.yaml --train
# 恢复训练
python run.py --config confs/neus-dtu_like.yaml --train --resume ckpt_path

# test to 生成mesh + video
python run.py --config confs/neus-dtu_like.yaml --test --resume ckpt_path
```

![image.png|333](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231006095349.png)

### exp2

```
python imgs2poses.py E:\3\Work\EXP\exp2\dataset\Miku_exp2
python gen_cameras.py E:\3\Work\EXP\exp2\dataset\Miku_exp2

# 训练
python run.py --config confs/neus-dtu_like.yaml --train
# 恢复训练
python run.py --config confs/neus-dtu_like.yaml --train --resume ckpt_path

# test to 生成mesh + video
python run.py --config confs/neus-dtu_like.yaml --test --resume ckpt_path
```


## Dataset

[数据集](Datasets.md)：

| Paper      | Dataset                                          | Link                                                                                                                                      |
| ---------- | ------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------- |
| NeRF       | nerf_synthetic,nerf_llff_data,LINEMOD,deepvoxels | [bmild/nerf: Code release for NeRF (Neural Radiance Fields) (github.com)](https://github.com/bmild/nerf#project-page--video--paper--data) |
| Neus       | dtu,BlenderMVS,custom                            | [Totoro97/NeuS: Code release for NeuS (github.com)](https://github.com/Totoro97/NeuS#project-page---paper--data)                          |
| Point-NeRF | dtu,nerf_synthetic,ScanNet,Tanks and temple      | [Xharlie/pointnerf: Point-NeRF: Point-based Neural Radiance Fields (github.com)](https://github.com/Xharlie/pointnerf#data-preparation)   |

自定义数据集：
[NeuS/preprocess_custom_data at main · Totoro97/NeuS (github.com)](https://github.com/Totoro97/NeuS/tree/main/preprocess_custom_data) - [Neus_custom_data](NeuS.md#Neus使用自制数据集) - [Neus-Instant-nsr-pl](Neus-Instant-nsr-pl.md#自定义数据集) - 大概需要的视图数量 20Simple/40Complex ref: https://arxiv.org/abs/2310.00684

```
<case_name>
|-- cameras_xxx.npz    # camera parameters
    |-- camera_mat_{}.npy，intrinsic
    |-- camera_mat_inv_{}.npy
    |-- world_mat_{}.npy，intrinsic @ w2c --> world to pixel
    |-- world_mat_inv_{}.npy
    |-- scale_mat_{}.npy ，根据手动清除point得到的sparse_points_interest.ply
    |-- scale_mat_inv_{}.npy
|-- image
    |-- 000.png        # target image for each view
    |-- 001.png
    ...
|-- mask
    |-- 000.png        # target mask each view (For unmasked setting, set all pixels as 255)
    |-- 001.png
    ...
```

neuralangelo 提供了 blender 插件可以可视化 colmap 数据，但是会出现 image plane 与 camera plane 不重合的情况

```
DATA_PATH
├─ database.db      (COLMAP database)
├─ images           (undistorted input images)
├─ images_raw       (raw input images)
├─ sparse           (COLMAP data from SfM)
│  ├─ cameras.bin   (camera parameters)
│  ├─ images.bin    (images and camera poses)
│  ├─ points3D.bin  (sparse point clouds)
│  ├─ 0             (a directory containing individual SfM models. There could also be 1, 2... etc.)
│  ...
├─ stereo (COLMAP data for MVS, not used here)
...
```

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231004155843.png)

**需要对 colmap 数据做(BA and) Undistortion**

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231004160230.png)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231004160314.png)




# 代码结构

## confs 配置文件

```yaml
name: ${model.name}-${dataset.name}-${basename:${dataset.root_dir}}
seed: 7 #3407
tag: ''

dataset:
  name: dtu
  root_dir: ./inputs/dtu_scan24
  render_cameras_name: cameras_sphere.npz
  object_cameras_name: cameras_sphere.npz
  img_downscale: 1
  apply_mask: False
  test_steps: 60
...
```

**omegaconf**获取 yaml 中参数
**argparse**获取终端输入的参数

```python
# ============ Register OmegaConf Recolvers ============= #
OmegaConf.register_new_resolver('calc_exp_lr_decay_rate', lambda factor, n: factor**(1./n))
OmegaConf.register_new_resolver('add', lambda a, b: a + b)
OmegaConf.register_new_resolver('sub', lambda a, b: a - b)
OmegaConf.register_new_resolver('mul', lambda a, b: a * b)
OmegaConf.register_new_resolver('div', lambda a, b: a / b)
OmegaConf.register_new_resolver('idiv', lambda a, b: a // b)
OmegaConf.register_new_resolver('basename', lambda p: os.path.basename(p))
# ======================================================= #

def load_config(*yaml_files, cli_args=[]):
    yaml_confs = [OmegaConf.load(f) for f in yaml_files]
    cli_conf = OmegaConf.from_cli(cli_args)
    conf = OmegaConf.merge(*yaml_confs, cli_conf)
    OmegaConf.resolve(conf) # 就地解析所有配置文件的内插$
    return conf

def config_to_primitive(config, resolve=True):
    return OmegaConf.to_container(config, resolve=resolve)
```

```python
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='path to config file')

    parser.add_argument('--resume', default=None, help='path to the weights to be resumed')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train', action='store_true')
    group.add_argument('--mesh', action='store_true')
    group.add_argument('--test', action='store_true')

    parser.add_argument('--outputs_dir', default='./outputs')
    parser.add_argument('--logs_dir', default='/root/tf-logs')

    args, extras = parser.parse_known_args()
    return args, extras
```

## run.py 主程序

**获取配置文件和终端输入**

```python
# extras：其他config_parser中没有添加的arg
args, extras = get_args()
from utils.misc import load_config, seed_everything
config = load_config(args.config, cli_args=extras)

config.trainer.outputs_dir = config.get('outputs_dir') or os.path.join(args.outputs_dir, config.name)

# args.resume : /root/autodl-tmp/new/NeRF-Mine/outputs/neus-dtu-Miku/@20230815-154030/ckpt/ckpt_000200.pth
if args.resume is not None:
    config.trainer.trial_name = args.resume.split('/')[-3]
else:
    config.trainer.trial_name = config.get('trial_name') or (config.tag + datetime.now().strftime('@%Y%m%d-%H%M%S'))

config.trainer.logs_dir = config.get('logs_dir') or os.path.join(args.logs_dir, config.name ,config.trainer.trial_name)
config.trainer.save_dir = config.get('save_dir') or os.path.join(config.trainer.outputs_dir, config.trainer.trial_name, 'save')
config.trainer.ckpt_dir = config.get('ckpt_dir') or os.path.join(config.trainer.outputs_dir, config.trainer.trial_name, 'ckpt')
config.trainer.code_dir = config.get('code_dir') or os.path.join(config.trainer.outputs_dir, config.trainer.trial_name, 'code')
config.trainer.config_dir = config.get('config_dir') or os.path.join(config.trainer.outputs_dir, config.trainer.trial_name, 'config')

if 'seed' not in config:
    config.seed = int(time.time() * 1000) % 1000
seed_everything(config.seed)

os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"
```

**global seed setting**

```python
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
```

### 根据配置导入模块

- from models.neus import NeuSModel
  - model = NeuSModel(config.model)
- from systems.neus import Trainer
  - trainer = Trainer(model, config.trainer, args)
- from process_data.dtu import NeRFDataset
  - train_dm = NeRFDataset(config.dataset,stage='train')
    - train_loader = torch.utils.data.DataLoader(train_dm, batch_size=3, shuffle=True)
  - val_dm = NeRFDataset(config.dataset,stage = 'valid')
    - val_loader = torch.utils.data.DataLoader(val_dm, batch_size=1)
  - test_dm = NeRFDataset(config.dataset,stage='test')
    - test_loader = torch.utils.data.DataLoader(test_dm, batch_size=1)

运行：

- trainer.train(train_loader, val_loader)
- trainer.test(test_loader)
- trainer.mesh()

## process_data

**train**

for data in train_loader:

data:

- pose: torch.Size([4, 4])
- direction: torch.Size([960, 544, 3])
- index: torch.Size([1])
- H: 960
- W: 544
- image: torch.Size([960, 544, 3])
- mask: torch.Size([960, 544])

```python
{'pose': tensor([[[-0.0898,  0.8295, -0.5512,  1.3804],
         [ 0.0600,  0.5570,  0.8284, -2.0745],
         [ 0.9941,  0.0413, -0.0998,  0.0576],
         [ 0.0000,  0.0000,  0.0000,  1.0000]]], device='cuda:0'),

'direction': tensor([[[[-0.3379, -0.5977,  1.0000],
          [-0.3354, -0.5977,  1.0000],
          [-0.3329, -0.5977,  1.0000],
          ...,
         [[-0.3379,  0.5990,  1.0000],
          [-0.3354,  0.5990,  1.0000],
          [-0.3329,  0.5990,  1.0000],
          ...,
          [ 0.3341,  0.5990,  1.0000],
          [ 0.3366,  0.5990,  1.0000],
          [ 0.3391,  0.5990,  1.0000]]]], device='cuda:0'),

'index': tensor([124]),
'H': ['480'],
'W': ['272'],
'image': tensor([[[[0.3242, 0.3438, 0.3164],
          [0.3281, 0.3477, 0.3203],
          [0.3320, 0.3398, 0.3125],
         ...,
         [[0.5977, 0.6133, 0.5938],
          [0.5977, 0.6133, 0.5938],
          [0.5977, 0.6133, 0.5938],
          ...,
          [0.6289, 0.7539, 0.8789],
          [0.6328, 0.7578, 0.8828],
          [0.6328, 0.7578, 0.8828]]]], device='cuda:0'),
'mask': tensor([[[0.9961, 0.9961, 0.9961,  ..., 0.9961, 0.9961, 0.9961],
     [0.9961, 0.9961, 0.9961,  ..., 0.9961, 0.9961, 0.9961],
     [0.9961, 0.9961, 0.9961,  ..., 0.9961, 0.9961, 0.9961],
     ...,
     [0.9961, 0.9961, 0.9961,  ..., 0.9961, 0.9961, 0.9961],
     [0.9961, 0.9961, 0.9961,  ..., 0.9961, 0.9961, 0.9961],
     [0.9961, 0.9961, 0.9961,  ..., 0.9961, 0.9961, 0.9961]]],
   device='cuda:0')}
```

## other ref code

DTU 提取 mesh 后，通过 object masks 移除背景[NeuralWarp/evaluation/mesh_filtering.py at main · fdarmon/NeuralWarp (github.com)](https://github.com/fdarmon/NeuralWarp/blob/main/evaluation/mesh_filtering.py#L108)
DTU 数据集的深度图数据可视化，  
调整可视化时的`v_min`和`v_max`[MVS/utils/read_and_visualize_pfm.py at master · doubleZ0108/MVS (github.com)](https://github.com/doubleZ0108/MVS/blob/master/utils/read_and_visualize_pfm.py)

# Results

[Metrics](Metrics.md#Metrics)

## Excel Function

将 C5：23.98 22.79 25.21 26.03 28.32 29.80 27.45 28.89 26.03 28.93 32.47 30.78 29.37 34.23 33.95，按空格拆分填入 C3 到 Q3
`=TRIM(MID(SUBSTITUTE($C$5," ",REPT(" ",LEN($C$5))),(COLUMN()-COLUMN($C$3))*LEN($C$5)+1,LEN($C$5)))`

# Code BUG

- [x] tf-logs 在测试时会新加一个文件夹问题
- [x] 训练过程中出现的 loss 错误

```
Error
1.
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 26.00 MiB (GPU 0; 23.69 GiB total capacity; 20.89 GiB already allocated; 23.69 MiB free; 22.11 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF  0% 0/2 [00:01<?, ?it/s]

with torch.no_grad():
    val_epoch

2.
[2023-07-25 21:10:06,584] INFO: ==>Training Epoch 1, lr = 0.010000
loss=0.3620 (nan), lr=0.008147: : 100% 178/178 [00:15<00:00, 11.85it/s]
[2023-07-25 21:10:21,609] INFO: ==>Training Epoch 2, lr = 0.008147
loss=0.1856 (0.3541), lr=0.006637: : 100% 178/178 [00:13<00:00, 13.39it/s]
[2023-07-25 21:10:34,907] INFO: ==>Training Epoch 3, lr = 0.006637
loss=0.1963 (0.2716), lr=0.005408: : 100% 178/178 [00:13<00:00, 13.45it/s]
[2023-07-25 21:10:48,139] INFO: ==>Training Epoch 4, lr = 0.005408
loss=nan (nan), lr=0.004406: : 100% 178/178 [00:06<00:00, 26.05it/s]
[2023-07-25 21:10:54,974] INFO: ==>Training Epoch 5, lr = 0.004406
loss=nan (nan), lr=0.003589: : 100% 178/178 [00:03<00:00, 46.24it/s]
[2023-07-25 21:10:58,824] INFO: ==>Validation at epoch 5
0% 0/2 [00:00<?, ?it/s]/root/NeRF-Mine/utils/mixins.py:160: RuntimeWarning: invalid value encountered i
n divide
img = (img - img.min()) / (img.max() - img.min())
/root/NeRF-Mine/utils/mixins.py:169: RuntimeWarning: invalid value encountered in cast
img = (img * 255.).astype(np.uint8)
psnr=4.844281196594238: : 100% 2/2 [00:05<00:00, 2.52s/it]
[2023-07-25 21:11:03,865] INFO: ==>Training Epoch 6, lr = 0.003589
loss=nan (nan), lr=0.002924: : 100% 178/178 [00:03<00:00, 47.74it/s]
[2023-07-25 21:11:07,595] INFO: ==>Training Epoch 7, lr = 0.002924

inv_s  ----> nan ， loss_mask ----> nan , loss_eikonal ----> nan
都有问题
sdf_grad_samples: 突然变为 0,3

guess1: lr太高1e-2 √

3.
训练出来的test_Video和mesh位置不准确 --> 数据集加载时的c2w没处理

4.
训练的效果不好且很慢 --> 没有使用processHashGrid
lr太低可能陷入局部最优

NOTE：训练过程中loss突然变得很大
TODO：防止过拟合-->添加 torch.cuda.amp.GradScaler() 解决 loss为nan或inf的问题
```

- [x] 导出 mesh 区域错误

![image.png|500](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230806160731.png)

- 可能是 mesh 网格的 ijk 区域大小设置有问题
- 或没有将 bound 进行坐标变换到训练时的世界坐标系

## Add

### Floaters No More

- OOM

```
0% 0/60 [00:00<?, ?it/s]Traceback (most recent call last):
File "run.py", line 97, in <module>
main()
File "run.py", line 88, in main
trainer.train(train_loader, val_loader)
File "/root/autodl-tmp/NeRF-Mine/systems/neus.py", line 164, in train
self.train_epoch(train_loader)
File "/root/autodl-tmp/NeRF-Mine/systems/neus.py", line 197, in train_epoch
self.scaler.scale(loss).backward()
File "/root/miniconda3/envs/neus/lib/python3.8/site-packages/torch/_tensor.py", line 488
, in backward
torch.autograd.backward(
File "/root/miniconda3/envs/neus/lib/python3.8/site-packages/torch/autograd/__init__.py"
, line 197, in backward
Variable._execution_engine.run_backward( # Calls into the C++ engine to run the backw
ard pass
File "/root/miniconda3/envs/neus/lib/python3.8/site-packages/torch/autograd/function.py"
, line 267, in apply
return user_fn(self, *args)
File "/root/autodl-tmp/NeRF-Mine/models/neus.py", line 29, in backward
return grad_output_colors * scaling.unsqueeze(-1), grad_output_sigmas * scaling, grad_
output_ray_dist
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 13.58 GiB (GPU 0; 23.70
GiB total capacity; 5.30 GiB already allocated; 7.14 GiB free; 14.86 GiB reserved in tota
l by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to a
void fragmentation. See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
0% 0/60 [00:00<?, ?it/s]
```

## Update

### Nerfacc

0.3.5 --> 0.5.3

**error1**：loss_eikonal 一直增大

- 在 fg 中添加了 def alpha_fn(t_starts,t_ends,ray_indices):
- 在 fg 中使用 ray_aabb_intersect 计算 near 和 far

效果差原因：

- 0.5.3 由于 Contraction 在射线遍历时低效，不再使用 ContractionType，因此对于背景 bg 使用 self.scene_aabb 会出现问题

解决 test：

- 对于背景的 unbounded 采用 prop 网格

**error1.5**:

- loss_rgb_mse 和 l1 损失为 nan
- 解决：背景 color 值为负数

**error2**: 对 unbounded 采用 prop 网格后，由于网络参数太多，出现 OOM

```
OOM
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 162.00 MiB (GPU 0; 23.70 GiB total capacity; 21.16 GiB already allocated; 150.56 MiB free; 22.31 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF

set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
```

> [通过设置 PYTORCH*CUDA_ALLOC_CONF 中的 max_split_size_mb 解决 Pytorch 的显存碎片化导致的 CUDA:Out Of Memory 问题*梦音 Yune 的博客-CSDN 博客](https://blog.csdn.net/MirageTanker/article/details/127998036)

- max_split_size_mb 设置后，显存也不足
- 调小 prop 的网格参数，prop_network 主要由 Hash Table 和 MLP 两部分组成，调小 HashTable 的 n_levels
