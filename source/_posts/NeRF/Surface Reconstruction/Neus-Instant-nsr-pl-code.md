---
title: Instant-nsr-pl的代码理解
date: 2023-07-03 22:02:46
tags:
    - Python
    - Code
    - Training & Inference Efficiency
categories: NeRF/Surface Reconstruction
---

Instant Neus的代码理解
参考了：
- [ngp_pl](https://github.com/kwea123/ngp_pl): Great Instant-NGP implementation in PyTorch-Lightning! Background model and GUI supported.
    - [Welcome to ⚡ PyTorch Lightning — PyTorch Lightning 1.9.5 documentation](https://lightning.ai/docs/pytorch/1.9.5/)
- [Instant-NSR](https://github.com/zhaofuq/Instant-NSR): NeuS implementation using multiresolution hash encoding.

<!-- more -->

<iframe frameborder="0" style="width:100%;height:593px;" src="https://viewer.diagrams.net/?highlight=0000ff&edit=_blank&layers=1&nav=1&title=neus.drawio#R7Ztbj6M2FIB%2FTaTZhx2BMSR53Emm3YdWXWlUtfMUecEBdwhGxkyS%2FvraxA4QkyxNNzHZ5mE0%2BIC5nMN3fC5k5M1Wm58ZypNfaYTTEXCizcibjwAAkzEQ%2F6Rku5O4AQx2kpiRSMlqwQv5Gyuho6QliXDROpBTmnKSt4UhzTIc8pYMMUbX7cOWNG1fNUcxNgQvIUpN6R8k4slOOgHjWv4ZkzjRV3aD6W7PCumD1ZMUCYrouiHynkfejFHKd1urzQynUntaL7t5Px3Zu78xhjPeZ4Lz%2B5f5b1%2FYxzmcv7LPPA3nDH301b3xrX5gHInnV0PKeEJjmqH0uZY%2BMVpmEZZndcSoPuYXSnMhdIXwL8z5VhkTlZwKUcJXqdqLN4T%2FKac%2F%2Bmr02tgz36gzV4OtHmScbRuT5PC1ua%2BeVo30vCXNuLoRNxDj3fPKhzyqRiUqaMlCfEJ3%2BnVELMb8xHFgb2yBCaYrLO5PzGM4RZy8t%2B8Dqdc13h9XW1RsKKP%2BCwOr876jtFRX4gyRDLPHJeGG8WvTSmusE8LxS44qLawF4G0zdqv2HTOON6eVaypDT4DqhVTuwlfsrGv0XEfJkgZ20LmQ%2BoI7H2fzAXry4dnkAxh8iJVkSeKS4QXNOVkJ3bDi4cMJVBw7qOgVUq%2BsoIOV8TVZmd5ZOZsVrycrY5useAYrNFtUy8kC5zRMFoW4%2F1OLyjBI2ceW1kiBhiKXlMlXSWpRTs70oDC0KfTA2yorOKNveEZTcQ5vntFMgrUkaXogQimJMzEMhd6wkD9JrRIR6n5SO1Ykiioqu2zUtuIFzAQO%2FNm0w0p%2Bh5XApazkgrtDO9uhjXs6NNfpfimu49HGxz3aV8RvxqN5jm2PNjni0Sotao%2B2U2yEOEopioQL%2BvF9m15yxh0GCq7qzLy7Mzs%2F03f6erMjr8GVcn3HoLDA6fJRElfgG3BjgW035t7rYf%2BBkr4FMRdapcQsiVWU5AznjIa4KKolavC0QOsJvzu503I%2BLbAvLVZzftfMVctc8IEXKxqVKRYRMs6Hz8rEOiuBoUdDa0WCcrkZlizdPjEUvskX41vqa9fsL6DMSbsm37VKgw5V7uPb71%2BzhXe%2Fc77f6Z2ZT636HTM1r9JHksWVy7mBirwPbTsdz2rS5zYwqaH5FigtTGpqLIAy7QkKsFrC0rfZrGHpvtVC%2FNFFzFB0A7gE1otYntV15cZxAX1rJMdeiyv1e80ayVcRaK0RGyIj%2B8LucBgZUIXE6clIO%2FZyLTLS%2B5Mh3yojZoWkXlIGGnzB6dBI0b3KOynnkOL3JAVazVL0bbYbiGJEInF1OuzvIvZFEZ2veB3ITK%2BJDDRrIndkeiPTN7H3rBYUQUfPvRRwONW202CnWmlUA3m2a8eTaDPApcc%2FKJB1NHv3zFyFI2DmhI12vNSi0OjooDuPs3KFGeL4QUxrdOkfPpgq%2F%2FEa9TrgDto%2BEXZ9heR12PJijXs4oMb9zflEr7dPtBpGQKtJ1VmFB7dpYauFh942BnY%2FnjXXPelnxdpXPKI8x1n0ILYHuLodRImd35pdN0oc35xHHBAtfavadmExAxgjqZLEaIIGiM1BUAg6kqvrBoXQ%2FEaTLCulyQZ%2BFX7rCHwRJjh8WxAZoonh%2Fyf8C%2FyDlHjsWw7%2FfNfQ%2FtCd3XDCP9i7hWfV20HT27WrSEP%2BFv0wPoBdXe%2FvFB%2BIYf176Wpf42fn3vM%2F"></iframe>


使用了PyTorch Lightning库
# 文件结构：

```
├───configs  # 配置文件
│ nerf-blender.yaml  
│ nerf-colmap.yaml  
│ neus-blender.yaml  
│ neus-bmvs.yaml  
│ neus-colmap.yaml  
│ neus-dtu.yaml  
│  
├───datasets  # 数据集加载
│ blender.py  
│ colmap.py  
│  colmap_utils.py  
│  dtu.py  
│  utils.py  
│  __init__.py  
│  
├───models  # model的神经网络结构和model的运算
│  base.py  
│  geometry.py  
│  nerf.py  
│  network_utils.py  
│  neus.py  
│  ray_utils.py  
│  texture.py  
│  utils.py  
│  __init__.py  
│  
├───scripts  # 自定义数据集时gen_poses+run_colmap生成三个bin文件['cameras', 'images', 'points3D']
│ imgs2poses.py  
│  
├───systems  # model模型加载和训练时每步的操作
│  base.py  
│  criterions.py  
│  nerf.py  
│  neus.py  
│  utils.py  
│  __init__.py  
│  
└───utils  
│ callbacks.py  
│ loggers.py  
│ misc.py  
│ mixins.py  
│ obj.py  
│ __init__.py  
```

# fit流程伪代码

train.fit()结构的伪代码

>[LightningModule — PyTorch Lightning 1.9.5 documentation](https://lightning.ai/docs/pytorch/1.9.5/common/lightning_module.html#hooks)

```
def fit(self):
    if global_rank == 0:
        # prepare data is called on GLOBAL_ZERO only
        prepare_data()

    configure_callbacks()

    with parallel(devices):
        # devices can be GPUs, TPUs, ...
        train_on_device(model)


def train_on_device(model):
    # called PER DEVICE
    setup("fit")
    configure_optimizers()
    on_fit_start()

    # the sanity check runs here

    on_train_start()
    for epoch in epochs:
        fit_loop()
    on_train_end()

    on_fit_end()
    teardown("fit")


def fit_loop():
    on_train_epoch_start()

    for batch in train_dataloader():
        on_train_batch_start()

        on_before_batch_transfer()
        transfer_batch_to_device()
        on_after_batch_transfer()

        training_step()

        on_before_zero_grad()
        optimizer_zero_grad()

        on_before_backward()
        backward()
        on_after_backward()

        on_before_optimizer_step()
        configure_gradient_clipping()
        optimizer_step()

        on_train_batch_end()

        if should_check_val:
            val_loop()
    # end training epoch
    training_epoch_end()

    on_train_epoch_end()


def val_loop():
    on_validation_model_eval()  # calls `model.eval()`
    torch.set_grad_enabled(False)

    on_validation_start()
    on_validation_epoch_start()

    val_outs = []
    for batch_idx, batch in enumerate(val_dataloader()):
        on_validation_batch_start(batch, batch_idx)

        batch = on_before_batch_transfer(batch)
        batch = transfer_batch_to_device(batch)
        batch = on_after_batch_transfer(batch)

        out = validation_step(batch, batch_idx)

        on_validation_batch_end(batch, batch_idx)
        val_outs.append(out)

    validation_epoch_end(val_outs)

    on_validation_epoch_end()
    on_validation_end()

    # set up for train
    on_validation_model_train()  # calls `model.train()`
    torch.set_grad_enabled(True)
```

# datasets

## init

```
datasets = {}

def register(name):
    def decorator(cls):
        datasets[name] = cls
        return cls
    return decorator

def make(name, config): # dtu ,config.datasets
    dataset = datasets[name](config) # dataset = datasets['dtu'](config)
    return dataset

from . import blender, colmap, dtu
```

## dtu

数据集加载操作，返回一个DataLoader

- load_K_Rt_from_P，从P矩阵中恢复K、R和T

```python
def load_K_Rt_from_P(P=None):
    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose
```

- create_spheric_poses，创建一个球形的相机位姿，用于测试时的位姿和输出视频

ref: torch.cross: [叉积 - 维基百科，自由的百科全书 (wikipedia.org)](https://zh.wikipedia.org/wiki/%E5%8F%89%E7%A7%AF#%E7%9F%A9%E9%98%B5%E8%A1%A8%E7%A4%BA)

{% note warning %}
理解特征值和特征向量是什么？torch.linalg.eig
{% endnote %}

```python
# 生成球形相机姿态, 测试使用
def create_spheric_poses(cameras, n_steps=120): # camears: (n_images,3) , n_steps: 60
    center = torch.as_tensor([0.,0.,0.], dtype=cameras.dtype, device=cameras.device)
    cam_center = F.normalize(cameras.mean(0), p=2, dim=-1) * cameras.mean(0).norm(2) # normalize: 将输入张量归一化为单位范数
    # cam_center: torch.Size([3])
    eigvecs = torch.linalg.eig(cameras.T @ cameras).eigenvectors  # eig: 计算方阵的特征值和特征向量, .eigenvectors: 返回特征向量
    # eigvecs: torch.Size([3, 3])
    rot_axis = F.normalize(eigvecs[:,1].real.float(), p=2, dim=-1) # torch.Size([3]) 中间一列绕y轴旋转
    up = rot_axis   # torch.Size([3])
    rot_dir = torch.cross(rot_axis, cam_center) # cross: 计算两个向量的叉积, (3,)x(3,)=(3,)
    max_angle = (F.normalize(cameras, p=2, dim=-1) * F.normalize(cam_center, p=2, dim=-1)).sum(-1).acos().max()
    # max_angle: torch.Size([]) ,一个标量
    # 相机位置每个点与相机中心的夹角
    all_c2w = []
    for theta in torch.linspace(-max_angle, max_angle, n_steps):
        cam_pos = cam_center * math.cos(theta) + rot_dir * math.sin(theta) # torch.Size([3])
        l = F.normalize(center - cam_pos, p=2, dim=0) # torch.Size([3])
        s = F.normalize(l.cross(up), p=2, dim=0)    # torch.Size([3])
        u = F.normalize(s.cross(l), p=2, dim=0)   # torch.Size([3])
        c2w = torch.cat([torch.stack([s, u, -l], dim=1), cam_pos[:,None]], axis=1) # (3,4)
        all_c2w.append(c2w) # (n_steps, 3, 4)

    all_c2w = torch.stack(all_c2w, dim=0)   # (n_steps, 3, 4)
```

- DTUDatasetBase, `setup(self,config,split)`
    - **load** cameras_file and `imread(/root_dir/image/*.png)` 
        - `cv2.imread : shape[0] is H , shape[1] is W` is different frome PIL image
    - img_downscale or img_wh **to downscale**
    - n_images(数据集图片数量) `max([int(k.split('_')[-1]) for k in cams.keys()]) + 1`
    - for i in range(n_images): 
        - `P = (world_mat @ scale_mat)[:3,:4]`
        - `K, c2w = load_K_Rt_from_P(P)`
        - `fx, fy, cx, cy = K[0,0] * self.factor, K[1,1] * self.factor, K[0,2] * self.factor, K[1,2] * self.factor`
            - `self.factor = w / W`
        - `directions = get_ray_directions(w, h, fx, fy, cx, cy)` 得到的光线方向i.e. rays_d为类似NeRF的计算方式，y与j反向，z远离物体的方向
            - self.directions append directions
        - c2w  to tensor float and for c2w, flip the sign of input camera coordinate yz
            - c2w_ = c2w.clone
            - `c2w_[:3,1:3] *= -1.` because Neus DTU data is different from blender or blender
            - all_c2w append c2w_
        - if train or val
            - open i:06d.png image (PIL image) (size : w,h)
            - resize to w,h by Image.BICUBIC
            - TF(torchvision.transforms.functional).to_tensor() : CHW
                - `CHW.permute(1, 2, 0)[...,:3]` : HWC
            - open mask and covert('L') , resize , to_tensor
            - all_fg_mask append mask
            - all_images append img
    - all_c2w : stack all_c2w
    - if test
        - all_c2w = 创建一个球形相机位姿create_spheric_poses
        - all_images = zeros(n_test_traj_steps,h,w,3) dtype = torch.float32
        - all_fg_masks = zeros(n_test_traj_steps,h,w) dtype = torch.float32
        - `directions = directions[0]`
    - else 
        - all_images = stack all_images
        - all_fg_masks = stack al_images
        - directions = stack self.directions
    - .float().to(self.rank) 
        - self.rank = get_rank()  = 0 ,1 ,2 ... gpu序号
    - 
- DTUDataset 继承Dataset和DTUDatasetBase
    - `__init__ , __len__ , __getitem__`
- DTUIterableDataset 继承IterableDataset 和 DTUDatasetBase
    - `__init__ , __iter__`
- DTUDataModule 继承pl.LightningDataModule
    - @datasets.register('dtu')
    - `__init__(self,config)`
    - setup(self,stage)
        - `stage in [None , 'fit']` : train_dataset = DTUIterableDataset(self.config,'train')
        - `stage in [None , 'fit' , 'validate']` : val_dataset = DTUDataset(self.config, self.config.get('val_split','train'))
        - `stage in [None , 'test']` : test_dataset = DTUDataset(self.config , self.config.get('test_split','test'))
        - `stage in [None , 'predict']` : predict_dataset = DTUDataset(self.config, 'train')
    - prepare_data
    - general_loader(self,dataset,batch_size)
        - return DataLoader(dataset,num_workers=os.cpu_count(),batch_size,pin_memory=True,sampler=None)
    - train_dataloader(self)
        - return self.general_loader(self.train_dataset,batch_size=1)
    - val_dataloader(self)
        - return self.general_loader(self.val_dataset,batch_size=1)
    - test_dataloader(self)
        - return self.general_loader(self.test_dataset,batch_size=1)
    - predict_dataloader(self)
        - return self.general_loader(self.predict_dataset,batch_size=1)

# models

## init

`@models.register('neus')` 修饰器的作用：
- 主要是为了实例化NeuSModel()的同时，在models字典中同时存入一个NeuSModel()值，对应的key为'neus'

当运行 `neus_model = NeuSModel()` 时，即例如运行`self.texture = models.make(self.config.texture.name, self.config.texture)`时 ，会运行`neus_model = register('neus')(NeusModel)`
返回给neus_model的值为decorator(cls) 函数的返回值，即NeusModel


```
models = {}

def register(name):
    def decorator(cls):
        models[name] = cls
        return cls
    return decorator

def make(name, config):
    model = models[name](config)
    return model

from . import nerf, neus, geometry, texture
```

## base

BaseModel ， 继承nn.Module
- `__init__(self,config)`
    - self.confg = config , self.rank = get(rank)
    - self.setup()
        - 如果有config.weights，则load_state_dict(torch.load(config.weights))
- setup()
- update_step(self,epoch,global_step)
- train(self,mode=True)
    - return super().train(mode=mode)
- eval(self)
    - return super().eval()
- regularizations(self,out)
    - return {}
- @torch.no_gard() export(self,export_config)
    - return {}

其他model需要继承于BaseModel

## neus

Neus中的两个网络

### VarianceNetwork 继承nn.Module

sigmoid函数的s参数在训练中变化

- `__init__(self,config)`
    - self.config, self.init_val
    - self.register_parameter来自nn.Module注册一个参数
    - if self.modulate
        - True: mod_start_steps, reach_max_steps, max_inv_s
        - False: none
- @property 将该函数变为类的属性: inv_s()
    - $val = e^{variance * 10.0}$
    - if self.modulate and do_mod
        - val = val.clamp_max(mod_val)
    - return val
- forward(self,x)
    - `return torch.ones([len(x), 1], device=self.variance.device) * self.inv_s` 
        - 输入长度x1大小的inv_s
- update_step(self,epoch,global_step)
    - if self.modulate
        - ...

### NeuSModel 继承BaseModel
@models.register('neus')
{% note info %}
得到经过网络渲染后的一系列参数: 
position: n_samples x 3

```
forward_:
{**out,
**{k + '_bg': v for k, v in out_bg.items()},
**{k + '_full': v for k, v in out_full.items()}
}

forward:
**{**out,
**{k + '_bg': v for k, v in out_bg.items()},
**{k + '_full': v for k, v in out_full.items()}
}
+ inv_s
```

- out: 
    - rgb, normal : n_rays x 3
    - opacity=权重累加, depth , rays_valid: opacity>0 : n_rays x 1
    - `num_samples: torch.as_tensor([len(t_starts)],dtype = torch.int32,device=rays.device)`
    - if self.training:
        - update: 
            - sdf, : n_samples x 1 
            - sdf_grad,  : n_samples x 3 
            - (weights, midpoints, dists, ray_indices).view(-1) : n_samples x 1

and
- if learned_background:
    - out_bg:
        - rgb, opacity, depth, rays_valid, num_samples
        - if self.training:
            - update: (weights, midpoints, intervals, ray_indices).view(-1)
- else: rgb = None, num_samples = 0, rays_valid = 0

and
- out_full 
    - rgb: `out_rgb + out_bg_rgb * (1.0 - out_opacity)` , n_rays x 1 
    - num_samples: out_num + out_bg_num , n_samples + n_samples_bg
    - rays_valid: out_valid + out_bg_valid , n_rays x 1

{% endnote %}


- setup
    - self.geometry
    - self.texture
    - self.geometry.contraction_type
    - if self.config.learned_background
        - self.geometry_bg
        - self.texture_bg
        - self.geometry_bg.contraction_type
        - self.near_plane_bg, self.far_plane_bg
        - self.cone_angle_bg = $10^{\frac{log_{10}(far.plane.bg)}{num.samples.per.ray.bg}}-1 = 10^{\frac{log_{10}(10^{3})}{64}}-1$
    - self.variace = VarianceNetwork(self.config.variance)
    - self.register_buffer('scene_aabb')
        - 即将 self.scene_aabb放在模型缓冲区，可以与参数一起保存，对这个变量进行优化
    - if self.config.grid_prune 使用nerfacc中整合的InstantNGP中的占据网格，跳过空间中空白的部分
        - self.occupancy_grid = OccupancyGrid(roi_aabb, resolution=128 , contraction_type=AABB)
        - if self.learned_background：
            - self.occupancy_grid_bg = OccupancyGrid(roi_aabb, resolution=256 , contraction_type=UN_BOUNDED_SPHERE)
    - self.randomized = true
    - self.background_color = None
    - self.render_step_size = $1.732 \times 2 \times \frac{radius}{num.samples.per.ray}$

---
- update_step(self,epoch,global_step)
    - update_module_step(m,epoch,global_step)
        - m: self.geometry, self.texture,self.variance
            - if learned_background self.geometry_bg, self_texture_bg
    - cos_anneal_end = config.cos_anneal_end = 20000
    - if cos_anneal_end == 0: self.cos_anneal_end = 1.0
        - else :min(1.0, global_step / cos_anneal_end)
    - occ_eval_fn(x)  `x: Nx3`
        - sdf = self.geometry(x,...)
        - `inv_s = self.variance(torch.zeros([1,3]))[:,:1].clip(1e-6,1e6)`
        - `inv_s = inv_s.expand(sdf.shape[0],1)`
        - `estimated_next_sdf = sdf[...,None] - self.render_step_size * 0.5`
            - $next = sdf - 1.732 \times 2 \times \frac{radius}{n.samples.perray} \cdot 0.5 = sdf - cos \cdot dist \cdot 0.5$
                - $cos = 2 \cdot \sqrt{3}$
        - `estimated_prev_sdf = sdf[...,None] + self.render_step_size * 0.5`
        - `prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)`
        - `next_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)`
        - `p = prev_cdf - next_cdf , c = prev_cdf`
        - `alpha = ((p + 1e-5) / (c + 1e-5)).view(-1, 1).clip(0.0, 1.0)`  Nx1
        - return alpha
    - occ_eval_fn_bg(x)
        - `density, _ = self.geometry_bg(x)`
        - `return density[...,None] * self.render_step_size_bg`
    - if self.training(在nn.Module中可以这样判断是否为训练模式) and self.grid_prune
        - self.occupancy_grid.every_n_step ：nerfacc的占据网格每n步更新一次
        - if learned_background: self.occupancy_grid_bg.every_n_step

---
- isosurface：判断是否等值面
    - mesh = self.geometry.isosurface()
    - return mesh

---
- get_alpha：获取$\alpha$值
    - `inv_s = self.variance(torch.zeros([1,3]))[:,:1].clip(1e-6,1e6)`
    - `inv_s = inv_s.expand(sdf.shape[0],1)`
    - `true_cos = (dirs * normal).sum(-1, keepdim=True)`
    - `iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - self.cos_anneal_ratio)+F.relu(-true_cos) * self.cos_anneal_ratio)`
    - `estimated_next_sdf = sdf[...,None] + iter_cos * dists.reshape(-1, 1) * 0.5`
    - `estimated_prev_sdf = sdf[...,None] - iter_cos * dists.reshape(-1, 1) * 0.5`
    - `prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)`
    - `next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)`
    - `p = prev_cdf - next_cdf`
    - `c = prev_cdf`
    - `alpha = ((p + 1e-5) / (c + 1e-5)).view(-1).clip(0.0, 1.0)` N,

___

- forward_bg_：背景的输出
    - `n_rays = rays.shape[0]`, `rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]`
    - sigma_fn(t_starts, t_ends, ray_indices) 
        - ref: [Volumetric Rendering — nerfacc 0.3.5 documentation](https://www.nerfacc.com/en/v0.3.5/apis/rendering.html)
        - `density, _ = self.geometry_bg(positions)`
        - return `density[...,None]`
    - `_, t_max = ray_aabb_intersect(rays_o, rays_d, self.scene_aabb)`
    - `near_plane = torch.where(t_max > 1e9, self.near_plane_bg, t_max)`
        - if t_max > 1e9, near_plane = self.near_plane_bg, else near_plane = t_max
    - with torch.no_grad():
        - ray_indices, t_starts, t_ends = ray_marching()
        - ref:[Volumetric Rendering — nerfacc 0.3.5 documentation](https://www.nerfacc.com/en/v0.3.5/apis/rendering.html)
    - ray_indices = ray_indices.long() 为`N_rays`
    - t_origins = rays_o[ray_indices] `N_rays x 3`
    - t_dirs = rays_d[ray_indices] `N_rays x 3`
    - midpoints = (t_starts + t_ends) / 2.`n_samples x 1`
    - positions = t_origins + t_dirs * midpoints  `n_samples x 3`
    - intervals = t_ends - t_starts 为`n_samples x 1`
        -  ***n_samples = N_rays1 * n_samples_ray1 + N_rays2 * n_samples_ray2 + ...***
    - density, feature = self.geometry_bg(positions)
        - density : n_samples x 1
        - feature : n_samples x feature_dim
    - rgb = self.texture_bg(feature, t_dirs)
        - rgb: n_samples x 3
    - weights = render_weight_from_density(t_starts, t_ends, density[...,None], ray_indices=ray_indices, n_rays=n_rays)
        - 从密度中得到权重: $w_i = T_i(1 - exp(-\sigma_i\delta_i)), \quad\textrm{where}\quad T_i = exp(-\sum_{j=1}^{i-1}\sigma_j\delta_j)$
        - ref: [nerfacc.render_weight_from_density — nerfacc 0.3.5 documentation](https://www.nerfacc.com/en/v0.3.5/apis/generated/nerfacc.render_weight_from_density.html#nerfacc.render_weight_from_density)
        - n_samples x 1
    - opacity = accumulate_along_rays(weights, ray_indices, values=None, n_rays=n_rays)
        - ref: [nerfacc.accumulate_along_rays — nerfacc 0.3.5 documentation](https://www.nerfacc.com/en/v0.3.5/apis/generated/nerfacc.accumulate_along_rays.html#nerfacc.accumulate_along_rays)
        - n_rays, 1
    - depth = accumulate_along_rays(weights, ray_indices, values=midpoints, n_rays=n_rays)
        - n_rays, 1
    - comp_rgb = accumulate_along_rays(weights, ray_indices, values=rgb, n_rays=n_rays)
        - n_rays , 3
    - comp_rgb = comp_rgb + self.background_color * (1.0 - opacity)
        - n_rays , 3
    - return out 

```python
out = {
    'comp_rgb': comp_rgb, # n_rays, 1
    'opacity': opacity, # n_rays, 1
    'depth': depth, # n_rays, 1
    'rays_valid': opacity > 0, # n_rays, 1
    'num_samples': torch.as_tensor([len(t_starts)], dtype=torch.int32,device=rays.device) # n_samples
}

if self.training:
    out.update({
        'weights': weights.view(-1), #  n_samples x 1
        'points': midpoints.view(-1), # n_samples x 1
        'intervals': intervals.view(-1), # n_samples x 1
        'ray_indices': ray_indices.view(-1) # n_samples
    })
```

---
- forward_：前景输出
    - `n_rays = rays.shape[0]`, `rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]`
    - with torch.no_grad():
        - ray_indices, t_starts, t_ends = ray_marching(...)
        - ref:[Volumetric Rendering — nerfacc 0.3.5 documentation](https://www.nerfacc.com/en/v0.3.5/apis/rendering.html)
    - ray_indices = ray_indices.long()
    - t_origins = rays_o[ray_indices]
    - t_dirs = rays_d[ray_indices]
    - midpoints = (t_starts + t_ends) / 2.
    - positions = t_origins + t_dirs * midpoints
    - dists = t_ends - t_starts
    - sdf, sdf_grad, feature = self.geometry(positions, with_grad=True, with_feature=True)
        - sdf : n_samples x 1 
        - sdf_grad: n_samples x 3
        - feature: n_samples x feature_dim
    - normal = F.normalize(sdf_grad, p=2, dim=-1) 法向量：将sdf的梯度归一化
        - normal: n_samples x 3
    - alpha = self.get_alpha(sdf, normal, t_dirs, dists)[...,None]
        - n_samples x 1
    - rgb = self.texture(feature, t_dirs, normal)
        - n_samples x 3
    - weights = render_weight_from_alpha(alpha, ray_indices=ray_indices, n_rays=n_rays)
        - 从$\alpha$中得到权重：$w_i = T_i\alpha_i, \quad\textrm{where}\quad T_i = \prod_{j=1}^{i-1}(1-\alpha_j)$
        - ref: [nerfacc.render_weight_from_alpha — nerfacc 0.3.5 documentation](https://www.nerfacc.com/en/v0.3.5/apis/generated/nerfacc.render_weight_from_alpha.html#nerfacc.render_weight_from_alpha)
        - n_samples x 1
    - opacity = accumulate_along_rays(weights, ray_indices, values=None, n_rays=n_rays)
        - n_rays, 1
    - depth = accumulate_along_rays(weights, ray_indices, values=midpoints, n_rays=n_rays)
        - n_rays, 1
    - comp_rgb = accumulate_along_rays(weights, ray_indices, values=rgb, n_rays=n_rays)
        - n_rays, 3
    - comp_normal = accumulate_along_rays(weights, ray_indices, values=normal, n_rays=n_rays)
        - n_rays, 3
    - comp_normal = F.normalize(comp_normal, p=2, dim=-1)
        - n_rays, 3
    - return `{**out, **{k + '_bg': v for k, v in out_bg.items()}, **{k + '_full': v for k, v in ut_full.items()}}`

```python
out = {
    'comp_rgb': comp_rgb, # n_rays, 3
    'comp_normal': comp_normal, # n_rays, 3
    'opacity': opacity, # n_rays, 1
    'depth': depth, # n_rays, 1
    'rays_valid': opacity > 0, # n_rays, 1
    'num_samples': torch.as_tensor([len(t_starts)], dtype=torch.int32, device=rays.device) #n_samples
}

if self.training:
    out.update({
        'sdf_samples': sdf, # n_samples , 1
        'sdf_grad_samples': sdf_grad, # n_samples , 3
        'weights': weights.view(-1), # n_samples , 1
        'points': midpoints.view(-1), # n_samples , 1
        'intervals': dists.view(-1), # n_samples , 1
        'ray_indices': ray_indices.view(-1)   # n_samples             
})

if self.config.learned_background:
    out_bg = self.forward_bg_(rays)
else:
    out_bg = {
        'comp_rgb': self.background_color[None,:].expand(*comp_rgb.shape),
        'num_samples': torch.zeros_like(out['num_samples']),
        'rays_valid': torch.zeros_like(out['rays_valid'])
    }

out_full = {
    'comp_rgb': out['comp_rgb'] + out_bg['comp_rgb'] * (1.0 - out['opacity']),
    'num_samples': out['num_samples'] + out_bg['num_samples'],
    'rays_valid': out['rays_valid'] | out_bg['rays_valid']
}
```

---

- forward(rays)
    - if self.training 
        - out = self.forward_(rays)
    - else
        - out = chunk_batch(self.forward_, self.config.ray_chunk, True, rays)
    - return `{**out, 'inv_s': self.variance.inv_s}`

- train(self, mode=True)
    - self.randomized = `mode and self.config.randomized`
    - return super().train(mode=mode)
- eval
    - self.randomized = False
    - return super().eval()

- regularizations
    - losses = {}
    - losses.update(self.geometry.regularizations(out))
    - losses.update(self.texture.regularizations(out))
    - return losses


@torch.no_grad()
- export：导出带有texture的mesh

```python
@torch.no_grad()
def export(self, export_config):
    mesh = self.isosurface()
    if export_config.export_vertex_color:
        _, sdf_grad, feature = chunk_batch(self.geometry, export_config.chunk_size, False, mesh['v_pos'].to(self.rank), with_grad=True, with_feature=True)
        normal = F.normalize(sdf_grad, p=2, dim=-1)
        rgb = self.texture(feature, -normal, normal) # set the viewing directions to the normal to get "albedo"
        mesh['v_rgb'] = rgb.cpu()
    return mesh
```


## network_utils

{% note info %}
各种编码方式和MLP网络
- VanillaFrequency, ProgressiveBandHashGrid, tcnn.Encoding
- VanillaMLP, tcnn.Network
可以使用的方法：
- get_encoding, get_mlp, get_encoding_with_network
{% endnote %}

- VanillaFrequency 继承nn.Module
    - `__init__(self, in_channels, config)`
        - self.N_freqs 即L
        -  self.in_channels, self.n_input_dims = in_channels, in_channels
        - `self.funcs = [torch.sin, torch.cos]`
        - `self.freq_bands = 2**torch.linspace(0, self.N_freqs-1, self.N_freqs)`
        - self.n_output_dims = self.in_channels * (len(self.funcs) * self.N_freqs) = 3x2xL
        - self.n_masking_step = config.get('n_masking_step', 0)
        - self.update_step 每步开始前都需要更新出mask
    - forward(self,x)
        - out = []
        - for freq , mask in zip(self.freq_bands, self.mask):
            - for func in self.funcs:
                - `out += [func(freq*x) * mask]`    
        - return torch.cat(out, -1)   
    - update_step(self,epoch,global_step)
        - if self.n_masking_step <= 0 or global_step is None:
            - self.mask = torch.ones(self.N_freqs, dtype=torch.float32) 与L相同形状的全1张量
        - else:
            - self.mask = (1. - torch.cos(math.pi * (global_step / self.n_masking_step * self.N_freqs - torch.arange(0, self.N_freqs)).clamp(0, 1))) / 2.
                - mask = $\left(1-cos\left(\pi \cdot \left(\frac{global.step \cdot L}{n.masking.step}-arrange\right).clamp\right)\right) \cdot 0.5$
            - rank_zero_debug(f'Update mask: {global_step}/{self.n_masking_step} {self.mask}')


Vanilla：最初始的
即NeRF中的频率编码方式
$\gamma(p)=\left(\sin \left(2^{0} \pi p\right), \cos \left(2^{0} \pi p\right), \cdots, \sin \left(2^{L-1} \pi p\right), \cos \left(2^{L-1} \pi p\right)\right)$

- ProgressiveBandHashGrid，继承nn.Module
    - `__init__(self, in_channels,config)`
        - self.n_input_dims = in_channels
        - encoding_config = config.copy()
        - encoding_config['otype'] = 'HashGrid'
        - with torch.cuda.device(get_rank()):
            - self.encoding = tcnn.Encoding(in_channels, encoding_config) 使用哈希编码
        - self.n_output_dims = self.encoding.n_output_dims
        - self.n_level = config['n_levels']，分辨率个数
        - self.n_features_per_level = config['n_features_per_level']，特征向量维数
        - self.start_level, self.start_step, self.update_steps = config['start_level'], config['start_step'], config['update_steps']
        - self.current_level = self.start_level
        - self.mask = torch.zeros(self.n_level * self.n_features_per_level, dtype=torch.float32, device=get_rank())
    - forward(self,x)
        - enc = self.encoding(x)
        - enc = enc * self.mask ,第一个step，mask为0，之后每过update_steps，更新一次mask
        - return enc
    - update_step(self,epoch,global_step)
        -  current_level = min(self.start_level + max(global_step - self.start_step, 0) // self.update_steps, self.n_level)
            - min(1+max(global_step-0,0)//update_steps, 16)
        - if current_level > self.current_level:
            - rank_zero_debug(f'Update current level to {current_level}')
        - self.current_level = current_level
        - `self.mask[:self.current_level * self.n_features_per_level] = 1.`
            - mask从0到(当前分辨率x特征向量维数) 置为1

- CompositeEncoding，继承nn.Module
    - `__init__(self, encoding , include_xyz = True, xyz_scale=1 , xyz_offset=0)`
        - self.encoding = encoding
        - self.include_xyz, self.xyz_scale, self.xyz_offset = include_xyz, xyz_scale, xyz_offset
        - self.n_output_dims = int(self.include_xyz) * self.encoding.n_input_dims + self.encoding.n_output_dims
            - $o.dim = int(TorF) \cdot n.idim +n.odim$
    - `forward(self,x,*args)`
        - return `self.encoding(x, *args) if not self.include_xyz else torch.cat([x * self.xyz_scale + self.xyz_offset, self.encoding(x, *args)], dim=-1)`
            - if include_xyz: `torch.cat([x * self.xyz_scale + self.xyz_offset, self.encoding(x, *args)], dim=-1)` 将输入x变为2x-1，并与编码后的输入cat起来
            - else : `self.encoding(x, *args)`
    - update(self, epoch, global_step)
        - update_module_step(self.encoding, epoch, global_step)

{% note info %}
config_to_primitive在utils/misc.py中，`return OmegaConf.to_container(config, resolve=resolve)`
由于OmegaConf config objects很占用内存，因此使用to_container转化为原始的容器，如dict。如果resolve值设置为 True，则将在转换期间解析内插`${foo}`。
{% endnote %}


- get_encoding(n_input_dims, config)
    - `if config.otype ==  'VanillaFrequency':`
        - encoding = VanillaFrequency(n_input_dims, config_to_primitive(config))
    - `elif config.otype == 'ProgressiveBandHashGrid':`
        - encoding = ProgressiveBandHashGrid(n_input_dims, config_to_primitive(config))
    - else:
        - with torch.cuda.device(get_rank()):
            - encoding = tcnn.Encoding(n_input_dims, config_to_primitive(config)) 
    - `encoding = CompositeEncoding(encoding, include_xyz=config.get('include_xyz', False), xyz_scale=2., xyz_offset=-1.)`
    - return encoding

---

- VanillaMLP , 继承nn.Module
    - NeRF中的MLP
    - `__init__(self,dim_in,dim_out,config)`
        - 一共n_hidden_layers个隐藏层，每个隐藏层有n_neurons个节点
        - `Sequential(*self.layers)`将每层都加入ModuleList中，并在内部实现forward，可以不写forward
            - ref: [详解PyTorch中的ModuleList和Sequential - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/75206669)
        - get_activation：utils中的激活函数方法，根据不同的output_activation选择不同的激活函数

```python
self.n_neurons, self.n_hidden_layers = config['n_neurons'], config['n_hidden_layers']
self.sphere_init, self.weight_norm = config.get('sphere_init', False), config.get('weight_norm', False)
self.sphere_init_radius = config.get('sphere_init_radius', 0.5)
self.layers = [self.make_linear(dim_in, self.n_neurons, is_first=True, is_last=False), self.make_activation()]
for i in range(self.n_hidden_layers - 1):
    self.layers += [self.make_linear(self.n_neurons, self.n_neurons, is_first=False, is_last=False), self.make_activation()]
self.layers += [self.make_linear(self.n_neurons, dim_out, is_first=False, is_last=True)]
self.layers = nn.Sequential(*self.layers)
self.output_activation = get_activation(config['output_activation'])
```

- VanillaMLP 接上
    - forward(self,x)，Sequential可以直接使用，而不需构建一个循环从ModuleList中依次执行
        - x = self.layers(x.float())
        - x = self.output_activation(x)
        - return x
    - make_linear(self,dim_in,dim_out,is_first,is_last)
        - layer = nn.Linear(dim_in, dim_out, bias=True) # network without bias will degrade quality
        - if self.sphere_init: 初始化每层的权重和偏置(常数或者正态分布)
            - if is_last:
                - torch.nn.init.constant_(layer.bias, -self.sphere_init_radius)
                - torch.nn.init.normal_(layer.weight, mean=math.sqrt(math.pi) / math.sqrt(dim_in), std=0.0001)
            - elif is_first:
                - torch.nn.init.constant_(layer.bias, 0.0)
                - torch.nn.init.constant_(layer.weight[:, 3:], 0.0)
                - torch.nn.init.normal_(layer.weight[:, :3], 0.0, math.sqrt(2) / math.sqrt(dim_out))
            - else:
                - torch.nn.init.constant_(layer.bias, 0.0)
                - torch.nn.init.normal_(layer.weight, 0.0, math.sqrt(2) / math.sqrt(dim_out))
        - else:
            - torch.nn.init.constant_(layer.bias, 0.0)
            - torch.nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
        - if self.weight_norm: 
            - layer = nn.utils.weight_normal(layer)
        - return layer
    - make_activation
        - if self.sphere_init:
            - return nn.Softplus(beta=100)
        - else:
            - return nn.ReLU(inplace=True)

---

- sphere_init_tcnn_network(n_input_dims, n_output_dims, config, network)
    - rank_zero_debug('Initialize tcnn MLP to approximately represent a sphere.')
    - padto = 16 if config.otype == 'FullyFusedMLP' else 8
    - n_input_dims = n_input_dims + (padto - n_input_dims % padto) % padto
        - $ni = ni + (padto - ni \% padto) \% padto$ 取余数
    - n_output_dims = n_output_dims + (padto - n_output_dims % padto) % padto
    - `data = list(network.parameters())[0].data`
    - `assert data.shape[0] == (n_input_dims + n_output_dims) * config.n_neurons + (config.n_hidden_layers - 1) * config.n_neurons**2`   
    - `new_data = []`

```python
# first layer
weight = torch.zeros((config.n_neurons, n_input_dims)).to(data)
torch.nn.init.constant_(weight[:, 3:], 0.0)
torch.nn.init.normal_(weight[:, :3], 0.0, math.sqrt(2) / math.sqrt(config.n_neurons))
new_data.append(weight.flatten())
# hidden layers
for i in range(config.n_hidden_layers - 1):
    weight = torch.zeros((config.n_neurons, config.n_neurons)).to(data)
    torch.nn.init.normal_(weight, 0.0, math.sqrt(2) / math.sqrt(config.n_neurons))
    new_data.append(weight.flatten())
# last layer
weight = torch.zeros((n_output_dims, config.n_neurons)).to(data)
torch.nn.init.normal_(weight, mean=math.sqrt(math.pi) / math.sqrt(config.n_neurons), std=0.0001)
new_data.append(weight.flatten())
new_data = torch.cat(new_data)
data.copy_(new_data)
```

- get_mlp(n_input_dims, n_output_dims, config)
    - `if config.otype == 'VanillaMLP':`
        - network = VanillaMLP(n_input_dims, n_output_dims, config_to_primitive(config))
    - `else:`
        - with torch.cuda.device(get_rank()):
            - network = tcnn.Network(n_input_dims, n_output_dims, config_to_primitive(config))
            - if config.get('sphere_init', False):
                - sphere_init_tcnn_network(n_input_dims, n_output_dims, config, network)
    - return network 返回一个model

- EncodingWithNetwork，继承nn.Module
    - `__init__(self,encoding,network)`
        - self.encoding, self.network = encoding, network
    - forward(self,x)
        - return self.network(self.encoding(x))
    - update_step(self, epoch, global_step)
        - update_module_step(self.encoding, epoch, global_step)
        - update_module_step(self.network, epoch, global_step)

- get_encoding_with_network(n_input_dims, n_output_dims, encoding_config, network_config)
    - `if encoding_config.otype in ['VanillaFrequency', 'ProgressiveBandHashGrid'] or network_config.otype in ['VanillaMLP']:`
        - encoding = get_encoding(n_input_dims, encoding_config)
        - network = get_mlp(encoding.n_output_dims, n_output_dims, network_config)
        - encoding_with_network = EncodingWithNetwork(encoding, network)
    - else:
        - with torch.cuda.device(get_rank()):
            - encoding_with_network = tcnn.NetworkWithInputEncoding(n_input_dims,n_output_dims,encoding_config,network_config)
    - return encoding_with_network

## geometry

{% note info %}
输入点的位置position经过MLP网络得到背景density, feature或者前景物体sdf, sdf_grad, feature
{% endnote %}


- contract_to_unisphere，根据contraction_type，将位置x缩放到合适大小
```python
def contract_to_unisphere(x, radius, contraction_type):
    if contraction_type == ContractionType.AABB:
        x = scale_anything(x, (-radius, radius), (0, 1))
    elif contraction_type == ContractionType.UN_BOUNDED_SPHERE:
        x = scale_anything(x, (-radius, radius), (0, 1))
        x = x * 2 - 1  # aabb is at [-1, 1]
        mag = x.norm(dim=-1, keepdim=True)
        mask = mag.squeeze(-1) > 1
        x[mask] = (2 - 1 / mag[mask]) * (x[mask] / mag[mask])
        x = x / 4 + 0.5  # [-inf, inf] is at [0, 1]
    else:
        raise NotImplementedError
    return x
```

将等值面三角网格化
- MarchingCubeHelper 继承nn.Module
    - `__init__(resolution, use_torch=True)`
        - self.resolution = resolution
        - self.use_torch = use_torch
        - self.points_range = (0, 1)
        - if self.use_torch:
            - import torchmcubes
            - self.mc_func = torchmcubes.marching_cubes
        - else:
            - import mcubes
            - self.mc_func = mcubes.marching_cubes
        - self.verts = None
    - grid_vertices()
        - if self.verts is None:
            - `x, y, z = torch.linspace(*self.points_range, self.resolution), torch.linspace(*self.points_range, self.resolution), torch.linspace(*self.points_range, self.resolution)`
                - `x: torch.Size([resolution])`
            - x, y, z = torch.meshgrid(x, y, z, indexing='ij')
                - `x: torch.Size([resolution, resolution, resolution])`
            - `verts = torch.cat([x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)], dim=-1).reshape(-1, 3)`
                - `verts: torch.Size([resolution ** 3, 3])`
            - self.verts = verts
        - return self.verts
    - forward(self,level,threshold=0.)
        - level = level.float().view(self.resolution, self.resolution, self.resolution)
        - if self.use_torch:
            - `verts, faces = self.mc_func(level.to(get_rank()), threshold)`
            - `verts, faces = verts.cpu(), faces.cpu().long()`
        - else:
            - `verts, faces = self.mc_func(-level.numpy(), threshold)` # transform to numpy
            - `verts, faces = torch.from_numpy(verts.astype(np.float32)), torch.from_numpy(faces.astype(np.int64))`
        - verts = verts / (self.resolution - 1.)
        - return {'v_pos': verts, 't_pos_idx': faces}

获得等值面的mesh网格，包括vertices和faces
- BaseImplicitGeometry，继承BaseModel
    - `__init__(self,config)`
        - `if self.config.isosurface is not None:`
            - `assert self.config.isosurface.method in ['mc', 'mc-torch']`
            - `if self.config.isosurface.method == 'mc-torch':`
                - `raise NotImplementedError("Please do not use mc-torch. It currently has some scaling issues I haven't fixed yet.")`
            - `self.helper = MarchingCubeHelper(self.config.isosurface.resolution, use_torch=self.config.isosurface.method=='mc-torch') `
        - self.contraction_type = None
        - self.radius = self.config.radius
    - forward_level(self,points)
        - raise NotImplementedError
    - isosurface_(self,vmin,vmax): 返回mesh

```python
def batch_func(x):
    x = torch.stack([
        scale_anything(x[...,0], (0, 1), (vmin[0], vmax[0])),
        scale_anything(x[...,1], (0, 1), (vmin[1], vmax[1])),
        scale_anything(x[...,2], (0, 1), (vmin[2], vmax[2])),
    ], dim=-1).to(self.rank)
    rv = self.forward_level(x).cpu() # 为 -density
    cleanup()
    return rv
# self.helper.grid_vertices(): torch.Size([resolution ** 3, 3])
level = chunk_batch(batch_func, self.config.isosurface.chunk, True, self.helper.grid_vertices())
mesh = self.helper(level, threshold=self.config.isosurface.threshold)
mesh['v_pos'] = torch.stack([
    scale_anything(mesh['v_pos'][...,0], (0, 1), (vmin[0], vmax[0])),
    scale_anything(mesh['v_pos'][...,1], (0, 1), (vmin[1], vmax[1])),
    scale_anything(mesh['v_pos'][...,2], (0, 1), (vmin[2], vmax[2]))
], dim=-1)
return mesh

in utils:
def scale_anything(dat, inp_scale, tgt_scale):
    if inp_scale is None:
        inp_scale = [dat.min(), dat.max()]
    dat = (dat  - inp_scale[0]) / (inp_scale[1] - inp_scale[0])
    dat = dat * (tgt_scale[1] - tgt_scale[0]) + tgt_scale[0]
    return dat
```

- BaseImplicitGeometry 接上
    - @torch.no_grad()
    - isosurface(self) 
        - if self.config.isosurface is None:
            - raise NotImplementedError
        - mesh_coarse = self.isosurface_((-self.radius, -self.radius, -self.radius), (self.radius, self.radius, self.radius))
        - vmin, vmax = mesh_coarse['v_pos'].amin(dim=0), mesh_coarse['v_pos'].amax(dim=0)
        - vmin_ = (vmin - (vmax - vmin) * 0.1).clamp(-self.radius, self.radius)
        - vmax_ = (vmax + (vmax - vmin) * 0.1).clamp(-self.radius, self.radius)
        - mesh_fine = self.isosurface_(vmin_, vmax_)
        - return mesh_fine  返回vertices和faces

背景几何：density和feature
@models.register('volume-density')
- VolumeDensity 继承BaseImplicitGeometry
    - setup()
        - self.n_input_dims = self.config.get('n_input_dims', 3)
        - self.n_output_dims = self.config.feature_dim
        - self.encoding_with_network = get_encoding_with_network(self.n_input_dims, self.n_output_dims, self.config.xyz_encoding_config, self.config.mlp_network_config)
    - forward(self,points) 根据编码方式和网络，得到density和feature
        - points = contract_to_unisphere(points, self.radius, self.contraction_type)
        - `out = self.encoding_with_network(points.view(-1, self.n_input_dims)).view(*points.shape[:-1], self.n_output_dims).float()`
        - density, feature = out[...,0], out
        - if 'density_activation' in self.config:
            - density = get_activation(self.config.density_activation)(density + float(self.config.density_bias))
        - if 'feature_activation' in self.config:
            - feature = get_activation(self.config.feature_activation)(feature)
        - return density, feature
    - forward_level(self,points) 根据编码方式和网络，得到-density，方便进行判断等值面isosurface
        - points = contract_to_unisphere(points, self.radius, self.contraction_type)
        - `density = self.encoding_with_network(points.reshape(-1, self.n_input_dims)).reshape(*points.shape[:-1], self.n_output_dims)[...,0]`
        - if 'density_activation' in self.config:
            - density = get_activation(self.config.density_activation)(density + float(self.config.density_bias))
        - return -density
    - update_step(self,epoch, global_step)
        - update_module_step(self.encoding_with_network, epoch, global_step)

前景物体几何：sdf, sdf_grad, feature
@models.register('volume-sdf')
- VolumeSDF
    - setup()
        - self.n_output_dims = self.config.feature_dim
        - encoding = get_encoding(3, self.config.xyz_encoding_config)
        - network = get_mlp(encoding.n_output_dims, self.n_output_dims, self.config.mlp_network_config)
        - self.encoding, self.network = encoding, network
        - self.grad_type = self.config.grad_type
    - forward(self, points, with_grad=True, with_feature=True)
        - `with torch.inference_mode(torch.is_inference_mode_enabled() and not (with_grad and self.grad_type == 'analytic'))`: 是否启用推理模式，当前为推理，并且没有grad，grad_type不是analytic
            - `with torch.set_grad_enabled(self.training or (with_grad and self.grad_type == 'analytic')):`
                - if with_grad and self.grad_type == 'analytic':
                - if not self.training:
                    - points = points.clone() # points may be in inference mode, get a copy to enable grad
                - points.requires_grad_(True)
            - points_ = points 初始位置
            - points = contract_to_unisphere(points, self.radius, self.contraction_type) **points normalized to (0, 1)**
            - `out = self.network(self.encoding(points.view(-1, 3))).view(*points.shape[:-1], self.n_output_dims).float()`
            - `sdf, feature = out[...,0], out`
            - if 'sdf_activation' in self.config: sdf激活
                - sdf = get_activation(self.config.sdf_activation)(sdf + float(self.config.sdf_bias))
            - if 'feature_activation' in self.config:
                - feature = get_activation(self.config.feature_activation)(feature)       
            - if with_grad: 求梯度的两种方法：自动微分or有限差分法
                - `if self.grad_type == 'analytic':` 
                    - grad = torch.autograd.grad(sdf, points_, grad_outputs=torch.ones_like(sdf),create_graph=True, retain_graph=True, only_inputs=True)[0]
                - `elif self.grad_type == 'finite_difference':`
                    - 有限差分得到$𝑓 ′ (𝑥) = (𝑓 (𝑥 + Δ𝑥) − (𝑓 𝑥 − Δ𝑥))/2Δ𝑥$，sdf对位置的梯度grad
        - `rv = [sdf]`
        - if with_grad:
            - rv.append(grad)
        - if with_feature:
            - rv.append(feature)
        - `rv = [v if self.training else v.detach() for v in rv]`
        - `return rv[0] if len(rv) == 1 else rv`

```python
有限差分法
eps = 0.001
points_d_ = torch.stack([
    points_ + torch.as_tensor([eps, 0.0, 0.0]).to(points_), # to(other): 返回一个与 Tensor other 具有相同 torch.dtype 和 torch.device 的 Tensor
    points_ + torch.as_tensor([-eps, 0.0, 0.0]).to(points_),
    points_ + torch.as_tensor([0.0, eps, 0.0]).to(points_),
    points_ + torch.as_tensor([0.0, -eps, 0.0]).to(points_),
    points_ + torch.as_tensor([0.0, 0.0, eps]).to(points_),
    points_ + torch.as_tensor([0.0, 0.0, -eps]).to(points_)
], dim=0).clamp(0, 1)
points_d = scale_anything(points_d_, (-self.radius, self.radius), (0, 1))
points_d_sdf = self.network(self.encoding(points_d.view(-1, 3)))[...,0].view(6, *points.shape[:-1]).float()
grad = torch.stack([
    0.5 * (points_d_sdf[0] - points_d_sdf[1]) / eps,
    0.5 * (points_d_sdf[2] - points_d_sdf[3]) / eps,
    0.5 * (points_d_sdf[4] - points_d_sdf[5]) / eps,
], dim=-1)
```

- VolumeSDF 接上
    - forward_level(self, points)
        - points = contract_to_unisphere(points, self.radius, self.contraction_type) # points normalized to (0, 1)
        - `sdf = self.network(self.encoding(points.view(-1, 3))).view(*points.shape[:-1], self.n_output_dims)[...,0]`
        - if 'sdf_activation' in self.config:
            - sdf = get_activation(self.config.sdf_activation)(sdf + float(self.config.sdf_bias))
        - return sdf
    - update_step(self, epoch , global_step)
        - update_module_step(self.encoding, epoch, global_step)
        - update_module_step(self.network, epoch, global_step)


## texture

{% note info %}
根据feature、dirs得到背景颜色
根据feature、dirs，以及normal得到前景颜色
{% endnote %}

前背景颜色值
@models.register('volume-radiance')
- VolumeRadiance，继承nn.Module
    - `__init__`
        - self.config = config
        - self.n_dir_dims = self.config.get('n_dir_dims', 3)
        - self.n_output_dims = 3
        - encoding = get_encoding(self.n_dir_dims, self.config.dir_encoding_config)
        - self.n_input_dims = self.config.input_feature_dim + encoding.n_output_dims
        - network = get_mlp(self.n_input_dims, self.n_output_dims, self.config.mlp_network_config) 
        - self.encoding = encoding
        - self.network = network
    - `forward(self, features, dirs, *args)`
        - dirs = (dirs + 1.) / 2. # (-1, 1) => (0, 1)
        - dirs_embd = self.encoding(dirs.view(-1, self.n_dir_dims))
        - `network_inp = torch.cat([features.view(-1, features.shape[-1]), dirs_embd] + [arg.view(-1, arg.shape[-1]) for arg in args], dim=-1)`
        - `color = self.network(network_inp).view(*features.shape[:-1], self.n_output_dims).float()`
        - if 'color_activation' in self.config:
            - color = get_activation(self.config.color_activation)(color)
        - return color
    - update_step(self, epoch, global_step)
        - update_module_step(self.encoding, epoch, global_step)
    - regularizations(self, out)
        - return {}

@models.register('volume-color') 
- VolumeColor，不使用编码方法
    - `__init__`
        - self.config = config
        - self.n_output_dims = 3
        - self.n_input_dims = self.config.input_feature_dim
        - network = get_mlp(self.n_input_dims, self.n_output_dims, self.config.mlp_network_config)
        - self.network = network
    - `forward(self, features, *args)`
        - `network_inp = features.view(-1, features.shape[-1])`
        - `color = self.network(network_inp).view(*features.shape[:-1], self.n_output_dims).float()`
        - if 'color_activation' in self.config:
            - color = get_activation(self.config.color_activation)(color)
        - return color
    - regularizations(self, out)
        - return {}

## ray_utils

- cast_rays

获取光线的方向(相机坐标下)
- get_ray_directions(W, H, fx, fy, cx, cy, use_pixel_centers=True)

```python
def get_ray_directions(W, H, fx, fy, cx, cy, use_pixel_centers=True):
    pixel_center = 0.5 if use_pixel_centers else 0 # 是否使用像素中心
    i, j = np.meshgrid(
        np.arange(W, dtype=np.float32) + pixel_center,
        np.arange(H, dtype=np.float32) + pixel_center,
        indexing='xy'
    )
    i, j = torch.from_numpy(i), torch.from_numpy(j)

    directions = torch.stack([(i - cx) / fx, -(j - cy) / fy, -torch.ones_like(i)], -1) # (H, W, 3)
    # 计算方式与NeRF相同
    return directions
```

获取光线
- get_rays(directions, c2w, keepdim=False):

```python
def get_rays(directions, c2w, keepdim=False):
    # Rotate ray directions from camera coordinate to the world coordinate
    # rays_d = directions @ c2w[:, :3].T # (H, W, 3) # slow?
    assert directions.shape[-1] == 3

    if directions.ndim == 2: # (N_rays, 3)
        assert c2w.ndim == 3 # (N_rays, 4, 4) / (1, 4, 4)
        rays_d = (directions[:,None,:] * c2w[:,:3,:3]).sum(-1) # (N_rays, 3)
        rays_o = c2w[:,:,3].expand(rays_d.shape)
    elif directions.ndim == 3: # (H, W, 3)
        if c2w.ndim == 2: # (4, 4)
            rays_d = (directions[:,:,None,:] * c2w[None,None,:3,:3]).sum(-1) # (H, W, 3)
            rays_o = c2w[None,None,:,3].expand(rays_d.shape)
        elif c2w.ndim == 3: # (B, 4, 4)
            rays_d = (directions[None,:,:,None,:] * c2w[:,None,None,:3,:3]).sum(-1) # (B, H, W, 3)
            rays_o = c2w[:,None,None,:,3].expand(rays_d.shape)

    if not keepdim:
        rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

    return rays_o, rays_d
```

## utils

分批使用func处理数据，并决定是否移动到cpu
- `chunk_batch(func, chunk_size, move_to_cpu, *args, **kwargs)`
    - B = None
    - for arg in args
        - if isinstance(arg, torch.Tensor):
            - `B = arg.shape[0]`
            - break
    - out = defaultdict(list)  将字典中同个key的多个value构成一个列表
        - ref: [(21条消息) python 字典defaultdict(list)_wanghua609的博客-CSDN博客](https://blog.csdn.net/weixin_38145317/article/details/93175217)
    - out_type = None
    - for i in range(0, B, chunk_size):
        - `out_chunk = func(*[arg[i:i+chunk_size] if isinstance(arg, torch.Tensor) else arg for arg in args], **kwargs)`
            - 使用func函数得到一批输出
        - if out_chunk is None:
            - continue
        - out_type = type(out_chunk)
        - if isinstance(out_chunk, torch.Tensor): 将out_chunk 变为字典
            - out_chunk = {0: out_chunk}
        - elif isinstance(out_chunk, tuple) or isinstance(out_chunk, list):
            - chunk_length = len(out_chunk)
            - out_chunk = {i: chunk for i, chunk in enumerate(out_chunk)}
        - elif isinstance(out_chunk, dict):
            - pass
        - else:
            - `print(f'Return value of func must be in type [torch.Tensor, list, tuple, dict], get {type(out_chunk)}.')`
            - exit(1)
        - for k, v in out_chunk.items():
            - v = v if torch.is_grad_enabled() else v.detach()
            - v = v.cpu() if move_to_cpu else v
            - `out[k].append(v)`
    - if out_type is None:
        - return
    - out = {k: torch.cat(v, dim=0) for k, v in out.items()}
    - if out_type is torch.Tensor:
        - return out[0]
    - `elif out_type in [tuple, list]`:
        - `return out_type([out[i] for i in range(chunk_length)])`
    - elif out_type is dict:
        - return out

将dat从inp缩放到tgt
- scale_anything(dat, inp_scale, tgt_scale): 
    - if inp_scale is None:
        - `inp_scale = [dat.min(), dat.max()]`
    - `dat = (dat  - inp_scale[0]) / (inp_scale[1] - inp_scale[0])`
    - `dat = dat * (tgt_scale[1] - tgt_scale[0]) + tgt_scale[0]`
    - return dat

激活函数
- get_activation(name)
    - if name is None:
        - return lambda x: x
    - name = name.lower() # lower： 将所有大写字符转换为小写
    - if name == 'none':  return lambda x: x
    - if name.startswith('scale'): 
        - scale_factor = float(name[5:])
        - return lambda x: x.clamp(0., scale_factor) / scale_factor
    - elif name.startswith('clamp'):
        - clamp_max = float(name[5:])
        - return lambda x: x.clamp(0., clamp_max)
    - elif name.startswith('mul'):
        - mul_factor = float(name[3:])
        - return lambda x: x * mul_factor
    - elif name == 'lin2srgb':`return lambda x: torch.where(x > 0.0031308, torch.pow(torch.clamp(x, min=0.0031308), 1.0/2.4)*1.055 - 0.055, 12.92*x).clamp(0., 1.)`
    - elif name == 'trunc_exp': return trunc_exp
    - elif name.startswith('+') or name.startswith('-'): return lambda x: x + float(name)
    - elif name == 'sigmoid':return lambda x: torch.sigmoid(x)
    - elif name == 'tanh': return lambda x: torch.tanh(x)
    - else:  return getattr(F, name)

# systems

## init

```
systems = {}

def register(name):
    def decorator(cls):
        systems[name] = cls
        return cls
    return decorator

def make(name, config, load_from_checkpoint=None):
    if load_from_checkpoint is None:
        system = systems[name](config)
    else:
        system = systems[name].load_from_checkpoint(load_from_checkpoint, strict=False, config=config)
    return system

from . import nerf, neus
```

## base

- BaseSystem，继承pl.LightningModule和SaverMixin
    - `__init__`
        - self.config = config
        - self.rank = get_rank()
        - self.prepare()
        - self.model = models.make(self.config.model.name, self.config.model)
    - prepare
        - pass
    - forward(self, batch)
        - raise NotImplementedError
    - C(self, value)

```python
C():
if isinstance(value, int) or isinstance(value, float):
    pass
else:
    value = config_to_primitive(value)
    if not isinstance(value, list):
        raise TypeError('Scalar specification only supports list, got', type(value))
    if len(value) == 3:
        value = [0] + value
    assert len(value) == 4
    start_step, start_value, end_value, end_step = value
    if isinstance(end_step, int):
        current_step = self.global_step
        value = start_value + (end_value - start_value) * max(min(1.0, (current_step - start_step) / (end_step - start_step)), 0.0)
    elif isinstance(end_step, float):
        current_step = self.current_epoch
        value = start_value + (end_value - start_value) * max(min(1.0, (current_step - start_step) / (end_step - start_step)), 0.0)
return value
```

- BaseSystem接上
    - preprocess_data(self, batch, stage)
        - pass
    - on_train_batch_start(self, batch, batch_idx, unused=0)等pl.LightningModule规定的方法

```python
"""
Implementing on_after_batch_transfer of DataModule does the same.
But on_after_batch_transfer does not support DP.
"""
def on_train_batch_start(self, batch, batch_idx, unused=0):
    self.dataset = self.trainer.datamodule.train_dataloader().dataset
    self.preprocess_data(batch, 'train')
    update_module_step(self.model, self.current_epoch, self.global_step)

def on_validation_batch_start(self, batch, batch_idx, dataloader_idx):
    self.dataset = self.trainer.datamodule.val_dataloader().dataset
    self.preprocess_data(batch, 'validation')
    update_module_step(self.model, self.current_epoch, self.global_step)

def on_test_batch_start(self, batch, batch_idx, dataloader_idx):
    self.dataset = self.trainer.datamodule.test_dataloader().dataset
    self.preprocess_data(batch, 'test')
    update_module_step(self.model, self.current_epoch, self.global_step)

def on_predict_batch_start(self, batch, batch_idx, dataloader_idx):
    self.dataset = self.trainer.datamodule.predict_dataloader().dataset
    self.preprocess_data(batch, 'predict')
    update_module_step(self.model, self.current_epoch, self.global_step)

def training_step(self, batch, batch_idx):
    raise NotImplementedError
    
def validation_step(self, batch, batch_idx):
    raise NotImplementedError

def validation_epoch_end(self, out):
    """
    Gather metrics from all devices, compute mean.
    Purge repeated results using data index.
    """
    raise NotImplementedError

def test_step(self, batch, batch_idx):        
    raise NotImplementedError

def test_epoch_end(self, out):
    """
    Gather metrics from all devices, compute mean.
    Purge repeated results using data index.
    """
    raise NotImplementedError

def export(self):
    raise NotImplementedError

def configure_optimizers(self):
    optim = parse_optimizer(self.config.system.optimizer, self.model)
    ret = {
        'optimizer': optim,
    }
    if 'scheduler' in self.config.system:
        ret.update({
            'lr_scheduler': parse_scheduler(self.config.system.scheduler, optim),
        })
    return ret
```

```python
in systems/utils.py

得到优化器optim
def parse_optimizer(config, model):
    if hasattr(config, 'params'):
        params = [{'params': get_parameters(model, name), 'name': name, **args} for name, args in config.params.items()]
        rank_zero_debug('Specify optimizer params:', config.params)
    else:
        params = model.parameters()
    if config.name in ['FusedAdam']:
        import apex
        optim = getattr(apex.optimizers, config.name)(params, **config.args)
    else:
        optim = getattr(torch.optim, config.name)(params, **config.args)
    return optim
```


## neus

有两种在console上输出信息的方式：
- self.print: correctly handle progress bar
- rank_zero_info: use the logging module

@systems.register('neus-system')
- NeuSSystem，继承BaseSystem
    - prepare
        - self.criterions = { 'psnr': PSNR()}
        - `self.train_num_samples = self.config.model.train_num_rays * (self.config.model.num_samples_per_ray + self.config.model.get('num_samples_per_ray_bg', 0))`
            - 训练采样数= 训练光线数x(每条光线上采样数+背景中每条光线采样数)
        - self.train_num_rays = self.config.model.train_num_rays
            - 训练光线数 = config中训练光线数
    - forward(self, batch)
        - `return self.model(batch['rays'])`
    - preprocess_data(self, batch, stage)
        - stage: train
            - if batch_image_sampling 随机抽train_num_rays张图片, 索引为index（随机多张图片中，每张图片随机选取一个像素生成光线）
                - directions :(n_images, H, W, 3) --> (train_num_rays, 3)
                - c2w：(n_images, 3, 4)  --> (train_num_rays, 3, 4)
                - rays_o, rays_d : (train_num_rays, 3)
                - rgb: (n_images, H, W, 3) --> (train_num_rays, 3)
                - fg_mask: (n_images, H, W) --> (train_num_rays,)
            - else 随机抽取一张图片，索引index长度为1（一张图片，随机选取多个像素生成光线）
                - directions :(n_images, H, W, 3) --> (train_num_rays, 3)
                - c2w：(n_images, 3, 4)  --> (1, 3, 4)
                - rays_o, rays_d : (train_num_rays, 3)
                - rgb: (n_images, H, W, 3) --> (train_num_rays, 3)
                - fg_mask: (n_images, H, W) --> (train_num_rays,)
        - stage: val
            - `index = batch['index']`
            - c2w: (n_images, 3, 4)  --> (3, 4)
            - directions: (n_images, H, W, 3) --> ( H, W, 3)
            - rays_o, rays_d : (H, W, 3)
            - rgb: (n_images, H, W, 3) --> (len(index)xHxW,3)
            - fg_mask: (n_images, H, W, 3) --> (len(index)xHxW)
        - stage: test
            - `index = batch['index']`
            - c2w: (n_test_traj_steps ,3,4) --> (3,4)
            - directions: (H,W,3) --> (H,W,3)
            - rays_o, rays_d : (H,W,3)
            - rgb: (n_test_traj_steps, H, W, 3) --> (HxW , 3)
            - fg_mask : (n_test_traj_steps, H, W) --> (HxW)
        - rays将rays_o和归一化后的rays_d，cat起来
        - stage: train
            - if bg_color: white
                - model.bg_color = torch.ones((3,))
            - if bg_color: random
                - model.bg_color = torch.rand((3,))
            - else: raise NotImplementedError
        - stage: val, test
            - model.bg_color = torch.ones((3,))
        - if apply_mask:
            - `rgb = rgb * fg_mask[...,None] + model.bg_color * (1 - fg_mask[...,None])`
        - batch.update({'rays': rays, 'rgb': rgb, 'fg_mask': fg_mask})

```python
def preprocess_data(self, batch, stage):
    if 'index' in batch: # validation / testing
        index = batch['index']
    else:
        if self.config.model.batch_image_sampling:
            index = torch.randint(0, len(self.dataset.all_images), size=(self.train_num_rays,), device=self.dataset.all_images.device)
        else:
            index = torch.randint(0, len(self.dataset.all_images), size=(1,), device=self.dataset.all_images.device)
    if stage in ['train']:
        c2w = self.dataset.all_c2w[index]
        x = torch.randint(
            0, self.dataset.w, size=(self.train_num_rays,), device=self.dataset.all_images.device
        )
        y = torch.randint(
            0, self.dataset.h, size=(self.train_num_rays,), device=self.dataset.all_images.device
        )
        if self.dataset.directions.ndim == 3: # (H, W, 3)
            directions = self.dataset.directions[y, x]
        elif self.dataset.directions.ndim == 4: # (N, H, W, 3)
            directions = self.dataset.directions[index, y, x]
        rays_o, rays_d = get_rays(directions, c2w)
        rgb = self.dataset.all_images[index, y, x].view(-1, self.dataset.all_images.shape[-1]).to(self.rank)
        fg_mask = self.dataset.all_fg_masks[index, y, x].view(-1).to(self.rank)
    else:
        c2w = self.dataset.all_c2w[index][0]
        if self.dataset.directions.ndim == 3: # (H, W, 3)
            directions = self.dataset.directions
        elif self.dataset.directions.ndim == 4: # (N, H, W, 3)
            directions = self.dataset.directions[index][0] 
        rays_o, rays_d = get_rays(directions, c2w)
        rgb = self.dataset.all_images[index].view(-1, self.dataset.all_images.shape[-1]).to(self.rank)
        fg_mask = self.dataset.all_fg_masks[index].view(-1).to(self.rank)

    rays = torch.cat([rays_o, F.normalize(rays_d, p=2, dim=-1)], dim=-1)

    if stage in ['train']:
        if self.config.model.background_color == 'white':
            self.model.background_color = torch.ones((3,), dtype=torch.float32, device=self.rank)
        elif self.config.model.background_color == 'random':
            self.model.background_color = torch.rand((3,), dtype=torch.float32, device=self.rank)
        else:
            raise NotImplementedError
    else:
        self.model.background_color = torch.ones((3,), dtype=torch.float32, device=self.rank)
    
    if self.dataset.apply_mask:
        rgb = rgb * fg_mask[...,None] + self.model.background_color * (1 - fg_mask[...,None])
    
    batch.update({
        'rays': rays,
        'rgb': rgb,
        'fg_mask': fg_mask
    })      
```

- NeuSSystem接上
    - training_step(self, batch, batch_idx)
        - `out = self(batch) = self.model(batch['rays'])`(self()相当于执行forward)
        - loss = 0
        - if dynamic_ray_sampling 动态更新训练光线数
            - `train_num_rays = int(train_num_rays*(train_num_samples/out['num_samples_full'].sum().item()))`
                - 如果采样得到的总点数多了，则减少光线数，如果总点数少了，则增加光线数
            - `self.train_num_rays = min(int(self.train_num_rays * 0.9 + train_num_rays * 0.1), self.config.model.max_train_num_rays)`
                - 最后训练光线数为，两者取最小：`原来num*0.9 + 更新后num * 0.1`与config.model.max_train_num_rays
        - loss_rgb_mse = F.mse_loss，log(loss_rgb_mse)，`loss+= loss_rgb_mse *lambda_rgb_mse`
            - render_color: `out['comp_rgb_full'][out['rays_valid_full'][...,0]]` 
            - gt_color: `batch['rgb'][out['rays_valid_full'][...,0]]`
        - loss_rgb_l1 = F.l1_loss，log(loss_rgb_l1)，`loss+= loss_rgb_l1 *lambda_rgb_l1`
            - `out['comp_rgb_full'][out['rays_valid_full'][...,0]]`
            - `batch['rgb'][out['rays_valid_full'][...,0]]`
        - loss_eikonal，log(loss_eikonal)，`loss+= loss_eikonal *lambda_eikonal`
            - `((torch.linalg.norm(out['sdf_grad_samples'], ord=2, dim=-1) - 1.)**2).mean()`
        - loss_mask，log(loss_mask)，`loss+= loss_mask *lambda_mask`
            - opacity.clamp(1.e-3, 1.-1.e-3)
            - `binary_cross_entropy(opacity, batch['fg_mask'].float())`
        - loss_opaque，log(loss_opaque)，`loss+= loss_opaque *lambda_opaque`
            - binary_cross_entropy(opacity, opacity)
        - loss_sparsity，log(loss_sparsity)，`loss+= loss_sparsity *lambda_sparsity`
            -  `torch.exp(-self.config.system.loss.sparsity_scale * out['sdf_samples'].abs()).mean()`
            - $\frac{1}{n.samples} \sum e^{-sparsity.scle \cdot |sdf|}$
        - if lambda_distortion>0
            - loss_distortion，log(loss_distortion)，`loss+= loss_distortion *lambda_distortion`
                - `flatten_eff_distloss(out['weights'], out['points'], out['intervals'], out['ray_indices'])`
        - if learned_background and lambda_distortion_bg>0
            - loss_distortion_bg，log(loss_distortion_bg)，`loss+= loss_distortion_bg *lambda_distortion_bg`
                - `flatten_eff_distloss(out['weights_bg'], out['points_bg'], out['intervals_bg'], out['ray_indices_bg'])`
        - losses_model_reg = self.model.regularizations(out)
        - for name, value in losses_model_reg.items():
            - self.log(f'train/loss_{name}', value)
            - loss_ = value * self.C(self.config.system.loss[f"lambda_{name}"])
            - loss += loss_
        - self.log('train/inv_s', out['inv_s'], prog_bar=True)
        - for name, value in self.config.system.loss.items():
        - if name.startswith('lambda'):
            - self.log(f'train_params/{name}', self.C(value))
        - self.log('train/num_rays', float(self.train_num_rays), prog_bar=True)
        - return {'loss' : loss}

loggers:

`Epoch 0: : 29159it [30:20, 16.02it/s, loss=0.0265, train/inv_s=145.0, train/num_rays=1739.0, val/psnr=23.30]`

![image.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/20230714161558.png)

![image.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/20230714161756.png)


- NeuSSystem接上
    - validation_step(self, batch, batch_idx)
        - out = self(batch)
        - `psnr = self.criterions['psnr'](out['comp_rgb_full'].to(batch['rgb']), batch['rgb'])`
        - W, H = self.dataset.img_wh
        - self.save_image_grid
        - return {'psnr': psnr,'index': batch['index']}

![image.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/20230714162151.png)


- NeuSSystem接上
    - validation_epoch_end(self,out)
        - out = self.all_gather(out) 将所有数据类型的输出拼接起来`Union[Tensor, Dict, List, Tuple]`
        - if self.trainer.is_global_zero: 
            - out_set = {}
            - for step_out in out:
                - DP:` if step_out['index'].ndim == 1: out_set[step_out['index'].item()] = {'psnr': step_out['psnr']}`
                    - ref: 单机vs多机[pytorch中的分布式训练之DP VS DDP - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/356967195)
                - DDP: `for oi, index in enumerate(step_out['index']):`
                    - `out_set[index[0].item()] = {'psnr': step_out['psnr'][oi]}`
            - psnr = $\frac{1}{len(index)}\sum psnr_{i}$
            - self.log(psnr)
    - test_step(self, batch, batch_idx)
        - out = self(batch)
        - `psnr = self.criterions['psnr'](out['comp_rgb_full'].to(batch['rgb']), batch['rgb'])`
        - W, H = self.dataset.img_wh
        - self.save_image_grid
            - 由于测试时，采用的相机位姿是未知的新视点，因此在image生成时，`batch['rgb']`即gt图为zero(黑色)
        - return {'psnr': psnr,'index': batch['index']}

![image.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/20230714212151.png)


- NeuSSystem接上
    - test_epoch_end(self,out)
        - 同validation
            - psnr = $\frac{1}{len(index)}\sum psnr_{i}$
            - self.log(psnr)
        - self.save_img_sequence()
        - self.export
    - export
        - mesh = self.model.export(self.config.export)
        - self.save_mesh() 



## criterions

- PSNR，继承nn.Module
    - forward(self, inputs , targets, valid_mask= None, reduction= 'mean')
        - assert reduction in ['mean', 'none']
        - `value = (inputs - targets)**2`，即$v = (inputs - targets)^{2}$
        - if valid_mask is not None:
            - value = value[valid_mask]
        - if reduction == 'mean':
            - return -10 * torch.log10(torch.mean(value))
            - $psnr = 10 \cdot log_{10}(\frac{1}{N} \sum v)$
        - elif reduction == 'none':
            - return -10 * torch.log10(torch.mean(value, dim=tuple(range(value.ndim)[1:])))


## utils

- ChainedScheduler
- SequentialLR
- ConstantLR
- LinearLR
- get_scheduler
- getattr_recursive
- get_parameters
- parse_optimizer
- parse_scheduler
- update_module_step(m,epoch,global_step)
    - if hasattr(m,'update_step') 如果m中有update_step这个属性or方法
        - m.update_step(epoch,global_step) 则执行m.update_step(epoch,global_step)
    - 如果m中没有update_step，则不执行操作


# utils

## mixins

class SaverMixin(): 被systems中的BaseSystem继承

- get_save_path(self,filename)

```python
def get_save_path(self, filename):
    save_path = os.path.join(self.save_dir, filename)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    return save_path
```

- save_image_grid(self, filename, imgs)
    - img = self.get_image_grid_(imgs)
    - cv2.imwrite(self.get_save_path(filename),img)

```python
in val step:
self.save_image_grid(f"it{self.global_step}-{batch['index'][0].item()}.png", [
    {'type': 'rgb', 'img': batch['rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
    {'type': 'rgb', 'img': out['comp_rgb_full'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}}
] + ([
    {'type': 'rgb', 'img': out['comp_rgb_bg'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
    {'type': 'rgb', 'img': out['comp_rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
] if self.config.model.learned_background else []) + [
    {'type': 'grayscale', 'img': out['depth'].view(H, W), 'kwargs': {}},
    {'type': 'rgb', 'img': out['comp_normal'].view(H, W, 3), 'kwargs': {'data_format': 'HWC', 'data_range': (-1, 1)}}
])

in test_step:
self.save_image_grid(f"it{self.global_step}-test/{batch['index'][0].item()}.png", [
    {'type': 'rgb', 'img': batch['rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
    {'type': 'rgb', 'img': out['comp_rgb_full'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}}
] + ([
    {'type': 'rgb', 'img': out['comp_rgb_bg'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
    {'type': 'rgb', 'img': out['comp_rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
] if self.config.model.learned_background else []) + [
    {'type': 'grayscale', 'img': out['depth'].view(H, W), 'kwargs': {}},
    {'type': 'rgb', 'img': out['comp_normal'].view(H, W, 3), 'kwargs': {'data_format': 'HWC', 'data_range': (-1, 1)}}
])
```

- get_image_grid_(self, imgs)

```python
def get_image_grid_(self, imgs):
    if isinstance(imgs[0], list):
        return np.concatenate([self.get_image_grid_(row) for row in imgs], axis=0)
    cols = []
    for col in imgs:
        assert col['type'] in ['rgb', 'uv', 'grayscale']
        if col['type'] == 'rgb':
            rgb_kwargs = self.DEFAULT_RGB_KWARGS.copy()
            rgb_kwargs.update(col['kwargs'])
            cols.append(self.get_rgb_image_(col['img'], **rgb_kwargs))
        elif col['type'] == 'uv':
            uv_kwargs = self.DEFAULT_UV_KWARGS.copy()
            uv_kwargs.update(col['kwargs'])
            cols.append(self.get_uv_image_(col['img'], **uv_kwargs))
        elif col['type'] == 'grayscale':
            grayscale_kwargs = self.DEFAULT_GRAYSCALE_KWARGS.copy()
            grayscale_kwargs.update(col['kwargs'])
            cols.append(self.get_grayscale_image_(col['img'], **grayscale_kwargs))
    return np.concatenate(cols, axis=1)

DEFAULT_RGB_KWARGS = {'data_format': 'CHW', 'data_range': (0, 1)}
DEFAULT_UV_KWARGS = {'data_format': 'CHW', 'data_range': (0, 1), 'cmap': 'checkerboard'}
DEFAULT_GRAYSCALE_KWARGS = {'data_range': None, 'cmap': 'jet'}
```


- get_rgb_image_(self, img, data_format, data_range))

```python
def get_rgb_image_(self, img, data_format, data_range):
    img = self.convert_data(img)
    assert data_format in ['CHW', 'HWC']
    if data_format == 'CHW':
        img = img.transpose(1, 2, 0)
    img = img.clip(min=data_range[0], max=data_range[1])
    img = ((img - data_range[0]) / (data_range[1] - data_range[0]) * 255.).astype(np.uint8)
    imgs = [img[...,start:start+3] for start in range(0, img.shape[-1], 3)]
    imgs = [img_ if img_.shape[-1] == 3 else np.concatenate([img_, np.zeros((img_.shape[0], img_.shape[1], 3 - img_.shape[2]), dtype=img_.dtype)], axis=-1) for img_ in imgs]
    img = np.concatenate(imgs, axis=1)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img
```


- get_grayscale_image_(self, img, data_range , cmap)

```python
def get_grayscale_image_(self, img, data_range, cmap):
    img = self.convert_data(img)
    img = np.nan_to_num(img)
    if data_range is None:
        img = (img - img.min()) / (img.max() - img.min())
    else:
        img = img.clip(data_range[0], data_range[1])
        img = (img - data_range[0]) / (data_range[1] - data_range[0])
    assert cmap in [None, 'jet', 'magma']
    if cmap == None:
        img = (img * 255.).astype(np.uint8)
        img = np.repeat(img[...,None], 3, axis=2)
    elif cmap == 'jet':
        img = (img * 255.).astype(np.uint8)
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    elif cmap == 'magma':
        img = 1. - img
        base = cm.get_cmap('magma')
        num_bins = 256
        colormap = LinearSegmentedColormap.from_list(
            f"{base.name}{num_bins}",
            base(np.linspace(0, 1, num_bins)),
            num_bins
        )(np.linspace(0, 1, num_bins))[:,:3]
        a = np.floor(img * 255.)
        b = (a + 1).clip(max=255.)
        f = img * 255. - a
        a = a.astype(np.uint16).clip(0, 255)
        b = b.astype(np.uint16).clip(0, 255)
        img = colormap[a] + (colormap[b] - colormap[a]) * f[...,None]
        img = (img * 255.).astype(np.uint8)
    return img
```

- convert_data(self, data)，将输入的数据转化成ndarry类型

```python
def convert_data(self, data): # isinstance 判断一个对象是否是一个已知的类型
    if isinstance(data, np.ndarray):
        return data
    elif isinstance(data, torch.Tensor):
        return data.cpu().numpy()
    elif isinstance(data, list):
        return [self.convert_data(d) for d in data]
    elif isinstance(data, dict):
        return {k: self.convert_data(v) for k, v in data.items()}
    else:
        raise TypeError('Data must be in type numpy.ndarray, torch.Tensor, list or dict, getting', type(data))
```

- save_img_sequence(self, filename, img_dir, matcher, save_format='gif', fps=30)

```python
in test step
self.save_img_sequence(
    f"it{self.global_step}-test", # mp4 or gif文件名
    f"it{self.global_step}-test", # test生成的图片保存目录
    '(\d+)\.png',
    save_format='mp4',
    fps=30
)

def save_img_sequence(self, filename, img_dir, matcher, save_format='gif', fps=30):
    assert save_format in ['gif', 'mp4']
    if not filename.endswith(save_format):
        filename += f".{save_format}"
    matcher = re.compile(matcher)
    img_dir = os.path.join(self.save_dir, img_dir)
    imgs = []
    for f in os.listdir(img_dir):
        if matcher.search(f):
            imgs.append(f)
    imgs = sorted(imgs, key=lambda f: int(matcher.search(f).groups()[0]))
    imgs = [cv2.imread(os.path.join(img_dir, f)) for f in imgs]
    
    if save_format == 'gif':
        imgs = [cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in imgs]
        imageio.mimsave(self.get_save_path(filename), imgs, fps=fps, palettesize=256)
    elif save_format == 'mp4':
        imgs = [cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in imgs]
        imageio.mimsave(self.get_save_path(filename), imgs, fps=fps)
```

- save_mesh()

```python
in export: 
self.save_mesh(
    f"it{self.global_step}-{self.config.model.geometry.isosurface.method}{self.config.model.geometry.isosurface.resolution}.obj",
    **mesh
)        

def save_mesh(self, filename, v_pos, t_pos_idx, v_tex=None, t_tex_idx=None, v_rgb=None):
    v_pos, t_pos_idx = self.convert_data(v_pos), self.convert_data(t_pos_idx)
    if v_rgb is not None:
        v_rgb = self.convert_data(v_rgb) # 转为numpy

    import trimesh
    mesh = trimesh.Trimesh(
        vertices=v_pos,
        faces=t_pos_idx,
        vertex_colors=v_rgb
    )
    mesh.export(self.get_save_path(filename))
```


obj文件：

{% note info %}
可以看出最后生成的模型在一个半径为1的单位圆中
{% endnote %}
```
# 每个点的位置值和颜色值
v -0.96953946 0.71037185 0.47863841 0.78431373 0.56470588 0.34117647
v -0.96868885 0.70891666 0.47863841 0.97647059 0.86666667 0.66666667
v -0.96868885 0.71037185 0.47713959 0.74901961 0.54509804 0.37647059
...

# 每个三角面的三个顶点的索引
f 2370 2366 2270
f 2366 2265 2270
f 2374 2372 2373
...
```