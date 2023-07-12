---
title: Instant Neus的代码理解
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
- create_spheric_poses
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
- DTUDataModule 继承pl.LightningDataMoudle
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
    - self.register_parameter来自nn.Moudle注册一个参数
    - if self.moudlate
        - True: mod_start_steps, reach_max_steps, max_inv_s
        - False: none
- @property 将该函数变为类的属性: inv_s()
    - $val = e^{variance * 10.0}$
    - if self.moudlate and do_mod
        - val = val.clamp_max(mod_val)
    - return val
- forward(self,x)
    - `return torch.ones([len(x), 1], device=self.variance.device) * self.inv_s` 
        - 输入长度x1大小的inv_s
- update_step(self,epoch,global_step)
    - if self.moudlate
        - ...

### NeuSModel 继承BaseModel
@models.register('neus')

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
    - if self.training(在nn.Moudle中可以这样判断是否为训练模式) and self.grid_prune
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
    - rgb = self.texture_bg(feature, t_dirs)
    - weights = render_weight_from_density(t_starts, t_ends, density[...,None], ray_indices=ray_indices, n_rays=n_rays)
        - 权重同NeRF: $w_i = T_i(1 - exp(-\sigma_i\delta_i)), \quad\textrm{where}\quad T_i = exp(-\sum_{j=1}^{i-1}\sigma_j\delta_j)$
        - ref: [nerfacc.render_weight_from_density — nerfacc 0.3.5 documentation](https://www.nerfacc.com/en/v0.3.5/apis/generated/nerfacc.render_weight_from_density.html#nerfacc.render_weight_from_density)
    - opacity = accumulate_along_rays(weights, ray_indices, values=None, n_rays=n_rays)
        - ref: [nerfacc.accumulate_along_rays — nerfacc 0.3.5 documentation](https://www.nerfacc.com/en/v0.3.5/apis/generated/nerfacc.accumulate_along_rays.html#nerfacc.accumulate_along_rays)
    - depth = accumulate_along_rays(weights, ray_indices, values=midpoints, n_rays=n_rays)
    - comp_rgb = accumulate_along_rays(weights, ray_indices, values=rgb, n_rays=n_rays)
    - comp_rgb = comp_rgb + self.background_color * (1.0 - opacity)
    - return out 

```
out = {
    'comp_rgb': comp_rgb,
    'opacity': opacity,
    'depth': depth,
    'rays_valid': opacity > 0,
    'num_samples': torch.as_tensor([len(t_starts)], dtype=torch.int32,device=rays.device)
}

if self.training:
    out.update({
        'weights': weights.view(-1),
        'points': midpoints.view(-1),
        'intervals': intervals.view(-1),
        'ray_indices': ray_indices.view(-1)
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
    - normal = F.normalize(sdf_grad, p=2, dim=-1) 法向量：将sdf的梯度归一化
    - alpha = self.get_alpha(sdf, normal, t_dirs, dists)[...,None]
    - rgb = self.texture(feature, t_dirs, normal)
    - weights = render_weight_from_alpha(alpha, ray_indices=ray_indices, n_rays=n_rays)
    - opacity = accumulate_along_rays(weights, ray_indices, values=None, n_rays=n_rays)
    - depth = accumulate_along_rays(weights, ray_indices, values=midpoints, n_rays=n_rays)
    - comp_rgb = accumulate_along_rays(weights, ray_indices, values=rgb, n_rays=n_rays)
    - comp_normal = accumulate_along_rays(weights, ray_indices, values=normal, n_rays=n_rays)
    - comp_normal = F.normalize(comp_normal, p=2, dim=-1)
    - return `{**out, **{k + '_bg': v for k, v in out_bg.items()}, **{k + '_full': v for k, v in ut_full.items()}}`

```
out = {
    'comp_rgb': comp_rgb,
    'comp_normal': comp_normal,
    'opacity': opacity,
    'depth': depth,
    'rays_valid': opacity > 0,
    'num_samples': torch.as_tensor([len(t_starts)], dtype=torch.int32, device=rays.device)
}

if self.training:
    out.update({
        'sdf_samples': sdf,
        'sdf_grad_samples': sdf_grad,
        'weights': weights.view(-1),
        'points': midpoints.view(-1),
        'intervals': dists.view(-1),
        'ray_indices': ray_indices.view(-1)                
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
```
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

各种编码方式和NeRF的MLP网络

- VanillaFrequency

Vanilla：最初始的
即NeRF中的频率编码方式

- ProgressiveBandHashGrid
- CompositeEncoding
- get_encoding
- VanillaMLP
    - NeRF中的MLP
- sphere_init_tcnn_network
- get_mlp
- EncodingWithNetwork
- get_encoding_with_network

## geometry

- contract_to_unisphere
- MarchingCubeHelper 继承nn.Moudle
- BaseImplicitGeometry
- VolumeDensity
    - @models.register('volume-density')
- VolumeSDF
    - @models.register('volume-sdf')

## texture

- VolumeRadiance
    - @models.register('volume-radiance')
- VolumeColor
    - @models.register('volume-color')

## utils

- `chunk_batch(func, chunk_size, move_to_cpu, *args, **kwargs)`
    - 

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

- BaseSystem

## neus

- ### NeuSSystem
    - @systems.register('neus-system')
- prepare
- forward
- preprocess_data
- training_step
- validation_step
- validation_epoch_end
- test_step
- test_epoch_end
- export



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