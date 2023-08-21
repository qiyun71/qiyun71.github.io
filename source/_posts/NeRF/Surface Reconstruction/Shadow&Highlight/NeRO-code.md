---
title: NeRO-code
date: 2023-08-01T15:17:39.000Z
tags:
  - Code
  - Python
  - NeRO
categories: NeRF/Surface Reconstruction/Shadow&Highlight
date updated: 2023-08-03 14:33
---

NeRO代码[liuyuan-pal/NeRO: [SIGGRAPH2023] NeRO: Neural Geometry and BRDF Reconstruction of Reflective Objects from Multiview Images (github.com)](https://github.com/liuyuan-pal/NeRO)

![image.png](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230728142918.png)

<!-- more -->

# Dataset

## DummyDataset(Dataset)

`name2dataset[self.cfg['train_dataset_type']](self.cfg['train_dataset_cfg'], True)`
相当于：`DummyDataset(self.cfg['train_dataset_cfg'], True)`

- is_train: 
    - `__getitem__ : return {}`
    - `__len__ : return 99999999` 
- else:
    - `__getitem__ : return {'index': index}`
    - `__len__ : return self.test_num` 

not is_train:
```python
if not is_train:
    # return 一个实例 eg：GlossyRealDatabase(self.cfg['database_name'])
    database = parse_database_name(self.cfg['database_name'])
    # 使用database中的get_img_ids方法，获取train_ids和test_ids
    train_ids, test_ids = get_database_split(database, 'validation')
    self.train_num = len(train_ids)
    self.test_num = len(test_ids)
    print('val' , is_train)
```

**database.py**:
- parse_database_name

```python
def parse_database_name(database_name:str)->BaseDatabase:
    name2database={
        'syn': GlossySyntheticDatabase,
        'real': GlossyRealDatabase,
        'custom': CustomDatabase,
    }
    database_type = database_name.split('/')[0]
    if database_type in name2database:
        return name2database[database_type](database_name)
    else:
        raise NotImplementedError
```

- get_database_split
    - 打乱self.img_ids，并split为test和train

```python 
def get_database_split(database: BaseDatabase, split_type='validation'):
    if split_type=='validation':
        random.seed(6033)
        img_ids = database.get_img_ids()
        random.shuffle(img_ids)
        test_ids = img_ids[:1]
        train_ids = img_ids[1:]
    elif split_type=='test':
        test_ids, train_ids = read_pickle('configs/synthetic_split_128.pkl')
    else:
        raise NotImplementedError
    return train_ids, test_ids
"""
f = open('E:\\BaiduSyncdisk\\NeRF_Proj\\NeRO\\configs\\synthetic_split_128.pkl','rb')
[array(['0', '4', '19', '2', '127', '71', '73', '56', '75', '95', '93',
       '110', '91', '7', '5', '3', '68', '30', '66', '113', '111', '33',
       '120', '31', '29', '14', '49', '11', '109', '61', '59', '57'],
      dtype='<U3'), 
['1', '6', '8', '9', '10', '12', '13', '15', '16', '17', '18', '20', '21', '22', '23', '24',
'25', '26', '27', '28', '32', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43',
'44', '45', '46', '47', '48', '50', '51', '52', '53', '54', '55', '58', '60', '62', '63',
'64', '65', '67', '69', '70', '72', '74', '76', '77', '78', '79', '80', '81', '82', '83',
'84', '85', '86', '87', '88', '89', '90', '92', '94', '96', '97', '98', '99', '100', '101',
'102', '103', '104', '105', '106', '107', '108', '112', '114', '115', '116', '117',
'118', '119', '121', '122', '123', '124', '125', '126']]
"""
```

## GlossyRealDatabase

### init
- meta_info

```python
meta_info={
    'bear': {'forward': np.asarray([0.539944,-0.342791,0.341446],np.float32), 'up': np.asarray((0.0512875,-0.645326,-0.762183),np.float32),},
    'coral': {'forward': np.asarray([0.004226,-0.235523,0.267582],np.float32), 'up': np.asarray((0.0477973,-0.748313,-0.661622),np.float32),},
    'maneki': {'forward': np.asarray([-2.336584, -0.406351, 0.482029], np.float32), 'up': np.asarray((-0.0117387, -0.738751, -0.673876), np.float32), },
    'bunny': {'forward': np.asarray([0.437076,-1.672467,1.436961],np.float32), 'up': np.asarray((-0.0693234,-0.644819,-.761185),np.float32),},
    'vase': {'forward': np.asarray([-0.911907, -0.132777, 0.180063], np.float32), 'up': np.asarray((-0.01911, -0.738918, -0.673524), np.float32), },
}
```
从CloudCampare中获取的数据，其中forward是手动设置的前向，up是根据手动截取的一小块平面的法向
[NeRO/custom_object.md at main · liuyuan-pal/NeRO (github.com)](https://github.com/liuyuan-pal/NeRO/blob/main/custom_object.md)

<div style="display:flex; justify-content:space-between;"> <img src="https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230803160338.png" alt="Image 1" style="width:50%;"><div style="width:10px;"></div> <img src="https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230803160313.png" alt="Image 2" style="width:50%;"> </div>

- `_parse_colmap` 从cache.pkl中读取数据，如果cache.pkl不存在，则从`/colmap/sparse/0`中读取并将数据写入到cache.pkl中
    - self.poses (3,4), self.Ks (3,3), self.image_names(dict is img_ids : img_name), self.img_ids(list len is num_images)
- `_normalize` 读取object_point_cloud.ply，获得截取出来的物体点云坐标，将该点云坐标标准化到单位bound中，并将世界基坐标系转换到手动设置的坐标系即up、forward
    - 即`_normalize`将点云坐标从原来的世界坐标系w转换到新的坐标系w'下
    - 将pose的w2c数据转换为w'2c，其中w'原点在object_point_cloud.ply中心，xyz轴是自定义的轴，如上图，但是scale不变
- `database_name: real/bear/raw_1024`中最后一个raw_1024
    - 如果以raw开头：
        - 将images中图片缩放并保存到images_raw_1024中
        - 并更换self.Ks中数据（由于缩放了W、H）

```python
"""
# f = open('E:\\BaiduSyncdisk\\NeRF_Proj\\NeRO\\data\\GlossyReal\\bear\\cache.pkl','rb')
data:
[{1: array([[-0.21238244, -0.73007965,  0.6495209 , -0.28842232],
       [ 0.4490811 ,  0.5174136 ,  0.7284294 , -2.9991536 ],
       [-0.86788243,  0.44639313,  0.21797541,  3.0713193 ]],
      dtype=float32), 2: array([[ 0.13014111, -0.75269127,  0.64538294, -0.3143167 ],
       [ 0.47691151,  0.6181942 ,  0.624813  , -2.9732292 ],
       [-0.8692633 ,  0.22647668,  0.43941963,  2.7372477 ]],
        dtype=float32), 
       ...}
{...96: array([[6.013423e+03, 0.000000e+00, 1.368000e+03],
       [0.000000e+00, 6.013423e+03, 1.824000e+03],
       [0.000000e+00, 0.000000e+00, 1.000000e+00]], dtype=float32), 97: array([[6.013423e+03, 0.000000e+00, 1.368000e+03],
       [0.000000e+00, 6.013423e+03, 1.824000e+03],
       [0.000000e+00, 0.000000e+00, 1.000000e+00]], dtype=float32)}
{1: '1436.jpg', 2: '1437.jpg', 3: '1438.jpg', 4: '1439.jpg', 5: '1440.jpg', 6: '1441.jpg', 7: '1442.jpg', 
8: '1443.jpg', 9: '1444.jpg', 10: '1445.jpg', 11: '1446.jpg', 12: '1447.jpg', 13: '1448.jpg', 14: '1449.jpg', 
15: '1450.jpg', 16: '1451.jpg', 17: '1452.jpg', 18: '1453.jpg', 19: '1454.jpg', 20: '1455.jpg', 21: '1456.jpg', 
22: '1457.jpg', 23: '1458.jpg', 24: '1459.jpg', 25: '1460.jpg', 26: '1461.jpg', 27: '1462.jpg', 28: '1463.jpg', 
29: '1464.jpg', 30: '1465.jpg', 31: '1466.jpg', 32: '1467.jpg', 33: '1468.jpg', 34: '1469.jpg', 35: '1470.jpg', 
36: '1471.jpg', 37: '1472.jpg', 38: '1473.jpg', 39: '1474.jpg', 40: '1475.jpg', 41: '1476.jpg', 42: '1477.jpg', 
43: '1478.jpg', 44: '1479.jpg', 45: '1480.jpg', 46: '1481.jpg', 47: '1482.jpg', 48: '1483.jpg', 49: '1484.jpg', 
50: '1485.jpg', 51: '1486.jpg', 52: '1487.jpg', 53: '1488.jpg', 54: '1489.jpg', 55: '1490.jpg', 56: '1491.jpg', 
57: '1492.jpg', 58: '1493.jpg', 59: '1494.jpg', 60: '1495.jpg', 61: '1496.jpg', 62: '1497.jpg', 63: '1498.jpg', 
64: '1499.jpg', 65: '1500.jpg', 66: '1501.jpg', 67: '1502.jpg', 68: '1503.jpg', 69: '1504.jpg', 70: '1505.jpg', 
71: '1506.jpg', 72: '1507.jpg', 73: '1508.jpg', 74: '1509.jpg', 75: '1510.jpg', 76: '1511.jpg', 77: '1512.jpg', 
78: '1513.jpg', 79: '1514.jpg', 80: '1515.jpg', 81: '1516.jpg', 82: '1517.jpg', 83: '1518.jpg', 84: '1519.jpg', 
85: '1520.jpg', 86: '1521.jpg', 87: '1522.jpg', 88: '1523.jpg', 89: '1524.jpg', 90: '1525.jpg', 91: '1526.jpg', 
92: '1527.jpg', 93: '1528.jpg', 94: '1529.jpg', 95: '1530.jpg', 96: '1531.jpg', 97: '1532.jpg'}, 
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 
30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 
57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 
84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97]]
"""
```


# Network&Render

## NeROShapeRenderer

### MLP

![image.png](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230728142918.png)


- SDFNetwork
    - ![SDFNetwork](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/SDFNetwork_modify.png)
    - 39-->256-->256-->256-->217-->256-->256-->256-->256-->257

- NeRFNetwork
    - ![Pasted image 20221206180113.png|600](https://raw.githubusercontent.com/yq010105/Blog_images/main/Pasted%20image%2020221206180113.png)
    - 84-->256-->256-->256-->256-->256+84-->256-->256-->256+27-->128-->3
    - 84-->256-->256-->256-->256-->256+84-->256-->256-->256-->1

- AppShadingNetwork
    - metallic_predictor
        - 259-->256-->256-->256-->1
        - 259 = (256 + 3) = (feature + points)
    - roughness_predictor
        - 259-->256-->256-->256-->1
    - albedo_predictor
        - 259-->256-->256-->256-->3
    - outer_light 直接光
        - 72-->256-->256-->256-->3
        - 72 = ref_roughness = self.sph_enc(reflective, roughness)
        - or self.sph_enc(normals, roughness) --> diffuse_lights
    - inner_light 间接光
        - 123-->256-->256-->256-->3
        - 123 = (51 + 72) = (pts + ref_roughness) 
            - pts : 2x3x8 + 3 = 51 (L=8)
    - inner_weight 遮挡概率occ_prob
        - 90-->256-->256-->256-->1
        - 90 = (51 + 39) = (pts + ref_)
            - ref_ = self.dir_enc(reflective)  = 2x3x6 + 3 = 39(L=6)
    - human_light_predictor  $[\alpha_{\mathrm{camera}},\mathrm{c}_{\mathrm{camera}}]=g_{\mathrm{camera}}(\mathrm{p}_{\mathrm{c}}),$
        - 24-->256-->256-->256-->4
        - 24 = pos_enc = IPE(mean, var, 0, 6)

| MLP                   | Encoding                     | in_dims                            | out_dims | layer | neurons |
| --------------------- | ---------------------------- | ---------------------------------- | -------- | ----- | ------- |
| SDFNetwork            | VanillaFrequency(L=6)        | 2x3x6+3=39                         | 257      | 8     | 256     |
| SingleVarianceNetwork | None                         | ...                                | ...      | ...   | ...     |
| NeRFNetwork           | VanillaFrequency(Lp=10,Lv=4) | 2x4x10+4=84(Nerf++:4) & 2x3x4+3=27 | 4        | 8     | 256     |
| AppShadingNetwork     | VanillaFrequency             | ...                                | ...      | ...   | ...     |
| metallic_predictor    | None                         | 256 + 3 = 259                      | 1        | 4     | 256     |
| roughness_predictor   | None                         | 256 + 3 = 259                      | 1        | 4     | 256     |
| albedo_predictor      | None                         | 256 + 3 = 259                      | 3        | 4     | 256     |
| outer_light           | IDE(Ref-NeRF)                | 72                                 | 3        | 4     | 256     |
| inner_light           | VanillaFrequency+IDE         | 51 + 72 = 123                      | 3        | 4     | 256     |
| inner_weight          | VanillaFrequency             | 51 + 39 = 90                       | 1        | 4     | 256     |
| human_light_predictor | IPE                          | 2 x 2 x 6 = 24                     | 4        | 4     | 256     |


- FG_LUT: `[1,256,256,2]`
    - from assets/bsdf_256_256.bin

**weight_norm:** weight_v, weight_g
![image.png](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230803155215.png)
### Render

- `_init_dataset`
    - parse_database_name：`self.database = GlossyRealDatabase(self.cfg['database_name'])`
    - get_database_split：self.train_ids, self.test_ids is `img_ids[1:] , img_ids[:1]`
    - build_imgs_info(self.database, self.train_ids)：根据 train_ids 加载 train_imgs_info
        - train_imgs_info：{'imgs': images, 'Ks': Ks, 'poses': poses}
    - imgs_info_to_torch(self.train_imgs_info, device = 'cpu')：将train_imgs_info转换为torch、如果为imgs，则permute(0,3,1,2)，然后to device
    - 同上加载test_imgs_info并to torch and to device
    - `_construct_ray_batch(self.train_imgs_info)` : 返回`train_batch = ray_batch`, `self.train_poses = poses# imn,3,4`, `tbn = rn = imn * h * w`, h, w
        - `{'dirs': dirs.float().reshape(rn, 3).to(device), 'rgbs': imgs.float().reshape(rn, 3).to(device),'idxs': idxs.long().reshape(rn, 1).to(device)}`
        - pose的`[:3,:3]`为正交矩阵
    - `_shuffle_train_batch`将train_batch数据打乱，每个像素点dir和rgb
- forward
    - is_train:
        - `outputs = self.train_step(step)`
    - else:
        - index = data['index']
        - outputs = self.test_step(index, step=step)

```python
def train_step(self, step):
    rn = self.cfg['train_ray_num']
    # fetch to gpu
    train_ray_batch = {k: v[self.train_batch_i:self.train_batch_i + rn].cuda() for k, v in self.train_batch.items()}
    self.train_batch_i += rn
    if self.train_batch_i + rn >= self.tbn: self._shuffle_train_batch() # 当完成一个tbn = img_nums * h * w时，打乱一次顺序
    train_poses = self.train_poses.cuda()
    rays_o, rays_d, near, far, human_poses = self._process_ray_batch(train_ray_batch, train_poses)

    outputs = self.render(rays_o, rays_d, near, far, human_poses, -1, self.get_anneal_val(step), is_train=True, step=step)
    outputs['loss_rgb'] = self.compute_rgb_loss(outputs['ray_rgb'], train_ray_batch['rgbs'])  # ray_loss
    return outputs
```

#### train_step

- `_process_ray_batch`
    - input：ray_batch, poses
    - output：rays_o, rays_d, near, far, human_poses[idxs]
    - 将原世界坐标系下o点转换到手动设置的世界坐标系w'下，并将ray_batch中的dirs转换到手动设置的世界坐标系w'下
    - near_far_from_sphere:
        - get near and far through rays_o, rays_d
    - get_human_coordinate_poses
        - 根据pose得到human_poses（w2c）：用于判断从相机发出的光线在物体上反射是否击中human
        - human_poses: 在相机原点处（相机原点不动），z轴为原相机坐标系z轴在**与w'的xoy平面平行的xoy’平面**的投影单位向量，y轴为w'下z-方向的单位向量
        - ![image.png](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230728143216.png)
- get_anneal_val(step)
    - if self.cfg['anneal_end'] < 0 : `1`
    - else : `np.min([1.0, step / self.cfg['anneal_end']])`
- render(rays_o, rays_d, near, far, human_poses, perturb_overwrite=-1, cos_anneal_ratio=0.0, is_train=True, step=None)
    - input: rays_o, rays_d, near, far, human_poses, -1, self.get_anneal_val(step), is_train=True, step=step
    - output: ret = outputs
    - sample_ray(rays_o, rays_d, near, far, perturb)
        - 同neus，上采样+cat_z_vals堆叠z_vals
    - render_core
        - input: rays_o, rays_d, z_vals, human_poses, cos_anneal_ratio=cos_anneal_ratio, step=step, is_train=is_train
        - output: outputs
        - get dists , mid_z_vals , points , inner_mask , outer_mask ,dirs , human_poses_pt
        - if torch.sum(outer_mask) > 0: 背景使用NeRF网络得到颜色和不透明度
            - `alpha[outer_mask], sampled_color[outer_mask] = self.compute_density_alpha(points[outer_mask], dists[outer_mask], -dirs[outer_mask], self.outer_nerf)`
        - if torch.sum(inner_mask) > 0: 前景使用SDF和Color网络得到sdf和颜色、碰撞信息 ， else：`gradient_error = torch.zeros(1)`
            - `alpha[inner_mask], gradients, feature_vector, inv_s, sdf = self.compute_sdf_alpha(points[inner_mask], dists[inner_mask], dirs[inner_mask], cos_anneal_ratio, step)`
            - `sampled_color[inner_mask], occ_info = self.color_network(points[inner_mask], gradients, -dirs[inner_mask], feature_vector, human_poses_pt[inner_mask], step=step)`
                - sampled_color : 采样点颜色
                - occ_info : dict {'reflective': reflective, 'occ_prob': occ_prob,}
            - `gradient_error = (torch.linalg.norm(gradients, ord=2, dim=-1) - 1.0) ** 2` 梯度损失
        - alpha -- > weight 
        - sampled_color, weight --> color
        - outputs = {'ray_rgb': color,  'gradient_error': gradient_error,}
        - `if torch.sum(inner_mask) > 0: outputs['std'] = torch.mean(1 / inv_s)`  | `else:  outputs['std'] = torch.zeros(1)`
        - if step < 1000: 
            - mask = torch.norm(points, dim=-1) < 1.2
            - `outputs['sdf_pts'] = points[mask]`
            - `outputs['sdf_vals'] = self.sdf_network.sdf(points[mask])[..., 0]`
        - `if self.cfg['apply_occ_loss']:`
            - if torch.sum(inner_mask) > 0: `outputs['loss_occ'] = self.compute_occ_loss(occ_info, points[inner_mask], sdf, gradients, dirs[inner_mask], step)`
                - compute_occ_loss 碰撞损失
            - else: `outputs['loss_occ'] = torch.zeros(1)`
        - if not is_train: `outputs.update(self.compute_validation_info(z_vals, rays_o, rays_d, weights, human_poses, step))`
        - return outputs
- **rgb_loss**: `outputs['loss_rgb'] = self.compute_rgb_loss(outputs['ray_rgb'], train_ray_batch['rgbs'])`

```python
cfg['rgb_loss'] = 'charbonier'
def compute_rgb_loss(self, rgb_pr, rgb_gt):
    if self.cfg['rgb_loss'] == 'l2':
        rgb_loss = torch.sum((rgb_pr - rgb_gt) ** 2, -1)
    elif self.cfg['rgb_loss'] == 'l1':
        rgb_loss = torch.sum(F.l1_loss(rgb_pr, rgb_gt, reduction='none'), -1)
    elif self.cfg['rgb_loss'] == 'smooth_l1':
        rgb_loss = torch.sum(F.smooth_l1_loss(rgb_pr, rgb_gt, reduction='none', beta=0.25), -1)
    elif self.cfg['rgb_loss'] == 'charbonier':
        epsilon = 0.001
        rgb_loss = torch.sqrt(torch.sum((rgb_gt - rgb_pr) ** 2, dim=-1) + epsilon)
    else:
        raise NotImplementedError
    return rgb_loss
```

**render core** : MLP output
![render_core.png](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/render_core.png)

**Occ related**: Occ prob and loss_occ

![occ.png](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/occ.png)


##### loss_occ

```python
inter_dist, inter_prob, inter_sdf = get_intersection(self.sdf_inter_fun, self.deviation_network, points[mask], reflective[mask], sn0=64, sn1=16)  # pn,sn-1
occ_prob_gt = torch.sum(inter_prob, -1, keepdim=True)
return F.l1_loss(occ_prob[mask], occ_prob_gt)
```

```python
def get_intersection(sdf_fun, inv_fun, pts, dirs, sn0=128, sn1=9):
    """
    :param sdf_fun:
    :param inv_fun:
    :param pts:    pn,3
    :param dirs:   pn,3
    :param sn0: # 64
    :param sn1: # 16
    :return:
    """
    inside_mask = torch.norm(pts, dim=-1) < 0.999 # left some margin
    pn, _ = pts.shape
    hit_z_vals = torch.zeros([pn, sn1-1])
    hit_weights = torch.zeros([pn, sn1-1])
    hit_sdf = -torch.ones([pn, sn1-1])
    if torch.sum(inside_mask)>0:
        pts = pts[inside_mask]
        dirs = dirs[inside_mask]
        max_dist = get_sphere_intersection(pts, dirs) # pn,1
        with torch.no_grad():
            z_vals = torch.linspace(0, 1, sn0) # sn0
            z_vals = max_dist * z_vals.unsqueeze(0) # pn,sn0
            weights, mid_sdf = get_weights(sdf_fun, inv_fun, z_vals, pts, dirs) # pn,sn0-1
            z_vals_new = sample_pdf(z_vals, weights, sn1, True) # pn,sn1
            weights, mid_sdf = get_weights(sdf_fun, inv_fun, z_vals_new, pts, dirs) # pn,sn1-1
            z_vals_mid = (z_vals_new[:,1:] + z_vals_new[:,:-1]) * 0.5
        hit_z_vals[inside_mask] = z_vals_mid
        hit_weights[inside_mask] = weights
        hit_sdf[inside_mask] = mid_sdf
    return hit_z_vals, hit_weights, hit_sdf
```


反射光线与单位球交点，到pts的距离dist

```python
def get_sphere_intersection(pts, dirs):
    dtx = torch.sum(pts*dirs,dim=-1,keepdim=True) # rn,1
    xtx = torch.sum(pts**2,dim=-1,keepdim=True) # rn,1
    dist = dtx ** 2 - xtx + 1
    assert torch.sum(dist<0)==0
    dist = -dtx + torch.sqrt(dist+1e-6) # rn,1
    return dist
```

![image.png](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230805163513.png)

从pts采样点，长度dist，方向为reflective，均匀采样sn个点
- 获取sn个采样点的sn-1个中点的坐标、sdf和invs
- 选取其中在物体表面上的点：`surface_mask = (cos_val < 0)`
- 计算这些点的sdf和权重
- 表面采样点的权重之和即为occ_prob_gt
- occ_prob_gt与occ_prob进行求L1损失即为loss_occ

```python
def get_weights(sdf_fun, inv_fun, z_vals, origins, dirs):
    points = z_vals.unsqueeze(-1) * dirs.unsqueeze(-2) + origins.unsqueeze(-2) # pn,sn,3
    inv_s = inv_fun(points[:, :-1, :])[..., 0]  # pn,sn-1
    sdf = sdf_fun(points)[..., 0]  # pn,sn

    prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]  # pn,sn-1
    prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
    mid_sdf = (prev_sdf + next_sdf) * 0.5
    cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)  # pn,sn-1
    surface_mask = (cos_val < 0)  # pn,sn-1
    cos_val = torch.clamp(cos_val, max=0)

    dist = next_z_vals - prev_z_vals  # pn,sn-1
    prev_esti_sdf = mid_sdf - cos_val * dist * 0.5  # pn, sn-1
    next_esti_sdf = mid_sdf + cos_val * dist * 0.5
    prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
    next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
    alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5) * surface_mask.float()
    weights = alpha * torch.cumprod(torch.cat([torch.ones([alpha.shape[0], 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]
    mid_sdf[~surface_mask]=-1.0
    return weights, mid_sdf
```

## NeROMaterialRenderer

### MLP

shader_network = MCShadingNetwork
- feats_network = MaterialFeatsNetwork
    - 51-->256-->256-->256-->256+51-->256-->256-->256-->256
    - 51: pts = get_embedder(8,3)(x) = 8 x 2 x 3 + 3 = 51
- metallic_predictor
    - 259-->256-->256-->256-->1
    - 259 = 256(feature) + 3(pts)
- roughness_predictor
    - 259-->256-->256-->256-->1
- albedo_predictor
    - 259-->256-->256-->256-->3
- outer_light
    - 144-->256-->256-->256-->3
    - 144 = 72 x 2 = `torch.cat([outer_enc, sphere_pts], -1)`
        - outer_enc = self.sph_enc(directions, 0) = 72
        - sphere_pts = self.sph_enc(sphere_pts, 0) = 72
- human_light
    - 24-->256-->256-->256-->4
    - 24: `pos_enc = IPE(mean, var, 0, 6)  # 2*2*6`
- inner_light
    - 123(51 + 72)-->256-->256-->256-->3
    - 123 = `torch.cat([pos_enc, dir_enc], -1)`
        - pos_enc = self.pos_enc(points)  = 51
            - self.pos_enc = get_embedder(8, 3)
        - dir_enc = self.sph_enc(reflections, 0) = 72

| MLP                 | Encoding             | in_dims            | out_dims | layer | neurons |
| ------------------- | -------------------- | ------------------ | -------- | ----- | ------- |
| light_pts           | single parameter     | ...                | ...      | ...   | ...     |
| MCShadingNetwork    | ...                  | ...                | ...      | ...   | ...     |
| feats_network       | VanillaFrequency     | 8 x 2 x 3 + 3 = 51 | 256      | 8     | 256     |
| metallic_predictor  | None                 | 256 + 3 = 259      | 1        | 4     | 256     |
| roughness_predictor | None                 | 256 + 3 = 259      | 1        | 4     | 256     |
| albedo_predictor    | None                 | 256 + 3 = 259      | 3        | 4     | 256     |
| outer_light         | IDE                  | 72 + 72 = 144      | 3        | 4     | 256     |
| human_light         | IPE                  | 2 x 2 x 6 = 24     | 4        | 4     | 256     |
| inner_light         | VanillaFrequency+IDE | 51 + 72 = 123      | 3        | 4     | 256     |

### Render

- `_init_geometry`
    - self.mesh = open3d.io.read_triangle_mesh(self.cfg['mesh']) 读取 Stage1得到的mesh
        - [open3d.io.read_triangle_mesh — Open3D 0.16.0 documentation](http://www.open3d.org/docs/0.16.0/python_api/open3d.io.read_triangle_mesh.html#open3d.io.read_triangle_mesh)
    - self.ray_tracer = raytracing.RayTracer(np.asarray(self.mesh.vertices), np.asarray(self.mesh.triangles)) 获得raytracer,用于根据rays_o和rays_d得到intersections, face_normals, depth
        - [ashawkey/raytracing: A CUDA Mesh RayTracer with BVH acceleration, with python bindings and a GUI. (github.com)](https://github.com/ashawkey/raytracing)
- `_init_dataset`
    - parse_database_name 返回`self.database = GlossyRealDatabase(self.cfg['database_name'])`
    - get_database_split :  `self.train_ids, self.test_ids =img_ids[1:], img_ids[:1]`
    - if is_train:
        - build_imgs_info : train and test 
            - return {'imgs': images, 'Ks': Ks, 'poses': poses}
        - imgs_info_to_torch : train and test
        - `_construct_ray_batch(train_imgs_info)`
            - train_imgs_info to ray_batch
        - tbn = imn
        - `_shuffle_train_batch`

```python
def _init_dataset(self, is_train):
    # train/test split
    self.database = parse_database_name(self.cfg['database_name'])
    self.train_ids, self.test_ids = get_database_split(self.database, 'validation')
    self.train_ids = np.asarray(self.train_ids)

    if is_train:
        self.train_imgs_info = build_imgs_info(self.database, self.train_ids)
        self.train_imgs_info = imgs_info_to_torch(self.train_imgs_info, 'cpu')
        self.train_num = len(self.train_ids)

        self.test_imgs_info = build_imgs_info(self.database, self.test_ids)
        self.test_imgs_info = imgs_info_to_torch(self.test_imgs_info, 'cpu')
        self.test_num = len(self.test_ids)

        self.train_batch = self._construct_ray_batch(self.train_imgs_info)
        self.tbn = self.train_batch['rays_o'].shape[0]
        self._shuffle_train_batch()

self.train_batch from _construct_ray_batch's return ray_batch: 
if is_train:
    ray_batch={
        'rays_o': rays_o[hit_mask].to(device),
        'rays_d': rays_d[hit_mask].to(device),
        'inters': inters[hit_mask].to(device),
        'normals': normals[hit_mask].to(device),
        'depth': depth[hit_mask].to(device),
        'human_poses': human_poses[hit_mask].to(device),
        'rgb': rgb[hit_mask].to(device),
    }
else:
    assert imn==1
    ray_batch={
        'rays_o': rays_o[0].to(device),
        'rays_d': rays_d[0].to(device),
        'inters': inters[0].to(device),
        'normals': normals[0].to(device),
        'depth': depth[0].to(device),
        'human_poses': human_poses[0].to(device),
        'rgb': rgb[0].to(device),
        'hit_mask': hit_mask[0].to(device),
    }
```

- `_init_shader`
    - `self.cfg['shader_cfg']['is_real'] = self.cfg['database_name'].startswith('real')`
    - `self.shader_network = MCShadingNetwork(self.cfg['shader_cfg'], lambda o,d: self.trace(o,d))` -- MLP Network
- forward
    - if is_train: self.train_step(step)




# Relight

```cmd
python relight.py --blender <path-to-your-blender> \
                  --name bear-neon \
                  --mesh data/meshes/bear_shape-300000.ply \
                  --material data/materials/bear_material-100000 \
                  --hdr data/hdr/neon_photostudio_4k.exr

eg: 
python relight.py --blender F:\Blender\blender.exe --name bear-neon --mesh data/meshes/bear_shape-300000.ply --material data/materials/bear_material-100000 --hdr data/hdr/neon_photostudio_4k.exr
```

```python
import argparse
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--blender', type=str)
    parser.add_argument('--mesh', type=str)
    parser.add_argument('--material', type=str)
    parser.add_argument('--hdr', type=str)
    parser.add_argument('--name', type=str)
    parser.add_argument('--trans', dest='trans', action='store_true', default=False)
    args = parser.parse_args()

    cmds=[
        args.blender, '--background', '--python', 'blender_backend/relight_backend.py', '--',
        '--output', f'data/relight/{args.name}',
        '--mesh', args.mesh,
        '--material', args.material,
        '--env_fn', args.hdr,
    ]
    if args.trans:
        cmds.append('--trans')
    subprocess.run(cmds)

if __name__=="__main__":
    main()
```

运行python relight.py后，子进程在cmd中运行以下命令：

```
F:\Blender\blender.exe --background # 无UI界面渲染，在后台运行
--python blender_backend/relight_backend.py # 运行给定的 Python 脚本文件
-- # 结束option processing，后续参数保持不变。通过 Python 的 sys.argv 访问，下面的参数用于python脚本程序
--output data/relight/bear-neon
--mesh data/meshes/bear_shape-300000.ply
--material data/materials/bear_material-100000
--env_fn  data/hdr/neon_photostudio_4k.exr

即先后台运行blender，在blender中运行python脚本：
python relight_backend.py --output data/relight/bear-neon
    --mesh data/meshes/bear_shape-300000.ply
    --material data/materials/bear_material-100000
    --env_fn  data/hdr/neon_photostudio_4k.exr
```

## blender中运行的python脚本

**blender_backend/relight_backend.py**: 

import bpy

**blender_backend/blender_utils.py**: 
