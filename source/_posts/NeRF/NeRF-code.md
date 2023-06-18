---
title: NeRF代码理解
date: 2023-06-15 14:50:33
tags:
    - NeRF code
categories: NeRF
---

[NeRF代码](https://github.com/yenchenlin/nerf-pytorch)，基于Pytorch

<!--more-->

code流程：(按照函数)
1. train()
    1. config_parser() 命令行参数输入
    2. load_blender_data() 数据加载
        1. pose_spherical() 坐标变换矩阵，c2w （渲染视频时，相机的位姿）
    3. create_nerf() 创建NeRF网络
        1. get_embedder() 位置编码
            1. Embedder()
        2. NeRF() 构建网络
        包括了一个使用run_network的函数network_query_fn
    1. if args.render_only: render_path() 渲染视频
        1. render() 
        2. to8b()  --> to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)
    2. get_rays_np() 获取光线 numpy
    3. get_rays() 获取光线 Tensor
    4. render() 渲染
        1. get_rays()
        2. ndc_rays()
        3. batchify_rays()
            1. render_rays()
                1. run_network()
                        1. batchify()
                        2. fn = network_fn = model
                2. raw2outputs()
                3. sample_pdf()
    5. img2mse() 计算loss
    6. mse2psnr() 计算信噪比


# config_parser()

```
def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()
    # 此处命令行更改的参数为 args.config
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--expname", type=str,
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/',
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern',
                        help='input data directory')
    # training options
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')
    parser.add_argument("--lrate", type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument("--no_batching", action='store_true',
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    
    ...
    return parser


use：    
parser = config_parser()
args = parser.parse_args()
basedir = args.basedir
```

# load_blender_data(basedir, half_res=False, testskip=1)

输入：
- basedir，数据集路径'E:\\3\\Work\dataset\\nerf_synthetic\\chair'
```ad-info 
title: chair 文件夹
collapse: True
- chair：
    - test
        - 200张png 800x800x4
    - train
        - 100张png
    - val
        - 100张png
    - .DS_Store
    - transforms_test.json
    - transforms_train.json
    - transforms_val.json
```
- half_res，是否将图像缩小一倍 （下采样）
- testskip，测试集跳着读取图像
输出： 
- imgs：train、val、test，三个集的图像数据   imgs.shape : (400, 800, 800, 4)
- poses：相机外参矩阵，相机位姿 poses.shape : (400, 4, 4)  400张图片的4x4相机外参矩阵
- render_poses：渲染位姿，生成视频的相机位姿（torch.Size([40, 4, 4])：40帧）
- [H, W, focal] 图片数据，高、宽、焦距
- i_split，三个array数组
    1. 0,1,2,...,99 （100张train图像）
    2. 100,101,...,199 （100张val图像）
    3. 200,201,...399 （200张test图像）


# create_nerf(args)

输入：
- args，由命令行和默认设置的arguments共同组成的字典
``` py
parser = config_parser()
args = parser.parse_args()
```

输出：
- render_kwargs_train
- render_kwargs_test
- start ：global step
- grad_vars
- optimizer

## NeRF(nn.Module)

继承nn.Module构建的类

def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False)

>[Pytorch 中的 forward理解 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/357021687)
```
model = NeRF(...) 实例化
model(input) 相当于 model.forward(input)
```
- D=args.netdepth, W=args.netwidth
- input_ch=input_ch, output_ch=output_ch
- skips=skips,
- input_ch_views=input_ch_views
- use_viewdirs=args.use_viewdirs



# get_rays_np(H, W, K, c2w) numpy版本(narray)

通过输入的图片大小和相机参数，得到从相机原点到图片每个像素的光线（方向向量d和相机原点o）

输入：
- H：图片的高
- W：图片的宽
- K：相机内参矩阵
```
K = np.array([
    [focal, 0, 0.5*W],
    [0, focal, 0.5*H],
    [0, 0, 1]
])
```
- c2w：相机外参矩阵
```
c2w = np.array([
            [ -0.9980267286300659,  0.04609514772891998,  -0.042636688798666, -0.17187398672103882],
            [ -0.06279052048921585,  -0.7326614260673523, 0.6776907444000244, 2.731858730316162],
            [-3.7252898543727042e-09, 0.6790306568145752,0.7341099381446838, 2.959291696548462],
            [ 0.0,0.0,0.0,1.0 ]])
```

输出：从相机原点到800x800图片中每个像素生成的光线 $r(t)=\textbf{o}+t\textbf{d}$
- rays_o：光线原点（世界坐标系下）(800, 800, 3)
- rays_d：光线的方向向量（世界坐标系下）(800, 800, 3)

# render()

`(H, W, K, chunk=1024*32, rays=None, c2w=None, ndc=True,near=0., far=1.,use_viewdirs=False, c2w_staticcam=None, **kwargs)`

渲染流程图

<div class="mxgraph" style="max-width:100%;border:1px solid transparent;" data-mxgraph="{&quot;highlight&quot;:&quot;#0000ff&quot;,&quot;nav&quot;:true,&quot;resize&quot;:true,&quot;toolbar&quot;:&quot;zoom layers lightbox&quot;,&quot;edit&quot;:&quot;_blank&quot;,&quot;xml&quot;:&quot;&lt;mxfile host=\&quot;Electron\&quot; modified=\&quot;2023-06-18T08:47:40.341Z\&quot; agent=\&quot;5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) draw.io/14.6.13 Chrome/89.0.4389.128 Electron/12.0.7 Safari/537.36\&quot; etag=\&quot;FGUkbtxcUlMtdqIB7Mou\&quot; version=\&quot;14.6.13\&quot; type=\&quot;device\&quot;&gt;&lt;diagram id=\&quot;SaKEZgpW3Xd3I7sOL0pk\&quot; name=\&quot;第 1 页\&quot;&gt;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&lt;/diagram&gt;&lt;/mxfile&gt;&quot;}"></div>
<script type="text/javascript" src="https://viewer.diagrams.net/js/viewer-static.min.js"></script>


输入
- H
- W
- K
- chunk=args.chunk 同时处理光线的最大数量
- rays=batch_rays  `[2, batch_size, 3] ` 
- verbose=i < 10   
- retraw=True
- render_kwargs_train
```
render_kwargs_train = {
    'network_query_fn' : network_query_fn,
    'perturb' : args.perturb, # 默认为1 --perturb：在训练时对输入进行扰动，是否分层采样
    'N_importance' : args.N_importance, # --N_importance：每条射线的附加精细采样数 = 128
    'network_fine' : model_fine,
    'N_samples' : args.N_samples, # --N_samples：每条射线的粗略采样数 64
    'network_fn' : model,
    'use_viewdirs' : args.use_viewdirs, # --use_viewdirs：使用全5D的输入代替3D的输入
    'white_bkgd' : args.white_bkgd,
    'raw_noise_std' : args.raw_noise_std, # 默认0 --raw_noise_std：添加到输入的噪声标准差
}
```

输出：`ret_list + [ret_dict]`
- rgb: `all_ret['rgb_map']`
- disp: `all_ret['disp_map']`
- acc: `all_ret['acc_map']`
- extras: `[{'raw': raw , '...': ...}]`

## get_rays(H, W, K, c2w) torch版本(tensor)

通过输入的图片大小和相机参数，得到从相机原点到图片每个像素的光线（方向向量d和相机原点o）

输入：
- H：图片的高
- W：图片的宽
- K：相机内参矩阵
```
K = np.array([
    [focal, 0, 0.5*W],
    [0, focal, 0.5*H],
    [0, 0, 1]
])
```
- c2w：相机外参矩阵
```
c2w = np.array([
            [ -0.9980267286300659,  0.04609514772891998,  -0.042636688798666, -0.17187398672103882],
            [ -0.06279052048921585,  -0.7326614260673523, 0.6776907444000244, 2.731858730316162],
            [-3.7252898543727042e-09, 0.6790306568145752,0.7341099381446838, 2.959291696548462],
            [ 0.0,0.0,0.0,1.0 ]])
```

输出：从相机原点到800x800图片中每个像素生成的光线 $r(t)=\textbf{o}+t\textbf{d}$
- rays_o：光线原点（世界坐标系下）(800, 800, 3)
- rays_d：光线的方向向量（世界坐标系下）(800, 800, 3)


## ndc_rays(H, W, focal, near, rays_o, rays_d)

输入：
- H
- W
- focal
- near
- rays_o
- rays_d

输出：NDC坐标系下
- rays_o
- rays_d


## batchify_rays(rays_flat, chunk=1024*32, **kwargs)

输入：
- rays_flat
- chunk=1024\*32
- \*\*kwargs


输出：
- all_ret: 将ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map} 拼接起来

ret：chunk(1024 \* 32) --> all_ret：800x800

### render_rays()

输入：
- ray_batch
- 继承来自render()输入的参数：
- network_fn,
- network_query_fn,
- N_samples,
- retraw=False,
- lindisp=False,
- perturb=0.,
- N_importance=0, 每条光线增加的采样数
- network_fine=None,
- white_bkgd=False,
- raw_noise_std=0.,
- verbose=False,
- pytest=False

输出：
ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map}
```
if retraw:
    ret['raw'] = raw

if N_importance > 0:
    ret['rgb0'] = rgb_map_0
    ret['disp0'] = disp_map_0
    ret['acc0'] = acc_map_0
    ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]
```


#### run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024\*64)

输入：
- inputs 
- viewdirs
- fn：network_fn = model ，经过一次MLP训练
- embed_fn
- embeddirs_fn
- netchunk=1024\*64

输出：

- outputs: chunk * N_samples * 4 (每条光线，每个采样点的RGBσ)

在这里调用了run_network函数
```
    network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                netchunk=args.netchunk)
```

##### batchify(fn, chunk)

输入：
- fn : model
- chunk : 1024* 32 

如果chunk没有数据，则返回fn，否则：
```
def ret(inputs):
    return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
            # 每个chunk的大小为1024*32，即每次fn处理1024*32个点，最后返回结果
```

#### raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False)

体渲染在其中
chunk = N_rays，射线数
输入：
- raw: chunk * N_samples * 4
- z_vals: chunk * N_samples
- rays_d: chunk * 3
- raw_noise_std=0, 
- white_bkgd=False, 
- pytest=False)

输出：
- rgb_map, `[chunk, 3]`
- disp_map:  `[chunk]`
- acc_map: `[chunk]`
- weights: `[chunk, N_samples]`
- depth_map: `[chunk]`

#### sample_pdf(bins, weights, N_samples, det=False, pytest=False)

输入：
- bins, chunk * 63
- weights, chunk * 62
- N_samples = N_importance
- det=False  = (perturb\==0.)
- pytest=False

![image.png|555](https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/20230618145142.png)

输出：
- samples:  chunk * N_importance
