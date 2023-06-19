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
    2. load_???_data() 数据加载
        1. pose_spherical() 坐标变换矩阵，c2w （渲染视频时，相机的位姿）
    3. create_nerf() 创建NeRF网络
        1. get_embedder() 位置编码
            1. Embedder()
        2. NeRF() 构建网络
            包括了一个构建了一个run_network的函数network_query_fn
    4. if args.render_only: 
        1. render_path() 渲染视频
            1. render() 
        2. to8b()  - 将rgbs图片的rgb值放大255倍`to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8) `
    5. get_rays_np() 获取光线 numpy
    6. get_rays() 获取光线 Tensor
    7. render() 渲染
        1. get_rays()
        2. ndc_rays()
        3. batchify_rays()
            1. render_rays()
                1. run_network()
                        1. batchify()
                        2. fn = network_fn = model
                2. raw2outputs()
                3. sample_pdf()
    8. img2mse() 计算loss
    9. mse2psnr() 计算信噪比



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

# load_???\_data()
## load_llff_data()

输入：
- args.datadir ：'./data/llff/fern'
- args.factor,
- recenter=True, 
- bd_factor=.75,
- spherify=args.spherify
- path_zflat=False

输出：


## load_blender_data(basedir, half_res=False, testskip=1)

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
```
render_kwargs_train = {
    'network_query_fn' : network_query_fn,
    'perturb' : args.perturb, # 默认为1 --perturb：在训练时对输入进行扰动
    'N_importance' : args.N_importance, # --N_importance：每条射线的附加精细采样数
    'network_fine' : model_fine,
    'N_samples' : args.N_samples, # --N_samples：每条射线的粗略采样数
    'network_fn' : model,
    'use_viewdirs' : args.use_viewdirs, # --use_viewdirs：使用全5D的输入代替3D的输入
    'white_bkgd' : args.white_bkgd,
    'raw_noise_std' : args.raw_noise_std, # 默认0 --raw_noise_std：添加到输入的噪声标准差
}

# NDC only good for LLFF-style forward facing data
# NDC 全称是 Normalized Device Coordinates，即归一化设备坐标
if args.dataset_type != 'llff' or args.no_ndc:
    print('Not ndc!')
    render_kwargs_train['ndc'] = False
    render_kwargs_train['lindisp'] = args.lindisp
    
render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
render_kwargs_test['perturb'] = False
render_kwargs_test['raw_noise_std'] = 0.
```
- start ：global step
- grad_vars：model的参数列表，包括权重和偏置
    - `grad_vars = list(model.parameters())`
    - if args.N_importance > 0: `grad_vars += list(model_fine.parameters())`
- optimizer
```
optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

if 加载了ckpt ：  optimizer.load_state_dict(ckpt['optimizer_state_dict'])
```

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


## get_rays_np(H, W, K, c2w) numpy版本(narray)

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



# if render_only:
## render_path()

输入：
- render_poses, 测试集的渲染相机位姿  200 * 4 * 4 
- hwf, 
- K, 相机内参矩阵
- chunk, `args.chunk=1024*32`
- render_kwargs = render_kwargs_test
- gt_imgs=None, 
- savedir=None, 
- render_factor=0

输出：
- rgbs, 200 x W x H x 3
- disps，200xWxH

```
render_path(render_poses, hwf, K, args.chunk, render_kwargs_test, gt_imgs=images, savedir=testsavedir, render_factor=args.render_factor)
```

### render()

```
rgb, disp, acc, _ = render(H, W, K, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)
```

### 


需要对render输入的光线做batch: `rays = batch_rays`

#### 在train()中

<div class="mxgraph" style="max-width:100%;border:1px solid transparent;" data-mxgraph="{&quot;highlight&quot;:&quot;#0000ff&quot;,&quot;nav&quot;:true,&quot;resize&quot;:true,&quot;toolbar&quot;:&quot;zoom layers lightbox&quot;,&quot;edit&quot;:&quot;_blank&quot;,&quot;xml&quot;:&quot;&lt;mxfile host=\&quot;Electron\&quot; modified=\&quot;2023-06-19T03:20:26.056Z\&quot; agent=\&quot;5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) draw.io/14.6.13 Chrome/89.0.4389.128 Electron/12.0.7 Safari/537.36\&quot; etag=\&quot;B7AM0VIngZ7lAPHURmRf\&quot; version=\&quot;14.6.13\&quot; type=\&quot;device\&quot;&gt;&lt;diagram id=\&quot;bZodXSPFXZZJwMvDvhFf\&quot; name=\&quot;第 1 页\&quot;&gt;7Vxbb6M4FP41SO1DK+4hjyFNdx52Ryt1tJ15ipzgEHYIzgKZJPvr1wabm01C2obLNtJoCr5hvuNz8flMJG26OfwWgu36D+RAX1Jl5yBpT5Kqqpam4z+k5JiWKKoqpyVu6Dm0LC948f6FtJA123kOjEoNY4T82NuWC5coCOAyLpWBMET7crMV8stP3QIXcgUvS+Dzpa+eE6/TUksd5eVfoOeu2ZMVc5zWbABrTN8kWgMH7QtF2kzSpiFCcXq1OUyhT9BjuKT9nmtqs4mFMIibdHA27vyvb64+//a6kr/9+2VvTF8fjHSUX8Df0Remk42PDAE3RLutpNkrFMRUPoqF72lPGMbwIBIFWLARZH6qSgYAXjoQbWAcHnETOpAm0y501YzoNPe5BJQRbbIuos8KAZW6mw2dA4MvKDYX4KScx2m/9mL4sgVLcr/H6oAxWscbn+CFL0G0TdfnyjtA5xI8i7jVC5EH88jA5MHTReDp1wJPHSB4bJTyQlQ7x1IbLpZqGUulcyx1Dssvh9eDNhhAu9dss5fuY9w37zHqs9KaPfce1gDB66v3GA8Xy955DzaDivvQB4No96qtiMJq08fPtaMtCEowmv/syE7JXiIfhZI2wZWhu7jDU8P/8OPlwtU9uSTAyQTLhxXYeP4x7YMHApttUqnhzSmWBPR/wdhbAq6mPEiUSIQMoVjbQ6UunSWpDFC4AX65ek/BJPV6Os+k0odxDMMH/KpLL3CF/bHI4wfge26QVi+xwGFYrvYCJ1kGpF4uTC2pjEMQRCs8KBs+gFmDPQqd8tOL3Rdg+ZM478B5qGCu6laGtaqP82ujgLzjRVsfUNS9wPcKD175CMTFCTHh4iuX/P2KRZ4glS4GvLjS9ZDWchpG3hSXn1G0N+nV6UhD0YySVRKplGrwKqVdTaP46BU6LnyhtyiM18hFAfBneamdCJmYnicin7zN7whtKXZ/46V6pOCBXYzKyMKDF38vXP8gQz0a9O7pQEdObo7sJsDv+714U+hFbvNuyR3rJ5YiecnTMsSYoF24hA0segxCF56yq7p4UYTQB7H3qzyRjxcxv225Gc2b0RyO0dT0vhlNq0ujySzjZUYzt5M/SmayfaNpNjSaSk0yox2rachdyvhtjvF9Mo6wROIJoUBwSWIxaNmzR6B7kjtaB7rVqffkU1a7CM4XIF6uiUWvLpJwjTaLXdSJoVStSv5gxFtKVbRjs65mKUedWspMc34UagZjKfWmhrJm296SgvA5oiOM6qKHgpSiOEQ/4TSNepjBWWFbUymigdkTjco0m+gIDiX9Ca3YeI6TLByRxpUX0zWUTqso3ZhXOpHOqdfSOfUW79/i/SHH+3pZoTSj63hfbXzqoe7l389RVVy7Zgpcuzx6FMCSpb4/Hpdek8vq5cccLIGdVq4WHDG0hgVfc6qqZTR7feTmDJoNyKp20VR4kxeCYzTHnor5j0XIXMddiCTVDh3yn7u4P/TtVMQ58DOz2R3ejBe4bZTesFFipvzsTknrNBHPpnmRTnWqTWfIrDzeoHbLlDs3XNqNIr7tfkS7H2+O5+0FSYcnAvkAdkJVutgUhVmtboU0PqjqqZ+qiuOtfusD/RQLqs77KVm8LBr7Kdr1T+ThOWbLyZTFq4mNkM6LdqoslGwW7zg+e/qsAU33DTqoKXEoCPsgqcyhkFk4IFon76d8cBTUNF+sq51GQXx68g7bYhLm3B/O7y34FgOJjazut8hag+PXPcBO1ZtgJ8qpX+8rigYnsmHgMM1f+Gj5s2KVAqfAo2bfdgk91RuI/iaG5yNtTQ3lVJCPIZAPK3unH9Mq6RLLrMg9nT/nyPiBKqlmq7qAruwRTXMo0dRADxloVtOIa9ypT+S/WbDJAQN8+XWOd1tOurGxD3e4la3cC/IB/19+daxWvYGAEMp2PK1QrBrPeScHQuYkn5PtMNWDTQRVjVfStTjP29mfW5xjQbZbKM7rfden8t59ZkqTZ8l6lma6ZFvSZAK3CCvkzJDGsmTNovVutcIoprU4YqMX9leWZqgINAdW6SaVoFaDKkUWcIjCDzquFpAafLIuQJ9HF4y+HR0xPt1xrZbjEb1pPGJ0ylTofDxCZ8x5s96yEwa3A9eMzmk+Q+BpLmcnKDVBeQkxKZFzELV8wymyoZ5pOEcznOIY6gmG0+zCCWqhCa+QkQoZo1BLJwi5hIREqNAJBQ2WvQ1wYSQZdrB9JNE62jwu18hbwjtKOtxLxlMt2ZBr1E34rQm/Ks66oKP9QM2sJDc0Qe4rO/3VSnhs8BnbhLfGkdo0uXAG7RZ6QFobDQ4YDgY7tdWIdRgpbWMk9wy3vvxKxmU7pJHg1GaruDHD23PcTI5CGYmyLK0i15fzmWeQMznkhEet20RO9EFHdZt+VfLp5Fa8ZerJrJFfO9TTyNAeRzWadSn7NKoat5bPY5iCqGM2lsaTJO1pSuOpZE9ICY57rDFJfeLysUkZCVXOUqNEklgjfT4MZHGinAblWVdyJEyQGmfRZE3zz5IXHFW2AKYgLyjMkBtXM0F9+UWyM8a78ktuGZ/bmenmwywRUUSX+Dn1KDFHNwWRTp4P+SgFwbf5b+6mpjf/6WJt9h8=&lt;/diagram&gt;&lt;/mxfile&gt;&quot;}"></div>
<script type="text/javascript" src="https://viewer.diagrams.net/js/viewer-static.min.js"></script>

# render()

```
(H, W, K, chunk=1024*32, rays=None, c2w=None, ndc=True,near=0., far=1.,use_viewdirs=False, c2w_staticcam=None, **kwargs)
```

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
eg:  `all_ret['raw'] : W * H * N_samples +N_importance * 4`
eg: `all_ret['rgb_map'] : W * H * 3`


## 渲染流程图

<div class="mxgraph" style="max-width:100%;border:1px solid transparent;" data-mxgraph="{&quot;highlight&quot;:&quot;#0000ff&quot;,&quot;nav&quot;:true,&quot;resize&quot;:true,&quot;toolbar&quot;:&quot;zoom layers lightbox&quot;,&quot;edit&quot;:&quot;_blank&quot;,&quot;xml&quot;:&quot;&lt;mxfile host=\&quot;Electron\&quot; modified=\&quot;2023-06-19T05:46:35.107Z\&quot; agent=\&quot;5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) draw.io/14.6.13 Chrome/89.0.4389.128 Electron/12.0.7 Safari/537.36\&quot; etag=\&quot;Q9G_GM6F3ujZ4JYMg4De\&quot; version=\&quot;14.6.13\&quot; type=\&quot;device\&quot;&gt;&lt;diagram id=\&quot;SaKEZgpW3Xd3I7sOL0pk\&quot; name=\&quot;第 1 页\&quot;&gt;7R1bc6O2+tcwm3QmHu7gR1+S3Zme7mm7Z7qnTx5sY5uzGFzAcbK//khCAiR9tnECNkncmW6MJJD47jcJxRitnz4n3mb1Wzz3Q0VX50+KMVZ0XTMNFf3BLc95i62becMyCeZ0UNnwLfjp00Z633IbzP2UG5jFcZgFG75xFkeRP8u4Ni9J4h0/bBGH/Kwbb+lLDd9mXii3fg/m2SpvdXWnbP/iB8sVm1mz+3nP2mOD6ZukK28e7ypNxr1ijJI4zvJf66eRH2LgMbjk9z3s6S0WlvhRVucG77P/FD5Gz9/+UhePm1/T2V9L444uNs2e2Qv7c/T+9DJOslW8jCMvvC9bh0m8jeY+fqqKrsox/4rjDWrUUOP//Cx7psj0tlmMmlbZOqS9/lOQ/Rff3rPo1d+VnvETfTK5eGYXUZY8V27Cl39X+8rbyBW7bxFHGV2IZqBrGWoUkGm8TWb+AVAx6vOSpZ8dGKfn4zAcKxNQnHz247WP1ocGJH7oZcEjT2ceJddlMa7EKPpBkXoCgulzH71wS2dae0GEWkKMLXQz/p0lpE2ghBLPGDW7VZD53zYegdEOMTuP09pwfvSTzH86CBnaa1mUc54Zb9HrXcmIRduqwoSm2hIwzUtyi1bhlZJzjnELxysl67TOLXpNbjE6xS26xC3ozed+0n3ecC/NG/ZVk9TlDaMmb1id4g1D4g20+kniPaed4w5HFbnDkrjDAZjDaIs5GCdeiDtepDkuxR1WTe5wO8UdlsQd0Xz2Nrij0AkX4w7jqjvqcodbkzs07VLssXncjf/8t7P68vB898fU/PEzjfw747KO5tkRPPfSVcHijWIbBK9pNo1tcusgQQKsMmATB1GWVp78O24oJYvLCxZDjEgcHq6Z/cPjbdU6eAP6ka+4pNPi1V8unVxJsk+9bLYKFs/dFO+ia6Bf3DXQLuo3vy35rtUNM2kXizOBEkh337eA5wR6u+iG4eu0IuElGWsYYtBNE8RCvlR6W0k1p+oKXZjIOCL8xfHFwmprC+GGdrQFQ6gUQnobysKwLq4snKuyqC096kZZtW6FkswLO3x6+0huS1/UxXhLHsFRKevq7agLy3G4eUz3sPQ3hCDHqeMdxzpJuwjjW1IuuqRcIj/bxcmPyT9bP3meLCJFt0NEF8NpwvGX/c8W55kJJd6lhBQHaIBmbJ4IPbJ+9GtJ/uKb1GQbTegEnddcpn5xzXVZE/htaa66UV7NbkWOnSp+HINnd8s8zfgUxrckHuQgtL+e+nMiFjrGvnZfAJAqs68LcG9rQWj9GoSuz712Te7VLxakOLhuIJaHWpV7S3FdxbWVe1tx+8qwr9ybyhC1qLhlMFIG2iK6ma220Y9f0HjbuMW3DExl4OAfQ1VxHzrHaEzyFIxmA3rSADjNao3TNAlIXS+j0Tg+O1pG05Lxzyyct+busXULpQNEN3WxuMZ2nR6vnUxNLiBgCqxp7QTnz67lNfXZ5HI+8uvYRPbusOGmkp8qc/QqTWtabZ1fffX/fLjp9Xq3neMny+S5yTZkbtL0s6qgy7LTG1ZBdf22jll+uuwaYX7pHKuIbpFtyqxyXrfo3Qc12uIUpy6ndMxYcyROmSW+l/mTyE8WnWcYC6h2OCvDGLKxe/MdXWOP8cst/YGYSo1xhFbuwgnERkK4AUKWuk39yWPg7+ZBIicfEYgzHkFplsQ//FEcosUZ4yiOMDcvgjAUmrwwWEbocoYQhwx4Y4gRFsy8cEA71sF8TkQBRBG8eGiBKAyhskiHspo2QBVi8qI5qpDTwnkgQaYIrlmTfeUPgzUwFw0VpraHNbnmnQSJLGVo09jQYIwxRy+GmCFxnn+C9NnYwB36iDXN86YRseW9hOCODViw6w+Da0fg0D6A6/5ZcW1KuA6izRYnBdRc0vKc+XWSeutN6KN+G/1AYImmKf6DOw2xkWD6ZLEuyfRSlNM1kSXRFckB7PdLPjWSfzqU/GuPfGQHAyQfkXpuS+S9njyIypdo5OjECCbDj0U/pi3QDyB+ikTxeehHTk3Q+Oi8JiLtRkjoRqShbOUjfA1um3g4/0aHX4dh5COQo1h8YgGWjw7laNojR7nqfQ85vor8IL+lSQqEng8R4YckOiExaOtAVPa8JlhfIro7dHmE8MzOiz35JaQ3eBNEh6/ZdAoWWv2+qjZEjEJ1tw1tWmmLGOEtVRJSup6kLi7OfdYDy6Md3cRwscMe4NVcK82bR3Hz+1ReV2kue/WJt9PjbZb7Zh2LZotFra4D6OTzHnkj+yVmruuQQhwt0f9THMIaPSjuB/IfLcFgdw1ZXZ3VdDJlex3RuQKFiQ4Grhoxpn5O0DLSwzO/ehIaU70GwgSBAXiO6jkFhiVnOrD9myynkzWCCQl3K7aqWEOKshHBmDXGjbguw9BUajPPg3RDbsIYluOplYdAd3uzWXHz0cE7AppUHkyWVxIuuEx/k60OTgVT44nqpU3rW4jG96G9FJA8c9uyvuXEy9dJsN4ge8uLELwkvb2K19NtehGd7aj8ziTXAnQ2yySdBXjX81RAwxaEFVDPBMNUv5Rde2jZFf7IJdRkM+9ejYajuUI9bf+cZ1KAAJTN2r/9Q/6AykMJYAWQZURo+tF8gE+lRdfTMJ79yJseArx4SufsEFy3sFLYwbPEbmGnyOLBoTf1w6E3+7EkCxVMl+M8f06WfTmLAsflHeKJoxxKqVLtaRorr6BkeccOJ37pplg2JF4sUp/b7yrvKBN1rkjqOVykjbPHH+SqPVXVLNNy+4aj9V3+sXv2475gDxqIBNkPyb2BCbKTQWO9mbQVs9+sYa/XQ0YbouHBnYbtLzonbz4KqQu9iSUcNFDerQPiuJJsB6q7dR2wf2yrJdkOpxF4T/TN4apF87/Ys89OSlGBzZit1cvB5pZ1NWHr6kfoTAh44MX2uxxc9/U0A8hE1lTgWM7z2sja2wqeOxrviCNgXRyCspuxXw3RMqxKlPSjRTNdTVREQFz9zIpI3g5x2JjYg8Vrjh+ZGbKMg8LVJmArtoffN/MFlAbPn6+7j5wN7IjBYF6NwtpGIbT97ExHB7/uHFFZ3mZxMlv1Zl7GDtkgB3QMHvBpG3QTxiCXx/jcjqEyHJEf98rAUu5dZYhucnBLf4xbuma1SF/yADcrndVqAbawJTiapeZ2tOIMP9F03qcipMPl9z6xzF3ZX+Ty8gE0OVf2s2ydM27K5qePmQePr3oOKQxEr4/T6SSQFOa9D+TBzc4F7xoRmzAuLIQCb/eJBbiIe4Pz/eLgQ6tNN14ELhdbJnfUyCDhOmZnKPgDas2AX5wj9BfZy2bY+xr1H2Qw8B0ulyhMO87e48omEKTz9fBrhBGwZ3ADVNQMYAjpS9YtJTE6laryXFH/7c/LLMupynNLLq8maocWiWUkv0omNTu1TCSq+VVS2d2pRf6cpNmcW2VuRqDWm58lS48IkNdoBP5SHr7aRtPAS4lB8OCFqX8LTkt2TVnDr/mpztZ435vDPt/5rQxN9KyL81YqVoYJ7Xgq3O3mzQy5yIPU7YThpNvmxl6J+6p6NvpG+Q6H7798aXRnYCMrlO2O6lpfpHaOMnljKrgJ6J1qMenqfpOmEYO1FTyfaLQ0IuJaDDtp4mdmoL3mJrS7rt+a5JMD63JQ8Ro/FDftCp6yA50ZYAF4bC14CGz6PqXWtQk392V1sk3NfLTGtqmJjtXnihF4uWK3MWC/+Wrf4uPlhS0IbTeGDk1qrWLVBL6astcfr+J6CMjJRlC9fw/Dvukbmfa6qyGnM9FbATK5hq71gCNYW4uKQlvJ/GwSBimp/Isw1nDDPJhl3Y8yW1Bm9axRZmBn2VE7e5aTMfUAb+jpSezPbcXaXnjrIHzOh6788NHH5L3fGq905JPinihO1l5Y6dtRuOBOk4pjNfQzxEN3aM2zIFrKd+6PdJKeANFFRJ+pspWQnizxonSBnsSeSVgXoyxO5vx8xY3Tohj4ToCVjreIYTDpZp/+sBjEsAERehRaQRQGbKZFGHuZML0osgp/vU3/oAnzUfQDoE8OQLUFrR3qbF0492zatlLJaPZUU1fa3mluHbBt8GS/+0mAoIuV0ivLxs26GVDrYl+SPrhuQC42lRnA4zF3ezP+OaWgrNz6SnsZyZ7CUmaiwqpE9Dj7HAEsX9uZEghnhwaWtBA4ygjmx4IHciAhcBTx2o8FDaQaE49zcknwGwdccz2e+0R5TLvX69FmvOXBGfNZiDqA6rqjaji2WDAGmq3QztTWshbATnuBVF4Ydv2S4/V7/ufX/M9M3+1F6qvRZzWDJvbpuwJJ0BdmINeiNcPKlHeANIMjotDy813vH3ChUX9wQxFG0WdwJ0YJd82P39UOlltkUnH7mQZF16ETCJr4uANsz8nlS21w6CKeIRevPM+3POk35k79fWvsqwMiVofC6u35RXI0sHH2PcSkUp8YVb/5Oh7hYsT+WOnn3wZTFddR7h1laCjDIfts2PD9cDVYlNjWJ1tgopDDb0fDRVIkqEAkApdhClYc36PsS96amLK4PilixHWLQSP9eNyIDTmSVgbjR1wnFEJiA+AoEus9EkhSK7EktRJOYrdDEaUCKkJQ6ZjpPStIvBxo4P8WC/neMiCfIxzzpK70EXP2MYu6DmXXIWHX/kBxTVIbTCqL6ZiHI3Y0JxhesdJKJqG62OE9tNhW10hFh4wJXHHtKgMb3XhHV0hhxg68x+fho2WPKyssxrj4XfoGGzM4ec3NBxObUpoarzSh8yygTxU4rYlH4PzxeuWbUsb4o2SZbGHbswvpuLPWE1j7fcsTUiJlVqRMjHAaqWmNaFxOI5JkynvXh/9ZBTi9Tf7xwjRWyPHE6N9iJ64a57sU0LsnN7ftSc/Gjk80xMCODXkdDQUN0GUSY4gWfZ/RW65+I59PNO7/Dw==&lt;/diagram&gt;&lt;/mxfile&gt;&quot;}"></div>
<script type="text/javascript" src="https://viewer.diagrams.net/js/viewer-static.min.js"></script>

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

仅需要对LLFF做Projection 变换到NDC坐标系下

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


## batchify_rays(rays_flat, chunk=1024\*32, \*\*kwargs)

输入：
- rays_flat: (WxH)x8 or (WxH)x11
- chunk=1024\*32
- \*\*kwargs


输出：
- all_ret: 将ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map} 拼接起来

**ret：chunk(1024 \* 32) --> all_ret：800x800**

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



# img2mse() and mse2psnr()

```
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
```

 数学公式： 
 
 $\text{MSE}(x, y) = \frac{1}{N}\sum_{i=1}^{N}(x_i - y_i)^2$

 $\text{PSNR}(x) = -10 \cdot \frac{\ln(x)}{\ln(10)}$