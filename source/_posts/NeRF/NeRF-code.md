---
title: NeRF代码理解
date: 2023-06-15 14:50:33
tags:
    - Code
    - NeRF
    - Python
categories: NeRF
---

[NeRF代码](https://github.com/yenchenlin/nerf-pytorch)(基于Pytorch)。流程图：[train流程图](#train流程) and [渲染流程图](#渲染流程图) (基于Drawio)

<!--more-->

# train流程

<iframe frameborder="0" style="width:100%;height:1724px;" src="https://viewer.diagrams.net/?highlight=0000ff&edit=_blank&layers=1&nav=1&title=train%E6%B5%81%E7%A8%8B.drawio#R7V1bk6O4Ff41rkoe4gKJ6%2BP0TM%2Fmsj1JZVJJ9sklG9kmDYgVcru9vz6SARss2aOYxoIUVVM9Rki2ONJ37jrM4Of0%2FSeK8u0LiXAyA1b0PoNfZgBA13f5f6LlULbYrmeVLRsaR1XbueF7%2FBuuGutuuzjCRasjIyRhcd5uXJEswyvWakOUkn2725ok7V%2FN0QZLDd9XKJFb%2FxVHbFu2BsA%2Ft%2F8Rx5tt%2Fcu2F5Z3UlR3rp6k2KKI7BtN8HkGP1NCWPkpff%2BME0G9mi7luK9X7p4mRnHGdAZ8%2B4ePf8qfQ7b6y28L9mf89PLPT3%2BoJluwQ%2F3AOOLPX10SyrZkQzKUPJ9bnyjZZREW32rxq3OfnwnJeaPNG%2F%2BDGTtUi4l2jPCmLUuT6i5%2Bj9m%2FxfC5W1390rjz5b365uPFob7IGD00BonLX5r3zsOOV%2FW4NclYNRHb49fl84qHvErGqqkgO7rCN2hXb0dEN5jd6AdOi81hgkmK%2Bfz4OIoTxOK39jxQtV03p37nFeUfqkX9Hxa4%2Bt43lOyqX2IUxdnvfi8t%2FHlZxUrstzHD33N0pMCeg7u9hGqyvmHK8PttwsqEqAc4YF6zi4pbBBV09mfk2TVj2DZQ51p9Uc%2BSyDThQxcfQBMf0CQ%2BgIQPLknW8WbBn7jA9CZOLDM4gf4lTmxPARSgAIrXG1CCCSh3AwVqAsW%2BsjEegxQoISUhKFrM4Nfy3yJCDI0DLyeBYQwvRuFiN8Byho4aLj1ue0dz2%2Fsmd72j3vXLBPO1oEPd89Cy5xpb3lNsedjXlvckSkpk4yZSLj6udjQ5PFG0ehUb40f0ayuuPVAT%2BJJmerLqGuRUMZATV%2FlwcvoSOXNS4EWRbzGNuf06wF3pBMPblTYcCyc%2BKy4tteWsxRiwgHVNYNszycNPzzMpp32ucWhyjW3Z0bGiGDG8yDBdD5AZKtRS6JtWS213Qsr9SNF1eNiOUaTILo9v%2BO9fBwgRaEkQcaCrB5HeFAYwuQQ7QETX5jMsTGSrL8NsT%2Bjr4tcdpofFOuN3jzcsussW1c0BQgiEkuniqLzq7kO96v4Eofsh5OlCCBiFkGzu8%2BkucLrEUTRIv7pC2sDAuLQZoXk6HKiEmlABRqECZUfYxA4%2FfI2h0eBJPc0GO4zX4ofoppjT0pdMsoQT7NMMeCgV7C5bFvmJWE1GuSXpclcYYZKh02KQriMzSMD5qCuzyKA3FjlpE%2FfDB%2BjarMA1CZ96mg34VKDJEdsOUZkAkt7tAePKhGzWHHAh0Y4%2FNWsTqGCUvOLPJCGUt2QkExBax0ly0YSSeJPxyxWnEubtT4KGIhjxqbqRxlF0xJ9qRdpr1seiQGlR3FA2hhzFmvQWxgGjSZzokwu5ulzIbBzBvcKFBAP6kdAeIENSOQIeypCgUV%2FayINvQNcRAIz60oDsCGAkWI5DZvvQNESAbDhc8po0TveUE0vsBSK6xhEmg6OuIydR%2BLoakdMbAzIT7%2FogO%2BAeKf6BDCjQzVuEXeNd1dC%2FkZjP8ezbtqzLHRXaFzulfIhq4MVmOc2kQxqfMxbtbYC2J9TNLAiMZojX07zwZFN0KBZZPgLVT5FpoM15e8s0gHI6cSYLrf9bU9R2ZXVcoWs81BSFo0lKHqA2DnUj29BoOnN9OnESWH2usRcYFViym%2B%2Fsphi%2BbAp1DzP1lp%2FgTPmiHUDia4LEsdQ7o6NVAAOvtZ1s61JgljPrzSaAcmZ%2BrS8OEIAQtMkV6uag9ubxcCab6n701Qrjj9Fn1Kaqp9nU%2FqPVCSOlHytJ1uvB48W2jLsIXaPiauR6u6PrxHKMBsA9b2KK%2Fa%2BxZ3SNHdklskRstY3Xh7FoD7ZlPKboTkjpgBTdQLxj9lD2tUD8UHHiSEaJeZxM3r4OONE2co3G3r3JlHrAGntdHRndeKHsbhj2kSSJF9owNM0Lp2OvHXCim4HvGj32Wk9ToV0PECTepWJtHiQemEByN0hc7dDRh3jFP1GuCDc65MLZXTS%2B%2BcJp7l3yZOBfFBO9GOA6twfwD%2BUcPtS17sqRrcZJ3PqIbqMpLQvSDh3cwNK0BnpLhPOMnjMcObg9Xae7Z%2FQMmic73SnaA7Jj%2BY6NwmoGitMaDxaC02GzDjjRTfjr7Ie9SwhKMg0Gt4WgFGq%2BGNCPEKyp2IBxgdI8wYs8GmKhIwnF0DGN4jrGPaH4HhTr%2BojNZkF5so%2BYEbrazleIDQ4jUhzF0dUI%2B8PIdDCx3sI6NWqN%2BjbqaTb2OkqSBcVCWIlcCi8RadRLyj9txCd%2BZ5HEhbg9A0%2Fir3tsi%2BIVm7lfBocPG7hz0E4TtB1VppKjqDzQW56gL0viY3FgkbgyosrAtoKOD63B6gM1HX%2F%2B07fnl79%2BGREpVfbJY0l5pUp79DYeKkJVRtVDqRgY1Q9HHhv1teu6G80hCCZ39gPWODDq8fJV1fuLYla7h%2BN0A9ICD5ArqhQeV2U3P1bhkW2qYxWrYyBNDBaed7bFFEsEZTRG2ebqgbt%2BqelC6c1SKqWnDrE9pGyVbRlNzriztB8YDAfSzc4Iu0qZ%2B3yJtXFyAm%2Flj7gaULsc4F%2B%2Bzu8yZBfeHtCP89GXs004AwV5kQ3xdBkE8quyzHPRYDSelQFqHoF20QCz2qXsGqi9MFH8VrthhDIyX6LV6x7R6FhJ4NiD%2F2Kjk2IcyVmccvrSecFwfn3gwOCoUmo8lSB%2BLBxlk7l5Xm98xR0C3SN8vWUT2JbZhLr7dJvBsDjdAIvZuiiBbAy8oKMFcFyw60CxjQAFeHL9KVV574e%2BXzac0m464ETXBAiMJuEHssqsEOkZ3i8Sik5V3DQ1gV0elWNOKsGYdQHVoZiH6gLhCG3yFhpNOn4D3UTw0KhTsJ7mbTwW6E3AarXFq9e8tLw1Ickfnw9cx3xvjBeKvurFjI%2BFotFznCOHYqhrJYdGTzyHWlZyLoSaFS%2BquqaWsHmLSkxajeKn1f1RYA6EsmdqAJibzoR2wJyu2dbZI91tjWWz7QbmGC5YcUwjaqGuPG0trvAxiShO0Ua8SWAU2FNUIlZjT5Vz0N9L0SyjSQdtW9DSBF%2FbFrQNgk87HGTUFgy1bMEafDktIz4t6LFfo3ReibxRwM1zwkHCzWjVpI%2BAm0HXS6hr6nV2Ud4VfT2V7T8dOfBuB1P9Oghw74DAud3frlUrdf9%2BgrWhlqW7ScgSJQvBZGZlOrDoKeAxWvbi9che%2BCUlhDVXij%2Fm9uV40hQ%2B%2Fxc%3D"></iframe>



## if use_batching
<iframe frameborder="0" style="width:100%;height:1113px;" src="https://viewer.diagrams.net/?highlight=0000ff&edit=_blank&layers=1&nav=1&title=train_batch.drawio#R7Vxbb6M4GP01SO1DK%2B6Qx5CkU612q5VmdjvzFJHgADsEZ4C0yf76tcHmZichbQmwjTSagm3AnO%2Fq7zgIymS9%2BxLZG%2B8P6IBAkEVnJyhTQZYlTZPRH9yyz1pMSc8a3Mh3yKCi4av%2FLyCNImnd%2Bg6IKwMTCIPE31QblzAMwTKptNlRBF%2Brw1YwqD51Y7uAafi6tAO29dl3Eo%2B8hWwU7Y%2FAdz36ZEkfZT1rmw4mbxJ7tgNfS03KTFAmEYRJdrTeTUCAwaO4ZNc9HOjNJxaBMGlygbN2539%2Fc9X5t%2BeV%2BO3fx1dt8nynZXd5sYMteWEy2WRPEXAjuN0IirWCYULkI5nonH0%2BmdILiBKw40nHXtCbFgAgzQFwDZJoj8aRqxSRYEaUxiDTfC0kIBlkiFdGnzbaROpufusCGHRAsDkDJ%2Bk0Tq%2Ben4CvG3uJz1%2BRNSCMvGQdYLzQoR1vMv1c%2BTvgNMLzqMTqILNg7imYLHgqDzy1LfDkAYJHe6uKKHeOpTJcLOUqllLnWKoMlo%2B7550yGEC7t2y9l%2BFj1LfoYfTGaPUDGPc4epgDBK%2Bv0WM0XCx7Fz3oDGrhQx0Mot2btsRLq%2FUAPdeKN3ZYgVH%2FtcUrJWsJAxgJyhh1Ru7iBk0N%2FUOPF0tHt%2FgQAydiLO9W9toP9tk16Eb2epN2KoqKJQGCF5D4S5vpqd4kTiWCbyGZm12tL5sl7gxhtLaDavcrARP3q9k8084AJAmI7tCrLv3Q5V6PRJnc2YHvhln3EukAiKrdfuikmoH7xdLU0s4kssN4hW5Kbx%2BCfMArjJzq08uXL%2BzlTxy8Q%2BeuhrmsmjnWsjoqjrUS8o4fbwKboO6HgV968CqAdlKeEBUuOnLx3yck8hSpTBmQcmX6kPUyFobfFLWfMLS3ZyCHHbyiVbwSz6RkjTUppTWLYrNX4LjgKzmFUeJBF4Z2MCtarVTI2PVMsXyKMb9DuCHY%2FYNUdU%2FAs7cJrCILdn7yvXT8A9%2FqXiNn0x25c3qypychet%2Fv5ZPSVfi0uCw9o9c1lmIMt9ESNHDgiR254JhnJS4dA3lUKSIQ2In%2FUq0cfbyI2WXL1WleneZwnKai9s1pml06TeoZz3OahZ%2F8UXGTrTtNvaHTlLReeU1N7FLGbwuM75NxjESUjDEFglpSj0HaHnwM3VS8jB6oZq%2F0QGJLVtsYzBd2svSwR68rSeTB9WIbd%2BIoZbNWPzBYTynzVmxma57S6NRT5pbzo9TTV0%2BpNnWUer8MhK0R7UF8KHsoSSlOIvgTTLKshzqcFfI1tSaSmE1JVqZY2CBQKhmMScfad5xUcXgWV1WmNoxOqRndiDU6ns3JbdmcfM33r%2Fn%2BkPN9tWpQitZ1vi833vXQ%2BOXP5qhqoV3ROaFdNO45sOSl74%2FHpT%2Fksvwh2xxMjp%2BWWkuOKFrDgq85VXVhNPuz5eZ8NBuQVZdFU2JdXmTv4zmKVDR%2BLCIaOm4iKMhW5OD%2F3MXtrtNdEW8AP3eb3eFNeYHrQun0Qol67pMrJaVfhXg6717b1PmUVpF1EO%2Bli527L%2BVKFF%2FXQLw1kD9H8%2FbD9IIphnwA66E6aazzkq2LLogUNrXqabSqi%2BOt0evt0YpmUqejlfjR0Ypc%2Bif00aRzddJFvjbRO2QTJRfVFCWfxTs20R7fcUCKfoNObSpMCkQxSKgyKXgWjh176ftJ78uFmlaNVblfuRBbpLxBvhinObe709kQO2IguZHZ%2FUJZabAJuwfYyWoT7HiV9fZ%2BS9FgXzYIHWr5iwAuf9a8UuiU2NT8F17cSPUGur%2BJ43mHr2lKPJXko3HkQ9veGceUWtHE1Gtyz16ICWTsjWoFZ7OuQC1HRF0fSjY1jK0Gitk04xr1Kyayv1yw8DYDdPg0R6stJ1vYWLsbNMqSbjn1gP8vyzqS69GAQwvlK56LEK1UfUrySreFzHFVJ19hyjsLC6qer2TKOS%2FGWZ9bnCNOzZsrzvZ%2B3Sez0X2mC%2BMHwXwQZqpgmcJ4DDYQGeRME0aiYM5ib7taIRSzXpSxkQPriZYZagItgJW6KSXI9aRKEjlMIvdnHa0lpBpbrAvh57EFrW8bSLRPt2mr3XxEbZqPaP3iK1Q2HyGvwESz3rITGrMCV7TOyT76jZP3sROEmiC8BJ%2BUKDiIg3zDMbLhMNNwimY4xjEcJhiOswtHqIUmvEJOKuSMwkE6gcslpCRCjU4ombTor20XxIJmhZt7nK3D9f3Sg%2F4S3BDS4VbQpgfJhsKirsK%2FmPDr4jyUdFw%2BUdNrxQ2FU%2FvK94BdJD3W2Iptyl6jTG2SHjiDDgs9IK21BtsMB4OdfNGMdRglbc0Qe4ZbJ9%2FKePcKyeDs3bwobtTx9hw3naFQDF6V5aLIdbJL83zkdAY57obrSyLH%2B1lHfZneKvl0dCneLvVE1aYn1JOhKffGAcs6l30y6s7twvsxdE7WMRsJo3Fa9tSF0USwxrgF5T3mCJc%2BUftIJ4yELOalUSxaZJEBmwbSPFHMkvL8UrwljFMap9nkgeGfpS5o1JYAOqcuyK2Qa625oE6%2BS3a%2B8659zy3ncztz3WyaxSOKiIqfMo8Kc3Q1EOHo%2FpDWDGQ33f71FHm%2F1o%2BKJv4mBbH3hf%2FV10x8%2BPUFXmmnXpXR8UbZel0iBNFqHu%2FDxMP1nKIQg5yxNRHG6cHIEqyZMDOwkzYN6pXRkyZSaWRGS%2BXDi2pUNsHjZZCOVeiAfnC06KDKqLUVGOdTj4qscH9Fd77OoNPia81ZuC6%2Bea3M%2FgM%3D"></iframe>

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

## get_embedder(args.multires, args.i_embed)

输入：
- args.multires, 输入的L
- args.i_embed，默认为0，使用位置编码，-1为无位置编码

输出：
- embed, 位置编码函数，`将(1024 * 32 * 64) * 3处理为 (1024 * 32 * 64) * 63`
    - input_ch = 3 , L = 10 并且包括输入维度
- embedder_obj.out_dim : 输出的维度

### Embedder()

```
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
    def create_embedding_fn(self):  
    
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

## network_query_fn

```
network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
    embed_fn=embed_fn,
    embeddirs_fn=embeddirs_fn,
    netchunk=args.netchunk)
```

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


需要对render输入的光线做batch: `rays = batch_rays`


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

<iframe frameborder="0" style="width:100%;height:1583px;" src="https://viewer.diagrams.net/?highlight=0000ff&edit=_blank&layers=1&nav=1&title=render%E6%B5%81%E7%A8%8B.drawio#R7R1bc6O2%2Btcwm3QmHu7gR1%2BS3Zme7mm7Z7qnTx5sY5uzGFzAcbK%2F%2FkhCAiR9tnECNkncmW6MJJD47jcJxRitnz4n3mb1Wzz3Q0VX50%2BKMVZ0XTMNFf3BLc95i62becMyCeZ0UNnwLfjp00Z633IbzP2UG5jFcZgFG75xFkeRP8u4Ni9J4h0%2FbBGH%2FKwbb%2BlLDd9mXii3fg%2Fm2SpvdXWnbP%2FiB8sVm1mz%2B3nP2mOD6ZukK28e7ypNxr1ijJI4zvJf66eRH2LgMbjk9z3s6S0WlvhRVucG77P%2FFD5Gz9%2F%2BUhePm1%2FT2V9L444uNs2e2Qv7c%2FT%2B9DJOslW8jCMvvC9bh0m8jeY%2BfqqKrsox%2F4rjDWrUUOP%2F%2FCx7psj0tlmMmlbZOqS9%2FlOQ%2FRff3rPo1d%2BVnvETfTK5eGYXUZY8V27Cl39X%2B8rbyBW7bxFHGV2IZqBrGWoUkGm8TWb%2BAVAx6vOSpZ8dGKfn4zAcKxNQnHz247WP1ocGJH7oZcEjT2ceJddlMa7EKPpBkXoCgulzH71wS2dae0GEWkKMLXQz%2Fp0lpE2ghBLPGDW7VZD53zYegdEOMTuP09pwfvSTzH86CBnaa1mUc54Zb9HrXcmIRduqwoSm2hIwzUtyi1bhlZJzjnELxysl67TOLXpNbjE6xS26xC3ozed%2B0n3ecC%2FNG%2FZVk9TlDaMmb1id4g1D4g20%2BkniPaed4w5HFbnDkrjDAZjDaIs5GCdeiDtepDkuxR1WTe5wO8UdlsQd0Xz2Nrij0AkX4w7jqjvqcodbkzs07VLssXncjf%2F8t7P68vB898fU%2FPEzjfw747KO5tkRPPfSVcHijWIbBK9pNo1tcusgQQKsMmATB1GWVp78O24oJYvLCxZDjEgcHq6Z%2FcPjbdU6eAP6ka%2B4pNPi1V8unVxJsk%2B9bLYKFs%2FdFO%2Bia6Bf3DXQLuo3vy35rtUNM2kXizOBEkh337eA5wR6u%2BiG4eu0IuElGWsYYtBNE8RCvlR6W0k1p%2BoKXZjIOCL8xfHFwmprC%2BGGdrQFQ6gUQnobysKwLq4snKuyqC096kZZtW6FkswLO3x6%2B0huS1%2FUxXhLHsFRKevq7agLy3G4eUz3sPQ3hCDHqeMdxzpJuwjjW1IuuqRcIj%2FbxcmPyT9bP3meLCJFt0NEF8NpwvGX%2Fc8W55kJJd6lhBQHaIBmbJ4IPbJ%2B9GtJ%2FuKb1GQbTegEnddcpn5xzXVZE%2Fhtaa66UV7NbkWOnSp%2BHINnd8s8zfgUxrckHuQgtL%2Be%2BnMiFjrGvnZfAJAqs68LcG9rQWj9GoSuz712Te7VLxakOLhuIJaHWpV7S3FdxbWVe1tx%2B8qwr9ybyhC1qLhlMFIG2iK6ma220Y9f0HjbuMW3DExl4OAfQ1VxHzrHaEzyFIxmA3rSADjNao3TNAlIXS%2Bj0Tg%2BO1pG05Lxzyyct%2BbusXULpQNEN3WxuMZ2nR6vnUxNLiBgCqxp7QTnz67lNfXZ5HI%2B8uvYRPbusOGmkp8qc%2FQqTWtabZ1fffX%2FfLjp9Xq3neMny%2BS5yTZkbtL0s6qgy7LTG1ZBdf22jll%2BuuwaYX7pHKuIbpFtyqxyXrfo3Qc12uIUpy6ndMxYcyROmSW%2Bl%2FmTyE8WnWcYC6h2OCvDGLKxe%2FMdXWOP8cst%2FYGYSo1xhFbuwgnERkK4AUKWuk39yWPg7%2BZBIicfEYgzHkFplsQ%2F%2FFEcosUZ4yiOMDcvgjAUmrwwWEbocoYQhwx4Y4gRFsy8cEA71sF8TkQBRBG8eGiBKAyhskiHspo2QBVi8qI5qpDTwnkgQaYIrlmTfeUPgzUwFw0VpraHNbnmnQSJLGVo09jQYIwxRy%2BGmCFxnn%2BC9NnYwB36iDXN86YRseW9hOCODViw6w%2BDa0fg0D6A6%2F5ZcW1KuA6izRYnBdRc0vKc%2BXWSeutN6KN%2BG%2F1AYImmKf6DOw2xkWD6ZLEuyfRSlNM1kSXRFckB7PdLPjWSfzqU%2FGuPfGQHAyQfkXpuS%2BS9njyIypdo5OjECCbDj0U%2Fpi3QDyB%2BikTxeehHTk3Q%2BOi8JiLtRkjoRqShbOUjfA1um3g4%2F0aHX4dh5COQo1h8YgGWjw7laNojR7nqfQ85vor8IL%2BlSQqEng8R4YckOiExaOtAVPa8JlhfIro7dHmE8MzOiz35JaQ3eBNEh6%2FZdAoWWv2%2BqjZEjEJ1tw1tWmmLGOEtVRJSup6kLi7OfdYDy6Md3cRwscMe4NVcK82bR3Hz%2B1ReV2kue%2FWJt9PjbZb7Zh2LZotFra4D6OTzHnkj%2ByVmruuQQhwt0f9THMIaPSjuB%2FIfLcFgdw1ZXZ3VdDJlex3RuQKFiQ4Grhoxpn5O0DLSwzO%2FehIaU70GwgSBAXiO6jkFhiVnOrD9myynkzWCCQl3K7aqWEOKshHBmDXGjbguw9BUajPPg3RDbsIYluOplYdAd3uzWXHz0cE7AppUHkyWVxIuuEx%2Fk60OTgVT44nqpU3rW4jG96G9FJA8c9uyvuXEy9dJsN4ge8uLELwkvb2K19NtehGd7aj8ziTXAnQ2yySdBXjX81RAwxaEFVDPBMNUv5Rde2jZFf7IJdRkM%2B9ejYajuUI9bf%2BcZ1KAAJTN2r%2F9Q%2F6AykMJYAWQZURo%2BtF8gE%2BlRdfTMJ79yJseArx4SufsEFy3sFLYwbPEbmGnyOLBoTf1w6E3%2B7EkCxVMl%2BM8f06WfTmLAsflHeKJoxxKqVLtaRorr6BkeccOJ37pplg2JF4sUp%2Fb7yrvKBN1rkjqOVykjbPHH%2BSqPVXVLNNy%2B4aj9V3%2BsXv2475gDxqIBNkPyb2BCbKTQWO9mbQVs9%2BsYa%2FXQ0YbouHBnYbtLzonbz4KqQu9iSUcNFDerQPiuJJsB6q7dR2wf2yrJdkOpxF4T%2FTN4apF87%2FYs89OSlGBzZit1cvB5pZ1NWHr6kfoTAh44MX2uxxc9%2FU0A8hE1lTgWM7z2sja2wqeOxrviCNgXRyCspuxXw3RMqxKlPSjRTNdTVREQFz9zIpI3g5x2JjYg8Vrjh%2BZGbKMg8LVJmArtoffN%2FMFlAbPn6%2B7j5wN7IjBYF6NwtpGIbT97ExHB7%2FuHFFZ3mZxMlv1Zl7GDtkgB3QMHvBpG3QTxiCXx%2FjcjqEyHJEf98rAUu5dZYhucnBLf4xbuma1SF%2FyADcrndVqAbawJTiapeZ2tOIMP9F03qcipMPl9z6xzF3ZX%2BTy8gE0OVf2s2ydM27K5qePmQePr3oOKQxEr4%2FT6SSQFOa9D%2BTBzc4F7xoRmzAuLIQCb%2FeJBbiIe4Pz%2FeLgQ6tNN14ELhdbJnfUyCDhOmZnKPgDas2AX5wj9BfZy2bY%2Bxr1H2Qw8B0ulyhMO87e48omEKTz9fBrhBGwZ3ADVNQMYAjpS9YtJTE6laryXFH%2F7c%2FLLMupynNLLq8maocWiWUkv0omNTu1TCSq%2BVVS2d2pRf6cpNmcW2VuRqDWm58lS48IkNdoBP5SHr7aRtPAS4lB8OCFqX8LTkt2TVnDr%2FmpztZ435vDPt%2F5rQxN9KyL81YqVoYJ7Xgq3O3mzQy5yIPU7YThpNvmxl6J%2B6p6NvpG%2BQ6H7798aXRnYCMrlO2O6lpfpHaOMnljKrgJ6J1qMenqfpOmEYO1FTyfaLQ0IuJaDDtp4mdmoL3mJrS7rt%2Ba5JMD63JQ8Ro%2FFDftCp6yA50ZYAF4bC14CGz6PqXWtQk392V1sk3NfLTGtqmJjtXnihF4uWK3MWC%2F%2BWrf4uPlhS0IbTeGDk1qrWLVBL6astcfr%2BJ6CMjJRlC9fw%2FDvukbmfa6qyGnM9FbATK5hq71gCNYW4uKQlvJ%2FGwSBimp%2FIsw1nDDPJhl3Y8yW1Bm9axRZmBn2VE7e5aTMfUAb%2BjpSezPbcXaXnjrIHzOh6788NHH5L3fGq905JPinihO1l5Y6dtRuOBOk4pjNfQzxEN3aM2zIFrKd%2B6PdJKeANFFRJ%2BpspWQnizxonSBnsSeSVgXoyxO5vx8xY3Tohj4ToCVjreIYTDpZp%2F%2BsBjEsAERehRaQRQGbKZFGHuZML0osgp%2FvU3%2FoAnzUfQDoE8OQLUFrR3qbF0492zatlLJaPZUU1fa3mluHbBt8GS%2F%2B0mAoIuV0ivLxs26GVDrYl%2BSPrhuQC42lRnA4zF3ezP%2BOaWgrNz6SnsZyZ7CUmaiwqpE9Dj7HAEsX9uZEghnhwaWtBA4ygjmx4IHciAhcBTx2o8FDaQaE49zcknwGwdccz2e%2B0R5TLvX69FmvOXBGfNZiDqA6rqjaji2WDAGmq3QztTWshbATnuBVF4Ydv2S4%2FV7%2FufX%2FM9M3%2B1F6qvRZzWDJvbpuwJJ0BdmINeiNcPKlHeANIMjotDy813vH3ChUX9wQxFG0WdwJ0YJd82P39UOlltkUnH7mQZF16ETCJr4uANsz8nlS21w6CKeIRevPM%2B3POk35k79fWvsqwMiVofC6u35RXI0sHH2PcSkUp8YVb%2F5Oh7hYsT%2BWOnn3wZTFddR7h1laCjDIfts2PD9cDVYlNjWJ1tgopDDb0fDRVIkqEAkApdhClYc36PsS96amLK4PilixHWLQSP9eNyIDTmSVgbjR1wnFEJiA%2BAoEus9EkhSK7EktRJOYrdDEaUCKkJQ6ZjpPStIvBxo4P8WC%2FneMiCfIxzzpK70EXP2MYu6DmXXIWHX%2FkBxTVIbTCqL6ZiHI3Y0JxhesdJKJqG62OE9tNhW10hFh4wJXHHtKgMb3XhHV0hhxg68x%2Bfho2WPKyssxrj4XfoGGzM4ec3NBxObUpoarzSh8yygTxU4rYlH4PzxeuWbUsb4o2SZbGHbswvpuLPWE1j7fcsTUiJlVqRMjHAaqWmNaFxOI5JkynvXh%2F9ZBTi9Tf7xwjRWyPHE6N9iJ64a57sU0LsnN7ftSc%2FGjk80xMCODXkdDQUN0GUSY4gWfZ%2FRW65%2BI59PNO7%2FDw%3D%3D"></iframe>


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