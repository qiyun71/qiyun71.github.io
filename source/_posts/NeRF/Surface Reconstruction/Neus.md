---
title: Neus
date: 2023-06-14 22:14:49
tags:
    - Neus
    - Surface Reconstruction
categories: NeRF/Surface Reconstruction
---

NeuS: Learning Neural Implicit Surfaces by Volume Rendering for Multi-view Reconstruction
实现了三维重建：从图片中获得点云

<!-- more -->

[Code: Totoro97/NeuS: Code release for NeuS (github.com)](https://github.com/Totoro97/NeuS)
[Project Page](https://lingjie0206.github.io/papers/NeuS/)

Neus的总目标是实现从2D图像输入中以高保真度重建对象和场景。
现有的神经表面重建方法，如DVR[Niemeyer等人，2020]和IDR[Yariv等人，2020]，需要前景掩码作为监督，很容易陷入局部最小值，因此在对具有严重自遮挡或薄结构的对象进行重建时面临困难。
最近的新视角合成神经方法，如NeRF[Mildenhall等人，2020]及其变体，使用体积渲染来产生具有优化鲁棒性的神经场景表示，即使对于非常复杂的对象也是如此。**然而，从这种学习的隐式表示中提取高质量的表面是困难的，因为在表示中没有足够的表面约束。**
在NeuS中，我们提出将表面表示为有符号距离函数（SDF）的零水平集，并开发了一种新的体积渲染方法来训练神经SDF表示。我们观察到，传统的体积渲染方法会导致固有的几何误差（即偏差）对于表面重建，因此提出了一个新的公式，它在一阶近似中没有偏差，从而即使在没有掩码监督的情况下也能实现更准确的表面重建。
在DTU数据集和BlendedMVS数据集上的实验证明，NeuS在高质量表面重建方面优于现有技术，尤其是对于具有复杂结构和自遮挡的对象和场景。

# 优点&不足之处

## 优点
- 通过手动在meshlab中clean稀疏点云ply中其他噪音位置的点云，构建了一个精确的bounds，可以将模型不包括背景且几乎没有噪声的生成出来
- 通过构建SDF场，其零水平集相比NeRF的密度场水平集(Threshold = 25)，生成的mesh更加精细，或者说更加平滑
    - 对图片中深度突然变化的部分，sdf也可以很好的重建出来
## 不足
- 对于无纹理物体(例如反光和阴影区域)的重建效果并不理想
- 需要手动在meshlab中clean稀疏点云ply中其他噪音位置的点云，这也是本文所说不需mask监督的方法
- (in paper:)一个有趣的未来研究方向是根据不同的局部几何特征，对不同空间位置具有不同方差的概率以及场景表示的优化进行建模

# 引言+相关工作

## SDF简单理解
SDF：输入一个空间中的点，输出为该点到某个表面（可以是曲面）最近的距离，符号在表面外部为正，内部为负。
给定一个物体的平面，我们定义SDF为空间某点到该平面距离为0的位置的点的集合（也就是物体表面的点）。如果空间中某点在物体表面之外，那么SDF>0；反之如果空间中某点在物体表面之内，那么SDF<0。这样子就可以找到物体表面在三维空间中的位置，自然而然的生成三维表面。

![image.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/20230629135016.png)

  1. mesh是一种由图表示的数据结构，基于顶点、边、面共同组成的多面体。它可以十分灵活的表示复杂物体的表面，在计算机图形学中有着广泛的应用。从nerf输出的物理意义就可以想到，density可以用来表示空间中沿光线照射方向的密度，**那么我们可以通过基于密度的阈值来控制得到的mesh**。这种方法的好处是，训练好一个nerf的模型就可以得到一个mesh了。但是这种方式也有很大的缺点：一是训练结果会有很多噪音而且生成的mesh会有很多的空洞，二是很难控制一个合理的阈值。
  2. 这里我们可以考虑使用有向距离场（Signed distance function 简称 SDF）来取代nerf建模。使用SDF的一大好处是，SDF函数本身在空间是连续的，这样子就不需要考虑离散化的问题。我们之后使用Marching cubes方法来生成mesh。
  3. NeRF生成一个带有密度和颜色信息的模型，通过使用SDF来代替密度，在密度大的地方表示为物体的表面，就可以生成一个mesh模型。

>[旷视3d CV master系列训练营二： 基于NeRF和SDF的三维重建与实践 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/553474332)

## 相较于以往的工作：

![Pasted image 20230531185214.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/Pasted%20image%2020230531185214.png)


IDR无法应对图片中深度突然变化的部分，这是因为它对每条光线只进行一次表面碰撞（上图（a）上），于是梯度也到此为止，这对于反向传播来说过于局部了，于是困于局部最小值，如下图IDR无法处理好突然加深的坑。

NeRF的体积渲染方法提出沿着每条光线进行多次采样（上图（a）下）然后进行α合并，可以应对突然的深度变化但NeRF是专注于生成新视点图像而不是表面重建所以有明显噪声。

>[NeuS: Learning Neural Implicit Surfaces by Volume Rendering for Multi-view Reconstruction - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/496752239)

新的体积渲染方案：Neus，使用SDF进行表面表示，并使用一种新的体积渲染方案来学习神经SDF表示。

## 本文方法

本文集中在通过经典的绘制技术从2D图像中学习编码3D空间中的几何和外观的隐式神经表示，限制在此范围内，相关工作可以大致分为基于表面渲染的方法和基于体积渲染的方法
- 基于表面渲染：假设光线的颜色仅依赖于光线与场景几何的交点的颜色，这使得梯度仅反向传播到交点附近的局部区域，很难重建具有严重自遮挡和突然深度变化的复杂物体，因此需要物体mask来进行监督
- 基于体积渲染：eg_NeRF，通过沿每条射线的采样点的α-合成颜色来渲染图像。正如在介绍中所解释的，它可以处理突然的深度变化并合成高质量的图像。提取学习到的隐式场的高保真表面是困难的，因为基于密度的场景表示对其等值集缺乏足够的约束。
相比之下，我们的方法结合了基于表面渲染和基于体积渲染的方法的优点，通过将场景空间约束为带符号距离函数SDF，但使用体积渲染来训练具有鲁棒性的representation。
- 同时UNISURF也通过体积渲染学习隐式曲面。在优化过程中，通过缩小体积渲染的样本区域来提高重建质量。
**UNISURF用占用值来表示表面，而我们的方法用SDF来表示场景**，(因此可以自然地提取表面作为场景的零水平集，产生比UNISURF更好的重建精度。)

# 方法

{% note info %} 构建了一个无偏，且collusion-aware的权重函数w(t) = T(t)ρ(t) {% endnote %}


给定一组三维物体的pose图像，目标是重建该物体的表面，该表面由神经隐式SDF的零水平集表示。
为了学习神经网络的权重，开发了一种**新的体积渲染方法**来渲染隐式SDF的图像，并最小化渲染图像与输入图像之间的差异。确保了Neus在重建复杂结构物体时的鲁棒性优化。

## 渲染步骤
### 场景表示

被重构物体的场景表示为两个函数：
- f：将空间点的空间坐标映射为该点到物体的符号距离
- c：编码与点x和观察方向v相关联的颜色
这两个函数都被MLP编码，物体的表面有SDF的零水平集表示 $\mathcal{S}=\left\{\mathbf{x}\in\mathbb{R}^3|f(\mathbf{x})=0\right\}.$

定义概率密度函数 $\begin{aligned} \phi_s(x)& =se^{-sx}/(1+e^{-sx})^2  \end{aligned}$

其为sigmoid函数的导数 $\Phi_s(x)=(1+e^{-sx})^{-1},\text{i.e.,}\phi_s(x)=\Phi_s'(x)$


### Neus与NeRF 体渲染函数对比

| Project          | Neus                                                                                                                   | NeRF                                                                                                                                                                                      |
| ---------------- | ---------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 渲染函数         | $$C(\mathbf{o},\mathbf{v})=\int_{0}^{+\infty}w(t)c(\mathbf{p}(t),\mathbf{v})\mathrm{d}t,$$                               | $$\mathrm{C}(r)=\int_{\mathrm{t}_{\mathrm{n}}}^{\mathrm{t}_{\mathrm{f}}} \mathrm{T}(\mathrm{t}) \sigma(\mathrm{r}(\mathrm{t})) \mathrm{c}(\mathrm{r}(\mathrm{t}), \mathrm{d}) \mathrm{dt}$$ |
| 权重             | $$w(t)=T(t)\rho(t),\text{where}T(t)=\exp\left(-\int_0^t\rho(u)\mathrm{d}u\right).$$ **无偏、且遮挡**                     | $$w(t)=T(t)\sigma(t) , \text { where } \mathrm{T}(\mathrm{t})=\exp \left(-\int_{\mathrm{t}_{\mathrm{n}}}^{\mathrm{t}} \sigma(\mathrm{r}(\mathrm{s})) \mathrm{ds}\right)$$   **遮挡但有偏**  |
| 不透明度密度函数 | $$\rho(t)=\max\left(\frac{-\frac{\mathrm{d}\Phi_s}{\mathrm{d}t}(f(\mathbf{p}(t)))}{\Phi_s(f(\mathbf{p}(t)))},0\right).$$ | $$\sigma(t)=\phi_s(f(\mathbf{p}(t)))$$                                                                                                                                                      |
| 离散化                 |       $$\hat{C}=\sum_{i=1}^n T_i\alpha_i c_i,$$  $$T_i=\prod_{j=1}^{i-1}(1-\alpha_j)$$ $$\alpha_i=\max\left(\frac{\Phi_s(f(\mathbf{p}(t_i))))-\Phi_s(f(\mathbf{p}(t_{i+1})))}{\Phi_s(f(\mathbf{p}(t_i)))},0\right).$$                                                                                                                 |                                                               $$\hat{C}(\mathbf{r})=\sum_{i=1}^{N} T_{i}\left(1-\exp \left(-\sigma_{i} \delta_{i}\right)\right) \mathbf{c}_{i} \text {, where } T_{i}=\exp \left(-\sum_{j=1}^{i-1} \sigma_{j} \delta_{j}\right)$$                                                                                                                            |

权重函数，Neus(右)，NeRF(左)
![Pasted image 20230606154119.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/Pasted%20image%2020230606154119.png)


### 重要推导

why `true_cos = (dirs * gradients).sum(-1, keepdim=True)`



## 训练损失函数

loss函数
$$\mathcal L=\mathcal L_{color}+\lambda\mathcal L_{reg}+\beta\mathcal L_{mask}.$$
$$\mathcal{L}_{color}=\frac{1}{m}\sum_k\mathcal{R}(\hat{C}_k,C_k).$$
Eikonal term,类似[IDR](https://lioryariv.github.io/idr/)

---
$$\mathcal{L}_{r e g}=\frac{1}{n m}\sum_{k,i}(\|\nabla f(\hat{\mathbf{p}}_{k,i})\|_{2}-1)^{2}.$$
[IGR](https://github.com/amosgropp/IGR)

---

$$\mathcal{L}_{mask}=\mathrm{BCE}(M_k,\hat{O}_k)$$
沿着相机ray的权重之和：
$$\hat{O}_k=\sum_{i=1}^n T_{k,i}\alpha_{k,i}$$
是否使用mask监督: (BCE是二值交叉熵损失)
$$M_{k} ∈ {0, 1}$$

分层采样类似NeRF




# 实验

## Dataset：DTU&BlendedMVS

DTU: 15个场景、1600 × 1200像素
BlendedMVS：7个场景、768 × 576像素

## Baseline

表面重建方法：
- IDR：高质量表面重建，但需要前景mask监督
- DVR，没有比较
体积渲染方法：
- NeRF：我们使用 25 的threshold 从学习的密度场中提取网格（在补充材料中验证了为什么用25做阈值）
广泛使用的经典MVS方法：（MVS：Multi-View Stereo Reconstruction多视点立体重建）
- colmap：从colmap的输出点云中，使用Screened Poisson Surface Reconstruction重建mesh

UNISURF：将表面渲染与以占用场做场景表示的体积渲染统一起来

## 实现细节

设感兴趣的物体在单位球体内，每批采样512条光线，单个RTX2080Ti：14h（with mask），16h（without mask）
对于w/o mask ，使用NeRF++对背景进行建模，网络架构和初始化方案与IDR相似

## 环境配置：
autodl镜像：
    Miniconda  conda3
    Python  3.8(ubuntu20.04)
    Cuda  11.8
`conda create -n neus python=3.8`
`pip install -r requirements.txt`
`pip --default-timeout=1000 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html`

## 命令行运行程序：
`python exp_runner.py --mode train --conf ./confs/wmask.conf --case <case_name>` # 带mask监督
`python exp_runner.py --mode train --conf ./confs/womask.conf --case <case_name>` # 不带mask监督
case_name是所选物体数据集文件夹的名称
如果训练过程中中断，可以设置`--is_continue` 来进行加载最后的ckpt继续训练：
`python exp_runner.py --mode train --conf ./confs/wmask.conf --case <case_name> --is_continue`

训练生成的结果会保存在根目录下exp文件夹下，可以看到meshes文件夹中保存了训练过程中的mesh模型，但面片较少。需要比较精细的mesh模型需要运行
`python exp_runner.py --mode validate_mesh --conf <config_file> --case <case_name> --is_continue # use latest checkpoint`
多视角渲染
`python exp_runner.py --mode interpolate_<img_idx_0>_<img_idx_1> --conf <config_file> --case <case_name> --is_continue # use latest checkpoint`


### eg1: clock_wmask
```
python exp_runner.py --mode train --conf ./confs/wmask.conf --case bmvs_clock #带mask
中断后继续训练
python exp_runner.py --mode train --conf ./confs/wmask.conf --case bmvs_clock --is_continue
将mesh模型精细化
python exp_runner.py --mode validate_mesh --conf ./confs/wmask.conf --case bmvs_clock --is_continue
插值多视角渲染成mp4(0~1之间新视角生成)
python exp_runner.py --mode interpolate_000_001 --conf ./confs/wmask.conf --case bmvs_clock --is_continue
```
![Pasted image 20230601130943.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/Pasted%20image%2020230601130943.png)


除了主体模型外还有一些噪音
![Pasted image 20230601131053.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/Pasted%20image%2020230601131053.png)


Neus： (结果与resolution无关)
- 使用Neus时，精细化模型参数设置为 `resolution=512` 可能与此有关
- 改为`resolution=1024` 运行一下validate_mesh
![Pasted image 20230602165940.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/Pasted%20image%2020230602165940.png)



虽然运行后，面数更多了，但是细节处依然不清楚，可能需要继续增加训练的步数
- 看代码后：resolution可以增加object的分辨率，再缩放到物体实际大小
    - 可以看到vertices和faces都增加了，但由于sdf生成的相同，因此表面变得更精细，但细节处不清楚的地方依然不清楚

### eg2: bear_womask
```
python exp_runner.py --mode train --conf ./confs/womask.conf --case bmvs_bear  #没有mask
python exp_runner.py --mode train --conf ./confs/womask.conf --case bmvs_bear  --is_continue
```

![Pasted image 20230601211543.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/Pasted%20image%2020230601211543.png)

![Pasted image 20230601211823.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/Pasted%20image%2020230601211823.png)


运行后，mesh模型的面很少
```
将mesh模型精细化
python exp_runner.py --mode validate_mesh --conf ./confs/womask.conf --case bmvs_bear --is_continue
```
![Pasted image 20230601211901.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/Pasted%20image%2020230601211901.png)

![Pasted image 20230601211955.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/Pasted%20image%2020230601211955.png)



### eg3: Miku

模型不够理想
```
python exp_runner.py --mode train --conf ./confs/womask.conf --case Miku
python exp_runner.py --mode validate_mesh --conf ./confs/womask.conf --case Miku --is_continue

python exp_runner.py --mode interpolate_0_38 --conf ./confs/womask.conf --case Miku --is_continue
```

效果：

![image.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/20230630141311.png)



- 底下的圆盘在sparse_points.ply中没有剔除干净，所以会出现额外的突出
- 面部表情、裙子装饰等细小部位不够细致，腿部也不够细致
- 头发部位and鞋子也由于在点云中没有完全去除噪点，因此会有额外的噪声被建模出来

0 to 38 render video：

![00300000_0_38.gif](https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/00300000_0_38.gif)



### Neus如何给模型加纹理：
>[How to reconstruct texture after generating mesh ? · Issue #48 · Totoro97/NeuS (github.com)](https://github.com/Totoro97/NeuS/issues/48)


## 渲染自定义视角的视频

根据中间两个img进行插值，中间插入生成的新视图图片

```
python exp_runner.py --mode interpolate_<img_idx_0>_<img_idx_1> --conf <config_file> --case <case_name> --is_continue # use latest checkpoint

eg:
python exp_runner.py --mode train --conf ./confs/womask.conf --case bmvs_bear
python exp_runner.py --mode interpolate_1_2 --conf ./confs/womask.conf --case bmvs_bear --is_continue # use latest checkpoint
```

eg: 1 to 2
![00300000_1_2.gif](https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/00300000_1_2.gif)

<div style="display: flex; justify-content: center;"> <img src="https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/001.png" alt="Image 1" style="width: 50%; height: auto; margin: 10px;"> to <img src="https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/002.png" alt="Image 2" style="width: 50%; height: auto; margin: 10px;"> </div>


# Neus使用自制数据集

## 自定义数据集colmap操作
自己拍一组照片: **手机或者相机 绕 物体拍一周，每张的角度不要超过30°（保证有overlap区域）**
>[colmap简介及入门级使用 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/397760339)

![Pasted image 20230531192515.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/Pasted%20image%2020230531192515.png)


打开colmap.bat进入gui界面，点击**Reconstruction**，再点击**Automatic reconstruction**

**workspace folder：选择workspace文件夹，注意不支持中文路径**
**Image folder：选择存放多视角图像的数据文件夹，注意不支持中文路径**
**Data type：选择 Individual images**
**Quality：看需要选择，选择High重建花费的时间最长，重建的质量不一定最好；**

在COLMAP中看生成的稀疏点云
![Pasted image 20230531191938.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/Pasted%20image%2020230531191938.png)


在 workspace folder 文件夹->dense->0文件夹下找到 fused.ply数据，用 meshlab中打开可以看到稠密的三维重建的结果。
在Meshlab中查看生成的稠密点云

![Pasted image 20230531192204.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/Pasted%20image%2020230531192204.png)



## Neus命令操作

>[(18条消息) 基于Nerf的三维重建算法Neus初探_Alpha狗蛋的博客-CSDN博客](https://blog.csdn.net/mockbird123/article/details/129934066)

两个文件夹：image和mask，一个文件
image文件夹就是rgb图片数据，算法默认支持png格式。
mask文件夹包含的是模型的前景图像，前景和后景以黑色和白色区分，如果配置文件选择withou mask，其实这个文件夹的数据是没有意义的。但必须有文件，且名称、图像像素要和image的图像一一对应。
最后是cameras_sphere.npz文件，它包括了相机的属性和图像的位姿信息等，这个是需要我们自己计算的。官方给出了两种计算方案，第二种是用colmap计算npz文件。

### 使用Colmap生成npz文件

可以提前通过colmap运行得到sparse/0/中的文件，或者通过img2poses中的run_colmap()生成，然后再得到sparse_points.ply

```
cd colmap_preprocess
python img2poses.py ${data_dir}
```

将会生成：`${data_dir}/sparse_points.ply`，在meshlab中选择多余部分的Vertices，并删除，然后保存为`${data_dir}/sparse_points_interest.ply`.

然后
```
python gen_cameras.py ${data_dir}
```

就会在 ${data_dir}下生成preprocessed，包括image、mask和cameras_sphere.npz
