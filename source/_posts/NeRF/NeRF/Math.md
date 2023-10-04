---
title: Math
date: 2023-07-31 19:59:15
tags:
    - Math
    - NeRF
categories: NeRF/NeRF
---

NeRF相关的数学知识

<!-- more -->

# linearColor 2 sRGB

固定色调映射函数，将线性颜色转换为sRGB[1]，并将输出颜色剪辑为lie[0,1]。
[Proposal for a Standard Default Color Space for the Internet - sRGB.-论文阅读讨论-ReadPaper - 轻松读论文 | 专业翻译 | 一键引文 | 图表同屏](https://readpaper.com/paper/35410341)

(Why)为什么要将线性RGB转换成sRGB
> [小tip: 了解LinearRGB和sRGB以及使用JS相互转换 « 张鑫旭-鑫空间-鑫生活 (zhangxinxu.com)](https://www.zhangxinxu.com/wordpress/2017/12/linear-rgb-srgb-js-convert/)

假设白板的光线反射率是100%，黑板的光线反射率是0%。则在线性RGB的世界中，50%灰色就是光线反射率为50%的灰色。

**人这种动物，对于真实世界的颜色感受，并不是线性的，而是曲线的**

![image.png](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230810162118.png)

(How)线性RGB与sRGB相互转化

```python
def linear_to_srgb(linear):
    if isinstance(linear, torch.Tensor):
        """Assumes `linear` is in [0, 1], see https://en.wikipedia.org/wiki/SRGB."""
        eps = torch.finfo(torch.float32).eps
        srgb0 = 323 / 25 * linear
        srgb1 = (211 * torch.clamp(linear, min=eps)**(5 / 12) - 11) / 200
        return torch.where(linear <= 0.0031308, srgb0, srgb1)
    elif isinstance(linear, np.ndarray):
        eps = np.finfo(np.float32).eps
        srgb0 = 323 / 25 * linear
        srgb1 = (211 * np.maximum(eps, linear) ** (5 / 12) - 11) / 200
        return np.where(linear <= 0.0031308, srgb0, srgb1)
    else:
        raise NotImplementedError

def srgb_to_linear(srgb):
    if isinstance(srgb, torch.Tensor):
        """Assumes `srgb` is in [0, 1], see https://en.wikipedia.org/wiki/SRGB."""
        eps = torch.finfo(torch.float32).eps
        linear0 = 25 / 323 * srgb
        linear1 = torch.clamp(((200 * srgb + 11) / (211)), min=eps)**(12 / 5)
        return torch.where(srgb <= 0.04045, linear0, linear1)
    elif isinstance(srgb, np.ndarray):
        """Assumes `srgb` is in [0, 1], see https://en.wikipedia.org/wiki/SRGB."""
        eps = np.finfo(np.float32).eps
        linear0 = 25 / 323 * srgb
        linear1 = np.maximum(((200 * srgb + 11) / (211)), eps)**(12 / 5)
        return np.where(srgb <= 0.04045, linear0, linear1)
    else:
        raise NotImplementedError
```


# 坐标变换

内参矩阵 = c2p
外参矩阵 = w2c
像素坐标 = `c2p * w2c * world_position`

## 相机内参矩阵intrinsic_c2p

![image.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/20230703144039.png)

> 理解与[NeRF OpenCV OpenGL COLMAP DeepVoxels坐标系朝向_nerf坐标系_培之的博客-CSDN博客](https://blog.csdn.net/OrdinaryMatthew/article/details/126670351)一致

NeRF = OpenGL = Blender
Neus = Colmap

| Method | Pixel to Camera coordinate                                                                                                                                                                                                                                                                                         | 
| ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| NeRF   | $\vec d = \begin{pmatrix} \frac{i-\frac{W}{2}}{f} \\ -\frac{j-\frac{H}{2}}{f} \\ -1 \\ \end{pmatrix}$ , $intrinsics = K = \begin{bmatrix} f & 0 & \frac{W}{2}  \\ 0 & f & \frac{H}{2}  \\ 0 & 0 & 1 \\ \end{bmatrix}$                                                                                              | 
| Neus   | $\vec d = intrinsics^{-1} \times  pixel = \begin{bmatrix} \frac{1}{f} & 0 & -\frac{W}{2 \cdot f}  \\ 0 & \frac{1}{f} & -\frac{H}{2 \cdot f} \\ 0 & 0 & 1 \\ \end{bmatrix} \begin{pmatrix} i \\ j \\ 1 \\ \end{pmatrix} = \begin{pmatrix} \frac{i-\frac{W}{2}}{f} \\ \frac{j-\frac{H}{2}}{f} \\ 1 \\ \end{pmatrix}$ |     

## 相机外参矩阵w2c

> [相机位姿(camera pose)与外参矩阵 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/642715876)

colmap处理得到的`/colmap/sparse/0`中文件cameras, images, points3D.bin or .txt
其中images.bin中：
```
# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
# POINTS2D[] as (X, Y, POINT3D_ID)
# Number of images: 2, mean observations per image: 2
```

pose: QW, QX, QY, QZ, TX, TY, TZ

由QW、QX、QY、QZ得到旋转矩阵

```
R = | 1 - 2*(qy^2 + qz^2)   2*(qx*qy - qw*qz)   2*(qx*qz + qw*qy) |
    | 2*(qx*qy + qw*qz)     1 - 2*(qx^2 + qz^2) 2*(qy*qz - qw*qx) |
    | 2*(qx*qz - qw*qy)     2*(qy*qz + qw*qx)   1 - 2*(qx^2 + qy^2) |

t = [[TX], 
    [TY], 
    [TZ]]
```

横向catR和t得到：`w2c = [R, t]`

### Neus

$c2w = \left[\begin{array}{c|c}\mathbf{R}_{c}&\mathbf{C}\\\hline\mathbf{0}&1\\\end{array}\right]$

c2w矩阵的值直接描述了相机坐标系的朝向和原点，因此称为相机位姿。具体的，旋转矩阵的第一列到第三列分别表示了相机坐标系的X, Y, Z轴在世界坐标系下对应的方向；平移向量表示的是相机原点在世界坐标系的对应位置。
![c2w.png](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230804191552.png)


$w2c = \left[\begin{array}{c|c}\mathbf{R}&t\\\hline\mathbf{0}&1\end{array}\right] = \left[\begin{array}{c|c}\mathbf{R}_{c}^{\top}&-\mathbf{R}_{c}^{\top}\mathbf{C}\\\hline\mathbf{0}&1\\\end{array}\right]$

同理由于R转置为$R_c$，因此在w2c中，第一行的X为相机坐标系的X轴在世界坐标系下对应方向

$w2c = [R ,t] = \begin{bmatrix} X & t_{x}  \\ Y & t_{y} \\ Z & t_{z} \\ \end{bmatrix}$

i.e.: 
- $\mathbf{R} = \mathbf{R}_{c}^{\top}$ <==> $\mathbf{R}_{c} = \mathbf{R}^{\top}$
- $t = - \mathbf{R}_{c}^{\top}\mathbf{C}$ <==> $\mathbf{C} = - \mathbf{R}_{c} t$

在colmap_read_model.py中读取colmap文件信息，得到w2c并转换为c2w
在gen_cameras.py中将c2w转换为w2c，然后world_mat = intrinsic @ w2c
在Dataset()中加载world_mat，并P = world_mat @ scale_mat，对P进行decomposeProjectionMatrix，得到intrinsics和pose = w2c @ scale_mat，使得c2w时对世界坐标进行缩放`/scale_mat`，**使得训练时感兴趣物体在单位坐标系内**

#### 光线生成
pose: c2w (3,4)
- `rays_v = pose[:3,:3] @ directions_cams`
- `rays_o = pose[:3, 3].expand(rays_v.shape)`

### NeRO

在database.py的_normalize中，对self.poses(w2c)作变换，使得变换后的self.poses为w'2c，即新的世界坐标系到相机坐标系的变换
其中w2c也乘以了scale，这也是为了使得c2w时对世界坐标进行缩放`/scale_mat`，**使得训练时感兴趣物体在单位坐标系内**

```python
# pose w2c --> w'2c
# x3 = R_rec @ (scale * (x0 + offset))
# R_rec.T @ x3 / scale - offset = x0

# pose [R,t] x_c = R @ x0 + t
# pose [R,t] x_c = R @ (R_rec.T @ x3 / scale - offset) + t
# x_c = R @ R_rec.T @ x3 + (t - R @ offset) * scale
# R_new = R @ R_rec.T    t_new = (t - R @ offset) * scale
for img_id, pose in self.poses.items():
    R, t = pose[:,:3], pose[:,3]
    R_new = R @ R_rec.T
    t_new = (t - R @ offset) * scale
    self.poses[img_id] = np.concatenate([R_new, t_new[:,None]], -1)
```

#### 光线生成

pose: w2c (3,4)

$w2c = \left[\begin{array}{c|c}\mathbf{R}&t\\\hline\mathbf{0}&1\end{array}\right] = \left[\begin{array}{c|c}\mathbf{R}_{c}^{\top}&-\mathbf{R}_{c}^{\top}\mathbf{C}\\\hline\mathbf{0}&1\\\end{array}\right]$
i.e.: 
- $\mathbf{R} = \mathbf{R}_{c}^{\top}$ <==> $\mathbf{R}_{c} = \mathbf{R}^{\top}$
- $t = - \mathbf{R}_{c}^{\top}\mathbf{C}$ <==> $\mathbf{C} = - \mathbf{R}_{c} t$

$\mathbf{C} = - \mathbf{R}^{\top} t$
世界坐标系下相机原点C: `rays_o = poses[:, :, :3].permute(0, 2, 1) @ -poses[:, :, 3:]`

世界坐标系下光线方向：`rays_d = poses[idxs, :, :3].permute(0, 2, 1) @ rays_d.unsqueeze(-1)` # (rays_d = ray_batch['dirs'])

# 反射Reflection


>[基于物理着色：BRDF - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/21376124)

Phong Reflection Model
$I_{Phong}=k_{a}I_{a}+k_{d}(n\cdot l)I_{d}+k_{s}(r\cdot v)^{\alpha}I_{s}$
- 其中下标$a$表示环境光（Ambient Light），下标$d$表示漫射光（Diffuse Light），下标$s$表示镜面光（Specular Light），$k$表示反射系数或者材质颜色，$I$表示光的颜色或者亮度，$\alpha$可以模拟表面粗糙程度，值越小越粗糙，越大越光滑
- 入射方向$\mathbf{l}$，反射方向$\mathbf{r} = 2(\mathbf{n} \cdot \mathbf{l})\mathbf{n} - \mathbf{l}$，法向量$\mathbf{n}$
- ![image.png](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230803205658.png)

- 漫射光和高光分别会根据入射方向、反射方向和观察方向的变化而变化，还可以通过$\alpha$参数来调节表面粗糙程度，从而控制高光区域大小和锐利程度，而且运算简单，适合当时的计算机处理能力。


## Blinn-Phong Reflection Model
$I_{Blinn-Phong}=k_aI_a+k_d(n\cdot l)I_d+k_s(n\cdot h)^\alpha I_s$
- 半角向量$\mathbf{h}$为光线入射向量和观察向量的中间向量：$h=\frac{l+v}{||l+v||}$
- Blinn-Phong相比Phong，在观察方向趋向平行于表面时，高光形状会拉长，更接近真实情况。

## 基于物理的分析模型

1967年Torrance-Sparrow在Theory for Off-Specular Reflection From Roughened Surfaces中使用辐射度学和微表面理论推导出粗糙表面的高光反射模型，1981年**Cook-Torrance**在A Reflectance Model for Computer Graphics中把这个模型引入到计算机图形学领域，现在无论是CG电影，还是3D游戏，基于物理着色都是使用的这个模型。**Cook-Torrance**：ROBERT L. COOK 和 KENNETH E. TORRANCE，提出这个BRDF的文章叫做《A Reflectance Model for Computer Graphics》，发表于1982年。
> PBR 中的 Cook-Torrance BRDF 中，Cook-Torrance 是谁？ - 房燕良的回答 - 知乎 https://www.zhihu.com/question/351339310/answer/865238779

- 辐射度学（Radiometry）是度量电磁辐射能量传输的学科，也是基于物理着色模型的基础。
    - 能量（Energy）$Q$，单位焦耳（$J$），每个光子都具有一定量的能量，和频率相关，频率越高，能量也越高，(波长越短)。
    - 功率（Power），单位瓦特（Watts），或者焦耳／秒（J/s）。
        - 辐射度学中，辐射功率也被称为**辐射通量**（Radiant Flux）或者通量（Flux），指单位时间内通过表面或者空间区域的能量的总量，用符号$Φ$表示：$\Phi={\frac{dQ}{dt}}。$
    - 辐照度（Irradiance），单位时间内到达单位面积的辐射能量，或**到达单位面积的辐射通量**。单位$W/m^2$ ，$E={\frac{d\Phi}{dA}}。$辐照度衡量的是到达表面的通量密度
    - 辐出度（Radiant Existance），辐出度衡量的是离开表面的通量密度
        - 辐照度和辐出度都可以称为辐射通量密度（Radiant Flux Density）离光源越远，通量密度越低
    - 辐射强度
        - 立体角（Solid Angle）立体角则是度量三维角度的量，用符号$\omega$表示，单位为立体弧度（也叫球面度，Steradian，简写为sr）等于立体角在单位球上对应的区域的面积（实际上也就是在任意半径的球上的面积除以半径的平方$\omega=\frac{s}{r^{2}}$），单位球的表面积是$4\pi$，所以整个球面的立体角也是$4\pi$。
        - 我们可以用一个向量和一个立体角来表示一束光线，向量表示这束光线的指向，立体角表示这束光线投射在单位球上的面积，也就是光束的粗细。
        - 辐射强度（Radiant Intensity），指**通过单位立体角的辐射通量**。用符号$I$表示，单位W/sr，定义为$I=\frac{d\Phi}{d\omega}$
        - 辐射强度不会随距离变化而变化，不像点光源的辐照度会随距离增大而衰减，这是因为立体角不会随距离变化而变化。
    - 辐射率（Radiance），指每**单位面积每单位立体角的辐射通量密度**。用符号$L$表示，单位$W/m^{2}sr$，$L=\frac{d\Phi}{d\omega dA^{\perp}}$其中$dA^{\perp}$⊥是微分面积$dA$在垂直于光线方向的投影，如下图所示
        - ![image.png](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230731202820.png)
        - **辐射率实际上可以看成是我们眼睛看到（或相机拍到）的物体上一点的颜色**。在基于物理着色时，计算表面一点的颜色就是计算它的辐射率。
        - 辐射率不会随距离变化而衰减，这和我们日常感受一致，在没有雾霾的干扰时，我们看到的物体表面上一点的颜色并不会随距离变化而变化。**为什么辐照度会随距离增大而衰减，但是我们看到的颜色却不会衰减呢？这是因为随着距离变大，我们看到的物体上的一块区域到达视网膜的通量密度会变小，同时这块区域在视网膜表面上的立体角也会变小，正好抵消了通量密度的变化。**

## BRDF
我们看到一个表面，实际上是周围环境的光照射到表面上，然后表面将一部分光反射到我们眼睛里，双向反射分布函数BRDF（Bidirectional Reflectance Distribution Function）是描述表面入射光和反射光关系的。

对于一个方向的入射光，表面会将光反射到表面上半球的各个方向，不同方向反射的比例是不同的，我们用BRDF来表示指定方向的反射光和入射光的比例关系

$f(l,v)=\frac{dL_{o}(v)}{dE(l)}$

- $l$是入射光方向，$v$是观察方向，也就是我们关心的反射光方向。
- $dL_{o}(v)$是表面反射到$v$方向的反射光的微分辐射率。表面反射到$v$方向的反射光的辐射率为$L_{o}(v)$，来自于表面上半球所有方向的入射光线的贡献，而微分辐射率$dL_{o}(v)$特指来自方向$l$的入射光贡献的反射辐射率。$W/m^{2}sr$
- $dE(l)$是表面上来自入射光方向$l$的微分辐照度。表面接收到的辐照度为$E(l)$，来自上半球所有方向的入射光线的贡献，而微分辐照度$dE(l)$特指来自于方向$l$的入射光。$W/m^2$ 
- BRDF：f单位为$\frac{1}{sr}$

>[(32 封私信 / 44 条消息) brdf为什么要定义为一个单位是sr-1的量？ - 知乎 (zhihu.com)](https://www.zhihu.com/question/28476602/answer/41003204)

表面对不同频率的光反射率可能不一样，因此BRDF和光的频率有关。在图形学中，将BRDF表示为RGB向量，三个分量各有自己的$f$函数

---

BRDF需要处理表面上半球的各个方向，如下图使用[球坐标系](https://link.zhihu.com/?target=https%3A//en.wikipedia.org/wiki/Spherical_coordinate_system)定义方向更加方便。球坐标系使用两个角度来确定一个方向：

1. 方向相对法线的角度$\theta$，称为极角（Polar Angle）或天顶角（Zenith Angle）
2. 方向在平面上的投影相对于平面上一个坐标轴的角度$\phi$，称为方位角（Azimuthal Angle）

因此BRDF也可以写成：$f(\theta_i,\phi_i,\theta_o,\phi_o)$

![image.png](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230731204150.png)

**怎么用BRDF来计算表面辐射率**

来自方向$l$的入射**辐射率**：$L_i(l)=\frac{d\Phi}{d\omega_idA^-}=\frac{d\Phi}{d\omega_idAcos\theta_i}=\frac{dE(l)}{d\omega_icos\theta_i}$
则照射到表面来自于方向$l$的入射光贡献的**微分辐照度**：$dE(l)=L_i(l)d\omega_icos\theta_i$

- 表面反射到$v$方向的由来自于方向$l$的入射光贡献的微分辐射率：$dL_o(v)=f(l,v)\otimes dE(l)=f(l,v)\otimes L_i(l)d\omega_icos\theta_i$
- 要计算表面反射到$v$方向的来自上半球所有方向入射光线贡献的**辐射率**，可以将上式对半球所有方向的光线积分：$L_{o}(v)=\int_{\Omega}f(l,v)\otimes L_{i}(l)cos\theta_{i}d\omega_{i}$(表面反射辐射率即方向$v$观察到的颜色)

**对于点光源、方向光等理想化的精准光源（Punctual Light）**，计算过程可以大大简化。我们考察单个精准光源照射表面，此时表面上的一点只会被来自一个方向的一条光线照射到（而面积光源照射表面时，表面上一点会被来自多个方向的多条光线照射到），则辐射率：$L_o(v)=f(l,v)\otimes E_Lcos\theta_i$，or多条光线：$L_{o}(v)=\sum_{k=1}^{n}f(l_{k},v)\otimes E_{L_{k}}cos\theta_{i_{k}}$

- 这里使用光源的辐照度，对于阳光等全局方向光，可以认为整个场景的辐照度是一个常数，对于点光源，辐照度随距离的平方衰减，用公式$E_{L}=\frac{\Phi}{4\pi r^{2}}$就可以求出到达表面的辐照度，Φ是光源的功率，比如100瓦的灯泡，r是表面离光源的距离

## BRDF（Microfacet Theory）

我们用法线分布函数（Normal Distribution Function，简写为NDF）D(h)来描述组成表面一点的所有微表面的法线分布概率，现在可以这样理解：向NDF输入一个朝向ℎ，NDF会返回朝向是ℎ的微表面数占微表面总数的比例（虽然实际并不是这样，这点我们在讲推导过程的时候再讲），比如有1%的微表面朝向是ℎ，那么就有1%的微表面可能将光线反射到v方向。
![image.png](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230731210724.png)


实际上并不是所有微表面都能收到接受到光线，如下面左边的图有一部分入射光线被遮挡住，这种现象称为Shadowing。也不是所有反射光线都能到达眼睛，下面中间的图，一部分反射光线被遮挡住了，这种现象称为Masking。光线在微表面之间还会互相反射，如下面右边的图，这可能也是一部分漫射光的来源，在建模高光时忽略掉这部分光线。

![image.png](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230731210734.png)

- Shadowing和Masking用几何衰减因子（Geometrical Attenuation Factor）$G(l,v)$来建模，输入入射和出射光线方向，输出值表示光线未被遮蔽而能从$l$反射到$v$方向的比例
- 光学平面并不会将所有光线都反射掉，而是一部分被反射，一部分被折射，反射比例符合菲涅尔方程（Fresnel Equations）$F(l,h)$

则BRDF镜面反射部分：
$f(l,v)=\frac{F(l,h)G(l,v)D(h)}{4cos\theta_icos\theta_o}=\frac{F(l,h)G(l,v)D(h)}{4(n\cdot l)(n\cdot v)}$
- n为宏观表面法线
- h为微表面法线

>[为什么BRDF的漫反射项要除以π？ - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/342807202)

光照模型有很多种[1](https://zhuanlan.zhihu.com/p/342807202#ref_1)，Cook-Torrance 光照模型是最常用的。光照一般划分为漫反射和高光。漫反射模型提出的有 [Lambert](https://link.zhihu.com/?target=https%3A//en.wikipedia.org/wiki/Lambertian_reflectance)、 [Oren-Nayar](https://link.zhihu.com/?target=https%3A//en.wikipedia.org/wiki/Oren%25E2%2580%2593Nayar_reflectance_model)、[Minnaert](https://link.zhihu.com/?target=https%3A//en.wikipedia.org/wiki/Minnaert_function) 三种，它们有着不同的计算公式。Cook-Torrance 采用 Lambert 漫反射后的计算如下：
$f=k_{d}f_{lambert}+k_{s}f_{cook-torrance},\quad f_{lambert}=\frac{c_{diffuse}}{\pi}$

**反照率**（albedo）是行星物理学中用来表示天体反射本领的物理量，定义为物体的**辐射度**（radiosity）与**辐照度**（irradiance）之比。射是出，照是入，出射量除以入射量，得到**无量纲量**。绝对**黑体**（black body）的反照率是0。煤炭呈黑色，反照率 接近0，因为它吸收了投射到其表面上的几乎所有可见光。镜面将可见光几乎全部反射出去，其反照率接近1。albedo 翻译成**反照率**，与 reflectance（**反射率**）是有区别的。反射率用来表示某一种波长的反射能量与入射能量之比；而反照率用来表示全波段的反射能量与入射能量之比。BRDF 的 R 是 reflectance，方程仅关注一种波长。


## NeRO中的BRDF
$$f(\omega_{i},\omega_{0})=\underbrace{(1-m)\frac{a}{\pi}}_{\mathrm{diffuse}}+\underbrace{\frac{DFG}{4(\omega_{i}\cdot\mathbf{n})(\omega_{0}\cdot\mathbf{n})}}_{\mathrm{specular}},$$
### Specular

微面BRDF，参考上

### Diffuse

>[Physically Based Rendering - 就决定是你了 | Ciel's Blog (ciel1012.github.io)](https://ciel1012.github.io/2019/05/30/pbr/)

specular用于描述光线击中物体表面后直接反射回去，使表面看起来像一面镜子。有些光会透入被照明物体的内部。这些光线要么被物体吸收（通常转换为热量），要么在物体内被散射。有一些散射光线有可能会返回表面，被眼球或相机捕捉到，这就是diffuse light。diffuse和subsurface scattering(次表面散射)描述的都是同一个现象。
根据材质，吸收和散射的光通常具有不同波长，因此**仅有部分光被吸收，使得物体具有了颜色**。散射通常是方向随机的，具有各向同性。**使用这种近似的着色器只需要输入一个反照率(albedo)，用来描述从表面散射回来的各种颜色的光的分量**。Diffuse color有时是一个同义词。

**镜面反射和漫反射是互相排斥的**。这是因为，如果一个光线想要漫反射，它必须先透射进材质里，也就是说，没有被镜面反射。这在着色语言中被称为“Energy Conservation（能量守恒）”，意思是一束光线在射出表面以后绝对不会比射入表面时更亮。

![image.png](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230731212626.png)

金属度相当于镜面反射，金属度越高，镜面反射越强

### 蒙特卡罗采样

>[一文看懂蒙特卡洛采样方法 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/338103692)
>[简明例析蒙特卡洛（Monte Carlo）抽样方法 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/39628670)
>[逆变换采样和拒绝采样 - barwe - 博客园 (cnblogs.com)](https://www.cnblogs.com/barwe/p/14140681.html)
>[Monte Carlo method - Wikipedia](https://en.wikipedia.org/wiki/Monte_Carlo_method)

如何在不知道目标概率密度函数的情况下，抽取所需数量的样本，使得这些样本符合目标概率密度函数。这个问题简称为抽样，是蒙特卡洛方法的基本操作步骤。
![v2-eb0945aa2185df958f4568e58300e77a_1440w.gif](https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/v2-eb0945aa2185df958f4568e58300e77a_1440w.gif)


MC sampling：
- Naive Method
    - 根据概率分布进行采样。对一个已知概率密度函数与累积概率密度函数的概率分布，我们可以直接从累积分布函数（cdf）进行采样
    - 类似逆变换采样
- Acceptance-Rejection Method
    - 逆变换采样虽然简单有效，但是当累积分布函数或者反函数难求时却难以实施，可使用MC的接受拒绝采样
    - 对于累积分布函数未知的分布，我们可以采用接受-拒绝采样。如下图所示，p(z)是我们希望采样的分布，q(z)是我们提议的分布(proposal distribution)，令kq(z)>p(z)，我们首先在kq(z)中按照直接采样的方法采样粒子，接下来判断这个粒子落在途中什么区域，对于落在灰色区域的粒子予以拒绝，落在红线下的粒子接受，最终得到符合p(z)的N个粒子
    - ![image.png](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230801135729.png)

数学推导：
![image.png|500](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230801135800.png)
1. 从 $f_r(x)$ 进行一次采样 $x_i$
2. 计算 $x_i$ 的 **接受概率** $\alpha$（Acceptance Probability）:$\alpha=\frac{f\left(x_i\right)}{f_r\left(x_i\right)}$
3. 从 (0,1) 均匀分布中进行一次采样 u
4. 如果 $\alpha$≥u，接受 $x_i$ 作为一个来自 f(x) 的采样；否则，重复第1步

```python
N=1000 #number of samples needed
i = 1
X = np.array([])
while i < N:
    u = np.random.rand()
    x = (np.random.rand()-0.5)*8
    res = u < eval(x)/ref(x)
    if res:
        X = np.hstack((X,x[res])) #accept
        ++i
```

![image.png](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230801140044.png)


- **Importance Sampling**
    - 接受拒绝采样完美的解决了累积分布函数不可求时的采样问题。但是接受拒绝采样非常依赖于提议分布(proposal distribution)的选择，如果提议分布选择的不好，可能采样时间很长却获得很少满足分布的粒子。
    - $E_{p(x)}[f(x)]=\int_a^bf(x)\frac{p(x)}{q(x)}q(x)dx=E_{q(x)}[f(x)\frac{p(x)}{q(x)}]$
    - 我们从提议分布q(x)中采样大量粒子$x_1,x_2,...,x_n$，每个粒子的权重是 $\frac{p(x_i)}{q(x_i)}$，通过加权平均的方式可以计算出期望:
    - $E_{p(x)}[f(x)]=\frac{1}{N}\sum f(x_i)\frac{p(x_i)}{q(x_i)}$
        - q提议的分布，p希望的采样分布

```python
N=100000
M=5000
x = (np.random.rand(N)-0.5)*16
w_x = eval(x)/ref(x)
w_x = w_x/sum(w_x)
w_xc = np.cumsum(w_x) #accumulate

X=np.array([])
for i in range(M):
    u = np.random.rand()
    X = np.hstack((X,x[w_xc>u][0]))
```
其中，w_xc是对归一化后的权重计算的累计分布概率。每次取最终样本时，都会先随机一个(0,1)之间的随机数，并使用这个累计分布概率做选择。样本的权重越大，被选中的概率就越高。

![image.png](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230801140645.png)
