---
title: NeRFactor
date: 2023-08-11 13:27:25
tags:
  - NeRFactor
  - Shadow&Highlight
categories: NeRF/Surface Reconstruction/Shadow&Highlight
---

| Title     | NeRFactor: Neural Factorization of Shape and Reflectance Under an Unknown Illumination                                                                                                                                                                                                                        |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Author    | [Xiuming Zhang](http://people.csail.mit.edu/xiuming/)    [Pratul P. Srinivasan](https://pratulsrinivasan.github.io/)    [Boyang Deng](https://boyangdeng.com/)    [Paul Debevec](http://www.pauldebevec.com/)    [William T. Freeman](http://billf.mit.edu/)    [Jonathan T. Barron](https://jonbarron.info/) |
| Conf/Jour | TOG 2021 (Proc. SIGGRAPH Asia)                                                                                                                                                                                                                                                                                |
| Year      | 2021                                                                                                                                                                                                                                                                                                          |
| Project   | [NeRFactor (xiuming.info)](https://xiuming.info/projects/nerfactor/)                                                                                                                                                                                                                                          |
| Paper     | [NeRFactor: Neural Factorization of Shape and Reflectance Under an Unknown Illumination (readpaper.com)](https://readpaper.com/pdf-annotate/note?pdfId=4667303505241849857&noteId=1915674464034570496)                                                                                                                                                                                                                                                                                                              |

![image.png](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230814133227.png)

贡献：
- NeRFactor在未知光照条件下从图像中恢复物体形状和反射率
    - 在没有任何监督的情况下，仅使用重渲染损失、简单的平滑先验和从真实世界BRDF测量中学习的数据驱动的BRDF先验，恢复了表面法线、光能见度、反照率和双向反射分布函数(BRDFs)的3D神经场
- 通过NeRFactor的分解，我们可以用点光或光探针图像重新照亮物体，从任意视点渲染图像，甚至编辑物体的反照率和BRDF

<!-- more -->

# LIMITATIONS

尽管我们证明NeRFactor优于其变体和基线方法，但仍然存在一些重要的局限性。
- 首先，为了使光可见性计算易于处理，我们将光探针图像的分辨率限制为16 × 32，该分辨率可能**不足以生成非常硬的阴影或恢复非常高频的brdf**。因此，当物体被非常高频的照明照亮时，如图S1(情况D)所示，其中太阳像素是完全HDR的，反照率估计中可能存在镜面或阴影残余(例如花瓶上的那些)。
- 其次，为了快速渲染，我们**只考虑单弹直接照明，因此NeRFactor不能适当地考虑间接照明效果**。
- 最后，NeRFactor使用NeRF或MVS初始化其几何估计。虽然它能够在一定程度上修复NeRF造成的错误，但**如果NeRF以一种碰巧不影响视图合成的方式估计特别差的几何形状，它可能会失败**。我们在两个真实的NeRF场景中观察到这一点，它们包含远处不正确的“浮动”几何体，从输入摄像机中看不到，但在物体上投下阴影。

# CONCLUSION

在本文中，我们提出了神经辐射分解(NeRFactor)，一种从多视图图像及其相机姿势中恢复物体形状和反射率的方法。**重要的是，NeRFactor在未知光照条件下从图像中恢复这些属性**，而大多数先前的工作需要在多个已知光照条件下进行观察。为了解决这个问题的病态性质，**NeRFactor依靠先验**来估计一组合理的形状、反射率和照明，这些都能解释观察到的图像。
这些先验包括简单但有效的空间平滑约束(在多层感知器[MLPs]上下文中实现)和真实BRDFs上的数据驱动先验。我们证明NeRFactor实现了高质量的几何形状，足以进行重照明和视图合成，产生令人信服的反照率以及空间变化的brdf，并生成正确反映主光源存在或不存在的照明估计。**通过NeRFactor的分解，我们可以用点光或光探针图像重新照亮物体，从任意视点渲染图像，甚至编辑物体的反照率和BRDF**。我们相信，这项工作在从随意捕获的照片中恢复全功能3D图形资产的目标方面取得了重要进展。

# AIR

我们解决了**从一个未知照明条件下照亮的物体的多视图图像(及其相机姿势)中恢复物体的形状和空间变化反射率**的问题。这使得在任意环境、照明和编辑物体的材料属性下渲染物体的新视图成为可能。我们称之为神经辐射分解(**NeRFactor**)的方法的关键是**将神经辐射场(NeRF)** [Mildenhall et al. 2020]表示的物体的体积几何图形提取为distill表面表示，然后在求解空间变化的反射率和环境照明的同时联合优化几何图形。
具体来说，NeRFactor在没有任何监督的情况下，仅使用重渲染损失、简单的平滑先验和从真实世界BRDF测量中学习的数据驱动的BRDF先验，恢复了表面法线、光能见度、反照率和双向反射分布函数(BRDFs)的3D神经场。通过显式地建模光可视性，NeRFactor能够从反照率中分离阴影，并在任意光照条件下合成真实的软阴影或硬阴影。NeRFactor能够在合成和真实场景的这个具有挑战性和约束不足的捕获设置中恢复令人信服的自由视点重照明3D模型。定性和定量实验表明，NeRFactor在各种任务中的表现优于经典和基于深度学习的最新技术

## Introduction

从捕获的图像中恢复物体的几何形状和材料属性，这样它就可以在新的照明条件下从任意视点渲染，这是计算机视觉和图形学中一个长期存在的问题。这个问题的困难源于其**基本的欠约束性质**，以前的工作通常通过使用额外的观察(如扫描几何形状、已知的照明条件或多个不同照明条件下的物体图像)来解决这个问题，或者通过限制性假设(如假设整个物体的单一材料或忽略自阴影)来解决这个问题。在这项工作中，我们证明了从未知自然光照条件下捕获的物体图像中恢复令人信服的可照明表示是可能的，如图1所示。我们的关键见解是，我们可以首先从输入图像中优化神经辐射场(NeRF) [Mildenhall等人，2020]来初始化我们模型的表面法线和光照可见度(尽管我们表明使用多视图立体[MVS]几何也有效)，然后共同优化这些初始估计以及空间变化反射率和照明条件，以最好地解释观察到的图像。使用NeRF生成初始化的高质量几何估计有助于打破形状，反射率和照明之间固有的模糊性，从而使我们能够恢复完整的3D模型，用于令人信服的视图合成和重新照明，仅使用重新渲染损失，每个组件的简单空间平滑先验，以及新颖的数据驱动双向反射分布函数(BRDF)先验。由于NeRFactor明确而有效地模拟光能见度，因此它能够从反照率估计中去除阴影，并在任意新颖的光照条件下合成逼真的软阴影或硬阴影。

![image.png](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230814133227.png)

虽然**NeRF估计的几何形状**对于视图合成是有效的，但它**有两个限制，使其不容易用于重新照明**。
- 首先，NeRF将形状建模为一个体积场，因此计算沿着摄像机光线的每个点的阴影和能见度对于整个半球的照明来说是昂贵的。
- 其次，由NeRF估计的几何形状包含无关的高频内容，虽然在视图合成结果中不明显，但将高频伪影引入到从NeRF几何形状计算的表面法线和光可见性中。

我们通过使用NeRF几何图形的“硬表面”近似来解决第一个问题，其中我们只在每条射线的单个点上执行阴影计算，对应于体积的预期终止深度。我们通过将该表面上任何3D位置的表面法线和光可见性表示为由多层感知器(mlp)参数化的连续函数来解决第二个问题，并鼓励这些函数接近于从预训练的NeRF中获得的值，并且在空间上平滑。
因此，我们的模型，我们称之为神经辐射因子分解Factorization(NeRFactor)，将观测到的图像考虑到估计的环境照明中，以及具有表面法线、光能见度、反照率和空间变化的BRDFs的物体的3D表面表示。这使我们能够在任意的新环境照明下渲染物体的新视图。

In summary, our main technical contributions are:
- 一种将物体在未知光照条件下的图像分解为形状、反射率和照度illumination的方法，从而支持自由视点重照明(带阴影)和材质编辑
- 将nerf估计的体积密度提取到表面几何形状(具有法线和光可见性)的策略，以便在改进几何形状和恢复反射率时用作初始化
- 一种新的数据驱动的BRDF先验，通过**对实际测量的BRDF**进行潜在代码模型的训练来学习。

**输入和输出**。NeRFactor的输入是一组**在未知环境照明条件下**的物体的多视图图像以及这些图像的相机姿势。NeRFactor联合估计表面法线、光能见度、反照率、空间变化brdf和环境照明的合理集合，这些集合共同解释了观测到的视图。然后，我们使用恢复的几何形状和反射率来**合成任意光照下的新视点的物体图像**。明确建模能见度，NeRFactor能够从反照率中去除阴影，并**在任意照明下合成软阴影或硬阴影**。

**假设**。NeRFactor认为物体是由坚硬的表面组成的，每条光线只有一个交点，因此**没有对散射、透明和半透明等体积光传输效应进行建模**。此外，为了简化计算，我们**只对直接光照进行建模**。最后，我们的反射率模型考虑了具有消色差镜面反射率的材料(电介质)，因此我们**不模拟金属材料**(尽管可以通过额外预测每个表面点的镜面颜色来轻松扩展我们的金属材料模型)。

## RELATED WORK

**Inverse rendering**
[佐藤等。1997;马斯纳1998;Yu et al. 1999;Ramamoorthi and Hanrahan 2001]，将观察图像中物体的外观分解为潜在的几何形状、材料属性和光照条件的任务，是计算机视觉和图形学中长期存在的问题。由于众所周知，完整的一般反向渲染问题是严重缺乏约束的，因此大多数先前的方法都通过假设没有阴影，学习形状，照明和反射率的先验，或者需要额外的观察，例如扫描几何形状，测量的照明条件，或在多种(已知)照明条件下的物体的额外图像来解决这个问题。
单幅图像逆渲染方法Methods for single-image inverse rendering
[Barron和Malik, 2014;Li et al. 2018;Sengupta等人2019;Yu and Smith 2019;Sang and Chandraker 2020;Wei等人。2020;Li等人。2020;Lichy等人。2021]很大程度上依赖于从大型数据集中学习到的几何、反射率和照明的强大先验。最近的方法可以从单个图像中有效地推断出这些因素的合理设置，**但不能恢复可以从任意视点查看的完整3D表示**。

大多数恢复因式全3D模型的方法都依赖于额外的观测，而不是强先验。**一种常见的策略是使用主动扫描获得的3D几何图形**[Lensch等人。2003;郭等。2019;Park等人。2020;Schmitt et al. 2020;Zhang等。2021a]，代理模型[Sato等。2003;Dong等人。2014;Georgoulis et al. 2015;Gao等。2020;Chen等人]剪影面具[Oxholm和Nishino 2014;戈达尔等人。2015;夏等。2016]，或多视点立体(MVS;然后是曲面重建和网格划分)[Laffont et al. 2012;Nam et al. 2018;菲利普等人。2019;Goel等人]。作为恢复反射率和精细几何之前的起点。在这项工作中，我们表明，从使用最先进的神经体积表示估计的几何形状开始，使我们能够仅使用在一次照明下捕获的图像恢复完全分解的3D模型，而无需任何额外的观察。至关重要的是，使用这种方式估计的初始几何形状使我们能够恢复对传统几何形状估计方法具有挑战性的物体的因子模型，包括具有高反射表面和详细几何形状的物体。

计算机图形界的大量工作集中在材料获取的特定子问题上，其目标是**从已知(通常是平面)几何形状的材料图像中估计双向反射分布函数(BRDF)属性**。这些方法传统上利用基于信号处理的重建策略，并使用复杂的控制相机和照明设置来充分采样BRDF [Foo 2015;Matusik et al. 2003;尼尔森等人。2015年]，而最近的方法使得从更休闲的智能手机设置中获取材料成为可能[Aittala等人。2015;Hui et al. 2017]。**然而，这一行的工作通常要求几何形状简单且完全已知**，而我们关注的是一个更一般的问题，即我们唯一的观察是具有复杂形状和空间变化反射率的物体的图像。

我们的工作建立在计算机视觉和图形社区的最新趋势之上，**该趋势将传统的形状表示(如多边形网格或离散体素网格)替换为将几何表示为参数函数的多层感知器(mlp)**。这些mlp经过优化，可以通过将3D坐标映射到该位置的物体或场景属性(如体积密度、占用率或签名距离)来近似连续3D几何。该策略已经成功地从3D观测中恢复连续的3D形状表示[Mescheder等人。2019;Park et al. 2019;Tancik等人]。和固定照明下观测到的图像[Mildenhall等。2020;Yariv等。2020]。神经辐射场(NeRF) [Mildenhall等。2020]技术在优化观察图像的体积几何和外观方面特别成功，目的是渲染逼真的新视图

NeRF启发了后续方法，扩展其神经表征以实现重照明[Bi et al. 2020;Boss等;2021;Srinivasan等人。2021;张等。2021b]。我们列出了这些并发方法与NeRFactor之间的区别如下
- Bi等人。[2020]和NeRV [Srinivasan等。2021]**需要多个已知的照明条件**，而NeRFactor只处理一个未知的照明。
- NeRD[Boss等人,2021]不模拟能见度或阴影，而NeRFactor可以成功地将阴影与反照率分开(如下所示)。NeRD使用分析BRDF，而NeRFactor使用编码先验的学习BRDF
- PhySG [Zhang等。2021b]不模拟能见度或阴影，并使用分析BRDF，就像NeRD一样。此外，PhySG假设非空间变化的反射率，而NeRFactor模型是空间变化的brdf。

# METHOD

NeRFactor的输入被**假设为一个未知照明条件下的物体的多视图图像(以及它们的相机姿势)**。NeRFactor将物体的形状和空间变化反射率表示为一组3D字段，每个字段由多层感知器(mlp)参数化，其权重经过优化，以便“解释”观察到的输入图像集。优化后，NeRFactor的输出，在物体的表面上的每个3D位置x: 表面法向$n$,任何方向的能见度$v(\omega_i)$ ，反照率$a$和反射比reflectance $z_{BRDF}$共同解释观察到的现象。通过恢复物体的几何形状和反射率，NeRFactor支持诸如自由视点重照明(带阴影)和材料编辑等应用程序。

我们在图2中可视化NeRFactor模型和它产生的分解示例。有关实现细节，包括网络架构，训练范例，运行时等，请参阅附录的A部分和我们的GitHub存储库。

![image.png](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230814141123.png)
*Model. NeRFactor leverages NeRF's 𝜎-volume as an initialization to predict, for each surface location $𝒙_{surf}$, surface normal 𝒏, light visibility 𝑣, albedo 𝒂, BRDF latent code $𝒛_{BRDF}$, and the lighting condition. 𝒙 denotes 3D locations, $𝝎_i$ light direction, $𝝎_o$ viewing direction, and $𝜙_d, 𝜃_h, 𝜃_d$ Rusinkiewicz coordinates. Note that NeRFactor is an **all-MLP architecture** that models only surface points (unlike NeRF that models the entire volume).*

![image.png](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230814140950.png)
*Example factorization. NeRFactor jointly solves for plausible surface normals, light visibility, albedo, BRDFs, and lighting that together explain the observed views. Here we visualize light visibility as ambient occlusion and 𝑧BRDF directly as RGBs (similar colors indicate similar materials).*

*NeRFactor是一种基于坐标的模型，它可以在无监督的情况下分解在一个未知光照条件下观察到的场景的外观。它通过使用重建损失、简单的平滑正则化和数据驱动的BRDF先验来解决这个严重不适定的问题。明确建模可见性，NeRFactor是一个基于物理的模型，支持任意光照下的阴影。*

## Shape

我们模型的输入与NeRF [Mildenhall等人，2020]使用的输入相同，因此我们可以将NeRF应用于我们的输入图像以计算初始几何(尽管使用多视图立体[MVS]几何作为初始化也有效，如第4.4节所示)。NeRF优化了一个神经辐射场:一个MLP，它从任何3D空间坐标和2D观看方向映射到该3D位置的体积密度和该位置的粒子沿2D观看方向发射的颜色。NeRFactor通过将NeRF的估计几何形状“提炼distilling”成一个连续的表面表示来初始化NeRFactor的几何形状，从而利用NeRF的估计几何形状。特别是，我们使用优化的NeRF来计算沿任何相机光线的预期表面位置，物体表面上每个点的表面法线，以及从物体表面上每个点的任何方向到达的光的可见度。本小节描述了我们如何从优化的NeRF中获得这些函数，以及如何使用mlp重新参数化它们，以便在初始化步骤之后对它们进行微调，以改善完全的重新呈现损失(图3)。

**表面点**。给定一个camera和一个trained的NeRF，我们根据NeRF的优化体积密度$\sigma$，计算从摄像机原点$\text{o}$沿着方向$\text{d}$的光线$\mathbf{r}(t)=\mathbf{o}+t\boldsymbol{d}$，预期会终止的位置：

$x_{\mathrm{surf}}=\boldsymbol{o}+\left(\int_{0}^{\infty}T(t)\sigma\big(\boldsymbol{r}(t)\big)tdt\right)\boldsymbol{d},$

在不被阻挡的情况下，射线行进距离𝑡的概率可以表示为$T(t)=\exp\left(-\int_{0}^{t}\sigma\big(\boldsymbol{r}(s)\big)ds\right)$。与维护完整的体积表示不同，我们将几何结构固定在这个从优化的NeRF提取出的表面上。这在训练和推断过程中都能够实现更高效的重新光照，**因为我们只需在每个相机射线预期终止的位置计算出射辐射，而不是沿着每条相机射线的每个点都进行计算**

**表面法线**。我们在任意3D位置计算解析表面法线$𝒏_{a}(𝒙)$，它是相对于𝒙的NeRF的𝜎-体积的负归一化梯度。然而，从训练过的NeRF派生出的法线往往带有噪声（图3），因此在渲染时会产生“凹凸”瑕疵（请参阅补充视频）。
![image.png](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230814143126.png)
*NeRFactor恢复的高质量几何。 (A) 我们可以直接从训练过的NeRF中导出表面法线和光能见度。然而，用这种方式导出的几何结构太嘈杂，无法用于重新光照（请参阅补充视频）。 (B) 联合优化形状和反射性能改善了NeRF的几何结构，但仍然存在显著的噪声（例如，图II中的条纹伪影）。 (C) 在平滑性约束下进行联合优化，得到了平滑的表面法线和光能见度，类似于地面真实情况。在所有入射光方向上平均的能见度是环境遮挡。*

因此，我们使用一个MLP $f_{\mathrm{n}}$对这些法线进行重新参数化，将从表面任何位置$x_{\mathrm{surf}}$映射到“去噪”的表面法线$n\colon f_{\mathrm{n}}:x_{\mathrm{surf}}\mapsto n.$在优化NeRFactor权重的联合优化过程中，我们鼓励这个MLP的输出：
I）保持接近预训练NeRF产生的法线，
II）在3D空间中平滑变化，
III）重现物体的观察外观。具
体而言，反映I）和II）的损失函数为...
$\ell_{\mathrm{n}}=\sum_{x_{\mathrm{surf}}}\left(\frac{\lambda_{1}}{3}\|f_{\mathrm{n}}(x_{\mathrm{surf}})-n_{\mathrm{a}}(x_{\mathrm{surf}})\|_{2}^{2}+\frac{\lambda_{2}}{3}\|f_{\mathrm{n}}(x_{\mathrm{surf}})-f_{\mathrm{n}}(x_{\mathrm{surf}}+\epsilon)\|_{1}\right)$
其中𝝐是从均值为零、标准差为0.01的高斯分布中采样得到的关于$x_{\mathrm{surf}}$的随机3D位移（对于实际场景，由于不同的场景尺度，𝝐的标准差为0.001或0.25），而$𝜆_1$和$𝜆_2$是分别设定为0.1和0.05的超参数。Oechsle等人在其并行工作中也使用了类似的表面法线平滑性损失，用于形状重建的目标。至关重要的是，不限制𝒙在预期表面上，增加了MLP的稳健性，为输出提供了一个“安全余地”，即使输入略微偏离表面，输出仍保持良好行为。正如图3所示，NeRFactor的法线MLP产生的法线质量明显优于NeRF产生的法线，并且足够平滑以用于重新光照（图5）。

![image.png](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230814145305.png)


**光能见度**。我们通过从点到每个光源位置在NeRF的𝜎体积中进行行进，计算每个点到每个光源的能见度$𝑣_𝑎$，类似于Bi等人[2020]的方法。然而，与上述估计的表面法线一样，直接从NeRF的𝜎体积中得出的能见度估计过于嘈杂，无法直接使用（图3），并导致渲染产生伪影（请参见补充视频）。我们通过将能见度函数重新参数化为另一个多层感知器（MLP），该MLP从表面位置$x_{\mathrm{surf}}$和光线方向$𝝎_i$映射到光能见度$v{:}f_\mathrm{V}:(x_\mathrm{surf},\omega_\mathrm{i})\mapsto v.$。我们优化$𝑓_𝑣$的权重，以鼓励恢复的能见度场：
I）接近从NeRF跟踪的能见度，
II）空间平滑，
III）复现观察到的外观。
具体而言，实施I）和II）的损失函数如下：
$\ell_\mathrm{v}=\sum_{x_\mathrm{surf}}\sum_{\omega_\mathrm{i}}\left(\lambda_3\big(f_\mathrm{v}(x_\mathrm{surf},\omega_\mathrm{i})-v_\mathrm{a}(x_\mathrm{surf},\omega_\mathrm{i})\big)^2+\lambda_4\big|f_\mathrm{v}(x_\mathrm{surf},\omega_\mathrm{i})-f_\mathrm{v}(x_\mathrm{surf}+\epsilon,\omega_\mathrm{i})\big|\right)$
其中，𝝐 是上文定义的随机位移，$𝜆_3$ 和$𝜆_4$ 分别是设置为 0.1 和 0.05 的超参数。如等式所示，在相同的$𝝎_i$ 条件下，鼓励不同空间位置的平滑性，而不是相反。这样做的目的是为了避免某一位置的可见度在不同光照位置上变得模糊。请注意，这与 Srinivasan 等人[2021]中的可见度fields类似，但在我们的案例中，我们优化了可见度 MLP 参数，以去噪从预训练 NeRF 得出的可见度，并将重新渲染损失降至最低。为了计算 NeRF 可见度，我们使用了一组固定的 512 个光照位置，并预先定义了光照分辨率（稍后讨论）。经过优化后，$f_{\mathrm{V}}$可以生成空间上平滑且真实的光线可见度估计值，如图 3 (II) 和图 4 (C)所示，我们可以看到所有光线方向的平均可见度（即环境遮挡）。

![image.png](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230814143126.png)
![image.png](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230814145248.png)

在实践中，**在对模型进行全面优化之前，我们对可见度和法线 MLP 进行了独立预训练**，使其仅再现来自 NeRF 𝜎 卷的可见度和法线值，而没有任何平滑度正则化或重新渲染损失。这为可见度maps提供了合理的初始化，从而避免反照率或双向反射率分布函数（BRDF）MLP 将阴影误解为 "涂抹在 "反射率变化上（见表 1 和图 S2 中的 "w/o geom.）

![image.png](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230814141123.png)

## Reflectance

我们的完整 BRDF 模型𝑹包括一个完全由反照率𝒂决定的漫反射部分（朗伯）和一个镜面空间变化 BRDF$f_\mathrm{r}$（根据入射光方向$\omega_{i}$ 和出射方向$\omega_{o}$ ，定义为表面上的任意位置$x_{\mathrm{surf}}$ ）从真实世界的反射率中学习的：$R(x_{\mathrm{surf}},\omega_{\mathrm{i}},\omega_{\mathrm{o}})=\frac{a(x_{\mathrm{surf}})}{\pi}+f_{\mathrm{r}}\left(x_{\mathrm{surf}},\omega_{\mathrm{i}},\omega_{\mathrm{o}}\right).$

神经渲染的现有技术已经探索了参数化的使用。使用microfacet模型等分析brdf [Bi等。2020;Srinivasan等人。2021]在类似nerf的环境中。我们还将在第5.1节中探讨NeRFactor的“分析BRDF”版本。尽管这些分析模型为优化探索提供了有效的BRDF参数化，但没有对参数本身施加先验:在microfacet模型中可表示的所有材料都被认为是同等先验的。此外，使用显式分析模型限制了可以回收的材料集，这可能不足以对所有现实世界的brdf进行建模。
NeRFactor不是假设一个解析BRDF，而是从一个学习到的反射函数开始，该函数经过预训练，可以再现大量的经验观察到的真实世界的BRDF，同时也可以学习这些真实世界BRDF的潜在空间。通过这样做，我们可以学习真实世界brdf的数据驱动先验，从而鼓励优化恢复合理的反射函数。使用这样的先验是至关重要的:因为我们所有观察到的图像都是在一个(未知)照明下拍摄的，我们的问题是高度病态的，所以先验是必要的，可以从所有可能的分解集合中消除最可能的场景分解的歧义。

**Albedo**. 我们将任何表面位置$x_{\mathrm{surf}}$的反照率𝒂参数化为一个 MLP $f_{\mathrm{a}}:x_{\mathrm{surf}}\mapsto a.$由于没有对反照率的直接监督，而且我们的模型只能观察到一种光照条件，因此我们依靠简单的空间平滑先验（和光照可见度）来区分 "含有阴影的白漆表面 "和 "黑白相间的白漆表面 "等情况。此外，观测视图的重建损失也是优化$\text{f}_a$ 的驱动因素。反映这种平滑先验的损失函数是

$\ell_{\mathrm{a}}=\lambda_{5}\sum_{x_{\mathrm{surf}}}\frac{1}{3}\big\Vert f_{\mathrm{a}}(x_{\mathrm{surf}})-f_{\mathrm{a}}(x_{\mathrm{surf}}+\epsilon)\big\Vert_{1},$

其中，𝝐 是与上述定义相同的随机三维扰动，$𝜆_5$ 是设置为 0.05 的超参数。$\text{f}_a$的输出在朗伯反射率中用作反照率，但在非漫反射分量中不用作反照率，**我们假定非漫反射分量的镜面高光颜色为白色**。根据 Ward 和 Shakespeare [1998] 的经验，我们将反照率预测值限制在 [0.03, 0.8]，方法是将网络的最终 sigmoid 输出值缩放 0.77，然后加上 0.03 的偏差。

**Learning priors from real-world BRDFs**. 对于 BRDF 的镜面成分，我们试图学习现实世界 BRDF 的潜在空间，以及将所学空间 $𝒛_{BRDF}$ 中的每个潜在代码转换为完整 4D BRDF 的配对 "解码器"。为此，我们采用了生成式潜在优化（GLO）方法[Bojanowski 等人，2018]，其他基于坐标的模型，如 Park 等人[2019]和 Martin-Brualla 等人[2021]也曾使用过这种方法。我们模型的$\text{f}_{r}$部分是使用 MERL 数据集[Matusik 等人，2003 年]预训练的。由于 MERL 数据集假定材料各向同性，我们使用 Rusinkiewicz 坐标[Rusinkiewicz 1998]$(\phi_{\mathbf{d}},\theta_{\mathbf{h}},\theta_{\mathbf{d}})$（3 个自由度）而不是 $𝝎_i$ 和 $𝝎_o$（4 个自由度）对$𝒇_r$ 的输入和输出方向进行参数化。用 𝒈 表示这种坐标转换：$g:(n,\omega_{\mathrm{i}},\omega_{0})\mapsto(\phi_{\mathrm{d}},\theta_{\mathrm{h}},\theta_{\mathrm{d}}),$，其中𝒏 是该点的表面法线。我们训练了一个函数$f_{\mathrm{r}}^{\prime}$（$𝒇_r$ 的重新参数化），该函数将潜在代码 $𝒛_{BRDF}$（代表 BRDF 特性）和 Rusinkiewicz 坐标$(\phi_{\mathbf{d}},\theta_{\mathbf{h}},\theta_{\mathbf{d}})$的组合映射到消色差反射率 𝒓：

$f_\mathrm{r}':(z_\mathrm{BRDF},(\phi_\mathrm{d},\theta_\mathrm{h},\theta_\mathrm{d}))\mapsto r.$

为了训练这个模型，我们优化了 MLP 的权重和潜在代码集 $𝒛_{BRDF}$，以再现一组真实世界的 BRDF。计算高动态范围（HDR）反射率值对数的简单均方误差来训练$𝒇 ′_{r}$

由于我们的反射率模型中的**颜色部分假定由反照率 MLP 处理**，因此我们将 MERL 数据集的 RGB 反射率值转换为消色差值†，从而舍弃了所有颜色信息。潜在 BRDF 识别代码$𝒛_{BRDF}$ 的参数为无约束三维向量，初始化为标准偏差为 0.01 的零均值各向同性高斯。在训练过程中，没有对 $𝒛_{BRDF}$ 施加稀疏性或规范惩罚。经过预训练后，BRDF MLP 的权重在整个模型的联合优化过程中被冻结，我们通过从头开始训练 BRDF 标识 MLP（图 2a），对每个$x_{\mathrm{surf}}$ 只预测${z_{\mathrm{BRDF}}}$：$f_\mathrm{z}:x_\mathrm{surf}\mapsto z_\mathrm{BRDF}.$.这可以看作是预测真实世界 BRDF 的可信空间中所有表面点的空间变化 BRDF。我们对 BRDF 特性 MLP 进行优化，以最小化重新渲染损失和与反照率相同的空间平滑先验：
$\ell_{z}=\lambda_{6}\sum_{x_{\mathrm{surf}}}\frac{\left\|f_{z}(x_{\mathrm{surf}})-f_{z}(x_{\mathrm{surf}}+\epsilon)\right\|_{1}}{\dim(z_{\mathrm{BRDF}})},$

其中，$𝜆_6$是一个超参数，设置为 0.01；$\dim(z_{\mathrm{BRDF}})$表示 BRDF 潜在代码的维数（在我们的实现中为 3，因为 MERL 数据集中只有 100 种材料）。最终的 BRDF 是朗伯分量与学习到的非漫反射之和（为简洁起见，去掉了$x_{\mathrm{surf}}$ 的下标）：

$R(x,\omega_{\mathrm{i}},\omega_{0})=\frac{f_{\mathrm{a}}(x)}{\pi}+f_{\mathrm{r}}^{\prime}\left(f_{\mathrm{z}}(x),g(f_{\mathrm{n}}(x),\omega_{\mathrm{i}},\omega_{0})\right),$

其中镜面高光颜色假定为白色。

## Lighting


我们采用一种简单而直接的照明表示:一幅HDR光探测图像[Debevec 1998]，**采用经纬度格式**。与球面谐波或球面高斯混合相比，这种表示允许我们的模型表示详细的高频照明，因此支持硬投射阴影。也就是说，使用这种表示的挑战很明显:它包含大量参数，并且每个像素/参数都可以独立于所有其他像素而变化。这个问题可以通过使用光可见性MLP来改善，它允许我们快速评估一个表面点对光探头所有像素的可见性。从经验上讲，我们在照明环境中使用16×32分辨率，因为我们不期望恢复超出该分辨率的高频内容(照明被物体的BRDFs有效地低通过滤[Ramamoorthi和Hanrahan 2001]，而且我们的物体没有光泽或镜面一样)。

为了使光照更加平滑，我们在光探针𝑳 的像素点上沿水平和垂直两个方向应用了简单的$\ell^{2}$梯度惩罚：
$\ell_{\mathrm{i}}=\lambda_{7}\left(\left\|\begin{bmatrix}-1&&1\end{bmatrix}*L\right\|_{2}^{2}+\left\|\begin{bmatrix}-1\\1\end{bmatrix}*L\right\|_{2}^{2}\right),$

其中，∗ 表示卷积算子，$𝜆_7$ 是一个超参数，设置为 $5 × 10^{-6}$（考虑到有 512 个像素具有 HDR 值）。在联合优化过程中，这些探测像素会直接根据最终重建损失和梯度惩罚进行更新。

## Rendering

鉴于每个点 $𝒙_{surf}$ 的表面法线、所有光照方向的可见度、反照率和 BRDF 以及估计的光照，最终基于物理的不可学习渲染器会渲染一幅图像，然后将其与观测图像进行比较。渲染图像中的误差会反向传播，但不包括预训练 NeRF 的𝜎-volume，从而推动表面法线、光能见度、反照率、BRDF 和光照的联合估计。

鉴于问题的不确定性（主要是由于我们只观测到一种未知光照），我们预计大部分有用信息将来自直接光照而非全局光照，因此只考虑单跳single-bounce直接光照（即从光源到物体表面再到摄像机）。这一假设也降低了评估模型的计算成本。在数学上，我们设置的渲染方程为（为简洁起见，再次去掉$𝒙_{surf}$的下标）：

$$\begin{aligned}
L_{0}(x,\omega_{0})=\int_{\Omega}R(x,\omega_{1},\omega_{0})L_{1}(x,\omega_{1})\big(\omega_{1}\cdot n(x)\big)d\omega_{1} \\
=\sum_{\omega_{\mathrm{i}}}R(x,\omega_{\mathrm{i}},\omega_{\mathrm{0}})L_{\mathrm{i}}(x,\omega_{\mathrm{i}})\big(\omega_{\mathrm{i}}\cdot f_{\mathrm{n}}(x)\big)\Delta\omega_{\mathrm{i}}=\sum_{\omega_{\mathrm{i}}}\bigg(\frac{f_{\mathrm{a}}(x)}{\pi}+ \\
f_{\mathrm{r}}^{\prime}\Big(f_{z}(x),g\big(f_{\mathrm{n}}(x),\omega_{\mathrm{i}},\omega_{0}\big)\Big)\Big)L_{\mathrm{i}}(x,\omega_{\mathrm{i}})\big(\omega_{\mathrm{i}}\cdot f_{\mathrm{n}}(x)\big)\Delta\omega_{\mathrm{i}},
\end{aligned}
$$

其中，$𝑳_o (𝒙, 𝝎_{o})$是从𝒙 看$𝝎_{o}$的出射辐射度，$𝑳_i (𝒙, 𝝎_{i})$是入射辐射度，被可见度 $f_{\mathbf{v}}(x,\omega_{\mathbf{i}}),$遮挡、沿$𝝎_{i}$ 直接到达𝒙 的光探测像素（因为我们只考虑单跳直接照射），$\Delta\omega_{\mathrm{i}}$是在$\omega_{\mathrm{i}}$处与照明样本相对应的实体角。

最终的重建损失 $ℓ_{recon}$ 只是渲染图像与观察图像之间的均方误差（权重为单位）。因此，我们的完整损失函数是之前定义的所有损失的总和：$\ell_\mathrm{recon}+\ell_\mathrm{n}+\ell_\mathrm{v}+\ell_\mathrm{a}+\ell_\mathrm{z}+\ell_\mathrm{i}.$

# RESULTS & APPLICATIONS

在本节中，我们将展示
I) NeRFactor实现的高质量几何图形，
II) NeRFactor联合估计形状、反射率和照明的能力，
III)使用单点光或任意光探针(图5和图6)实现自由视点重照明的应用，
IV)使用MVS而不是NeRF进行形状初始化时NeRFactor的性能。
最后是材料编辑的应用(图8)。参见附录的B部分，了解在这项工作中使用的各种类型的数据是如何呈现、捕获或收集的

## Shape Optimization

NeRFactor以表面点及其相关表面法线的形式联合估计物体的形状，以及它们对每个光位置的可见性。图3显示了这些几何属性。为了可视化光的可见性，我们取16×32光探测器每个像素对应的512幅可见性图的逐像素平均值，并将该平均图(即环境遮挡)可视化为灰度图像。参见补充视频的电影的每光能见度地图(即，阴影地图)。如图3所示，我们的表面法线和光线可见度是光滑的，与地面真实情况相似，这要归功于联合估计过程，该过程最大限度地减少了重渲染错误，并鼓励空间平滑。

![image.png](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230814143126.png)

如果我们消除空间平滑性约束，只依赖于重新渲染损失，我们最终会得到不足以渲染的嘈杂几何。虽然这些几何引起的伪影可能不会在低频照明下出现，但恶劣的照明条件(如没有环境照明的单点光，即一次一灯[OLAT])会显示它们，如补充视频中所示。也许令人惊讶的是，即使当我们的平滑限制被禁用时，NeRFactor估计的几何体仍然明显比原始的NeRF几何体少噪声(比较图3的[A]和[B]，见表1的[I])，因为重渲染损失鼓励更平滑的几何体。参见5.1节了解更多细节。

## Joint Estimation of Shape, Reflectance, & Lighting

在这个实验中，我们演示了NeRFactor如何将外观分解为具有复杂几何和/或反射率的场景的形状，反射率和照明。

在对反照率进行视觉化时，我们采用了固有图像文献所使用的惯例，**即假设反照率和阴影的绝对亮度是不可恢复的**[Land and McCann 1971]，此外，我们还**假设颜色恒定问题**(解决光源平均颜色和反照率平均颜色之间的模糊问题的全局颜色校正[Buchsbaum 1980])也不在考虑范围之内。
根据这两个假设，我们将预测的反照率可视化，并测量其精度，方法是首先用一个已识别的全局标量对每个RGB通道进行缩放，以便最小化与真实反照率相比的均方误差(mean squared error)，正如Barron和Malik[2014]所做的那样。除非另有说明，否则所有合成场景的反照率预测都以这种方式进行校正，并且我们应用伽马校正($\gamma$= 2.2)在图中正确地显示它们。我们估计的光探头没有按这种方式进行缩放(因为照明估计不是这项工作的主要目标)，并且通过简单地将所有RGB通道的最大强度缩放为1，然后应用伽马校正($\gamma$= 2.2)。

如图4 (B)所示，NeRFactor预测的高质量和光滑的表面法线接近地面真相，但在具有高频细节的区域(如热狗面包的凹凸不平的表面)除外。在鼓中，我们看到NeRFactor成功地重建了精细的细节，如钹中心的螺丝和鼓侧面的金属边缘。对于榕树，NeRFactor可以恢复复杂的叶片几何形状。环境遮挡贴图也正确地描绘了场景中每个点对灯光的平均曝光。反照率的恢复干净，几乎没有任何阴影或阴影细节不准确地归因于反照率变化;请注意，在反照率预测中，鼓上的阴影是如何缺失的。此外，预测的光探头正确地反映了主光源和蓝天的位置([I]中的蓝色像素)。在这三个场景中，预测的BRDF都是空间变化的，正确反映了场景的不同部分具有不同的材料，如(E)中不同的BRDF潜码所示。

![image.png](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230814145248.png)

而不是用更复杂的表示(如球面谐波)来表示照明，我们选择了一个直接的表示:一个纬度-经度图，其像素是HDR强度。由于光照在适度漫射的BRDF反射时被低通滤波器有效地卷积[Ramamoorthi and Hanrahan 2001]，我们不期望以高于16 × 32的分辨率恢复光照。如图4 (I)所示，NeRFactor估计了一个光探测器，它正确地捕获了最左边的明亮光源和蓝天。同样，在图4 (II)中，主光源的位置也得到了正确的估计(左边明亮的白色斑点)。

## Free-Viewpoint Relighting

NeRFactor估计3D场的形状和反射率，从而实现同步重光照和视图合成。因此，本文显示的所有重光照结果和补充视频都是从新的角度呈现的。为了探测NeRFactor的极限，我们使用了苛刻的测试照明条件，即一次只打开一个点灯(OLAT)，没有环境照明。这些测试照明诱导硬投射阴影，这有效地暴露了由于不准确的几何或材料导致的渲染伪影。出于可视化的目的，我们将relit结果(使用NeRF的预测不透明度或MVS的网格轮廓)合成到背景上，背景的颜色是光探针上半部分的平均值

如图5 (II)所示，NeRFactor合成了在三种测试OLAT条件下热狗投射的正确硬阴影。NeRFactor还会在OLAT条件(I)下生成逼真的榕树渲染图，特别是当(D)中的点光源背光照射榕树时。注意，(D)中的ground truth看起来比NeRFactor的结果更亮，因为NeRFactor只模拟直接光照，而ground truth图像是用全局光照渲染的。当我们用两个新的光探针重新照亮物体时，在热导板上合成了逼真的柔和阴影(II)。在无花果中，花瓶上的镜面正确地反射了两个测试探针中的主光源。在(F)中，树叶也表现出接近地面真值的真实镜面高光。在鼓(III)中，钹被正确地估计为镜面高光，并表现出真实反射，尽管与地面真值各向异性反射(D)不同。这是预期的，因为所有MERL brdf都是各向同性的[Matusik等人，2003]。尽管NeRFactor无法解释这些各向异性反射，但它正确地将它们排除在反照率之外，而不是将它们解释为反照率涂料，因为这样做会违反反照率平滑约束，并与那些反射的视图依赖性相矛盾。在lego中，针对OLAT测试条件(IV)，使用NeRFactor合成了逼真的硬阴影。

![image.png](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230814161008.png)


**Relighting real scenes**
我们将NeRFactor应用于Mildenhall等人[2020]捕获的两个真实场景，花瓶和松果vasedeck and pinecone。这些捕获特别适合NeRFactor:每个场景有大约100个由未知环境照明的多视图图像。与NeRF一样，我们运行COLMAP Structure From Motion (SFM) [Schönberger and Frahm 2016]来获取每个视图的相机内参和外参。然后，我们训练一个vanilla NeRF来获得初始形状估计，我们将其提取到NeRFactor中，并与反射率和照度一起进行优化。如图6 (I)所示，外观被分解为表面法线、光能见度、反照率和空间变化的BRDF潜码的照明和3D场，它们共同解释了观测到的视图。通过这种分解，我们通过使用新的任意光探针替换估计的照度来重新照亮场景(图6 [II])。因为我们的分解是完全3D的，所以所有的中间缓冲都可以从任何视点渲染，并且显示的重照明结果也来自新的视点。请注意，将这些真实场景绑定在3D盒子中，以避免遥远的几何形状阻挡某些方向的光线，并在重照明时投下阴影。

![image.png](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230814161454.png)

## Shape Initialization Using Multi-View Stereo

我们已经演示了NeRFactor如何使用从NeRF中提取的几何图形作为初始化，并在将反射率和光照联合分解的同时继续细化该几何图形。在这里，我们探讨NeRFactor是否可以与其他形状初始化(如MVS)一起工作。具体来说，我们考虑DTU-MVS数据集[Jensen等人。2014;Aanæs等。2016]为每个场景提供大约50个多视图图像(以及相应的相机姿势)。我们用泊松重构[Kazhdan等人]初始化NeRFactor的形状。Furukawa和Ponce[2009]重建的MVS。有关这些数据的更多细节，请参阅附录B部分。**这个实验不仅探索了形状初始化的另一种可能性，而且还探索了NeRFactor评估的真实图像的另一个来源**。

NeRFactor实现高质量的形状估计时，从MVS几何而不是NeRF几何开始。如图7 (A, B)所示，NeRFactor估算的表面法线和光可见度没有MVS所遭受的噪声，同时具有足够的几何细节。通过这些高质量的几何估计，NeRFactor实现了与最近邻输入图像相似的逼真视图合成结果(图7 [C])。scan110的发光材料确实有助于恢复高频照明条件(比较在[C]中恢复的两种照明条件)。然后，我们进一步从这个新颖的视角，用两个新颖的光探头重新照亮场景，如图(D, E)所示。除了逼真的镜面高光外，还要注意(D)中由NeRFactor合成的阴影，这要归功于它的可见性建模。请注意，NeRFactor选择用白色反照率和金色照明来解释scan110(而不是相反)，因为在4.2节中讨论了基本的模糊性，但仍然设法用这种合理的解释来真实地重新照亮场景。

![image.png](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230814161812.png)

## Material Editing

由于NeRFactor从外观上分解了漫反射反照率和镜面BRDF，因此可以编辑反照率，非漫反射BRDF，或两者兼而有之，然后在任意光照条件下从任何视点重新渲染编辑过的对象。这里我们重写了预估$z_{\mathrm{BRDF}}$F对MERL数据集中学习到的珍珠漆pearl-paint潜在代码和从涡轮颜色图线性插值的颜色的估计反照率，基于表面点$\text{x-coordinates.}$的空间变化。如图8(左)所示，通过NeRFactor进行因子分解，我们能够在两个具有挑战性的OLAT条件下真实地再现原始估计材料。此外，编辑后的材料也通过相同的测试OLAT条件(图8[右])重新渲染了逼真的高光和硬阴影。

![image.png](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230814162003.png)


# EVALUATION STUDIES

In this section, we perform **ablation studies** to evaluate the importance of each model component and **compare NeRFactor against both classic and deep learning-based state of the art in the tasks of appearance factorization and relighting**. 
For **quantitative** evaluations, we use as metrics **Peak Signal-to-Noise Ratio (PSNR)**, **Structural Similarity Index Measure (SSIM)** [Wang et al. 2004], and **Learned Perceptual Image Patch Similarity (LPIPS)** [Zhang et al. 2018]

另见附录C.1节，关于在不同输入光照条件下对同一物体的反照率估计是否保持一致。