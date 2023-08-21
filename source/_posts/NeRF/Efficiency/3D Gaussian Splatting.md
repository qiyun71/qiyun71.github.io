---
title: 3D Gaussian Splatting
date: 2023-08-17 19:43:10
tags:
  - 3D Gaussian Splatting
  - Real-Time Rendering
categories: NeRF/Efficiency
---

| Title     | 3D Gaussian Splatting  for Real-Time Radiance Field Rendering                                                                                                                                                                                                                                    |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Author    | [Bernhard Kerbl](https://scholar.google.at/citations?user=jeasMB0AAAAJ&hl=en)* 1,2      [Georgios Kopanas](https://grgkopanas.github.io/)* 1,2      [Thomas Leimkühler](https://people.mpi-inf.mpg.de/~tleimkue/)3      [George Drettakis](http://www-sop.inria.fr/members/George.Drettakis/)1,2 |
| Conf/Jour | SIGGRAPH                                                                                                                                                                                                                                                                                         |
| Year      | 2023                                                                                                                                                                                                                                                                                             |
| Project   | [3D Gaussian Splatting for Real-Time Radiance Field Rendering (inria.fr)](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)                                                                                                                                                             |
| Paper     | [3D Gaussian Splatting for Real-Time Radiance Field Rendering (readpaper.com)](https://readpaper.com/pdf-annotate/note?pdfId=4786887855003664385&noteId=1920399556132916992)                                                                                                                     |

![image.png](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230817194605.png)

- 将点描述成高斯体，对点云进行**优化**（点云模型）
    - 高质量、非结构化的离散表示——高斯体：均值控制位置，协方差控制高斯体形状（缩放+旋转）
    - 针对3D高斯特性的优化方法，并同时进行自适应密度控制
- Splatting的渲染方式，区别于体渲染
    - 实现了使用GPU进行快速可微的渲染，允许各向异性的抛雪球(Splatting)和快速反向传播

<!-- more -->

# Math

一维高斯分布$x\sim\mathcal{N}(\mu,\sigma^2)$
- 概率密度函数$p(x)=\frac1{\sigma\sqrt{2\pi}}exp(-\frac{(x-\mu)^2}{2\sigma^2})$

三维标准高斯：$\mathbf{v}=[a,b,c]^T$ 
$$
\begin{aligned}
p(\mathbf{v})& =p(a)p(b)p(c)  \\
&=\frac1{(2\pi)^{3/2}}exp(-\frac{a^2+b^2+c^2}2) \\
&=\frac1{(2\pi)^{3/2}}exp(-\frac12\mathbf{v}^T\mathbf{v})
\end{aligned}
$$
其中p(a),p(b),p(c)~N(0,1)

然后**推广到一般三维高斯**：$\mathbf{v}=\mathbf{A}(\mathbf{x}-\mu)$
$p(\mathbf{v})=\frac{1}{(2\pi)^{3/2}}exp(-\frac{1}{2}(\mathbf{x}-\mu)^T\mathbf{A}^T\mathbf{A}(\mathbf{x}-\mu))$ , 其中$\mathbf{x}=[x,y,z]^T$ ， $\mu=[E(x),E(y),E(z)]^T$ 

- 对两边求积分：$\int p(\mathbf{v}) d\mathbf{v} = 1=\iiint_{-\infty}^{+\infty}\frac{1}{(2\pi)^{3/2}}exp(-\frac{1}{2}(\mathbf{x}-\mu)^T\mathbf{A}^T\mathbf{A}(\mathbf{x}-\mu))d\mathbf{v}$ ,其中$d\mathbf{v}=d(\mathbf{A}(\mathbf{x}-\mu))=|\mathbf{A}|d\mathbf{x}$
- 得到$1 =\iiint_{-\infty}^{+\infty}\frac{|\mathbf{A}|}{(2\pi)^{3/2}}exp(-\frac{1}{2}(\mathbf{x}-\mu)^T\mathbf{A}^T\mathbf{A}(\mathbf{x}-\mu))d\mathbf{x}$
- 因此一般三维高斯概率密度函数$p(\mathbf{x})=\dfrac{|\mathbf{A}|}{(2\pi)^{3/2}}exp(-\dfrac{1}{2}(\mathbf{x}-\mu)^T\mathbf{A}^T\mathbf{A}(\mathbf{x}-\mu))$

概密函数-协方差矩阵形式：$p(\mathbf{x})=\frac1{|\Sigma|^{1/2}(2\pi)^{3/2}}exp(-\frac12(\mathbf{x}-\mu)^T\Sigma^{-1}(\mathbf{x}-\mu))$
协方差矩阵(对称矩阵)-特征值分解：$\Sigma=U\Lambda U^T$  ，其中$U^{T}U=I$

- 本文简化为$G(x)=e^{-\frac{1}{2}(x)^{T}\Sigma^{-1}(x)}$
- 协方差矩阵可以看做一个缩放$\Lambda = S S^T$(特征值)和一个旋转$U = R$(特征向量)
    - $\Sigma=RSS^{T}R^{T}$,  $\Sigma=\mathbf{A}\mathbf{A}^{T}= (U\Lambda^{1/2})(U\Lambda^{1/2})^T$

> [如何直观地理解「协方差矩阵」？ (zhihu.com)](https://www.zhihu.com/tardis/zm/art/37609917?source_id=1003)

# L&D&C

我们的方法并非没有局限性。In regions where the scene is not well observed we have artifacts;在这些地区，其他方法也很困难(例如，图11中的Mip-NeRF360)。尽管各向异性高斯函数具有如上所述的许多优点，但我们的方法可以创建拉长的伪影或“斑点”高斯函数(见图12);同样，以前的方法在这些情况下也很困难。
![image.png](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230818141809.png)

当我们的优化产生较大的高斯分布时，偶尔也会出现弹出现象popping artifacts;这往往发生在视觉依赖的区域。产生这些弹出伪影的一个原因是光栅化器中的保护带对高斯信号的微不足道的抑制。一种更有原则的筛选方法将减轻这些人为因素。另一个因素是我们简单的可见性算法，这可能导致高斯函数突然切换深度/混合顺序。**这可以通过反锯齿来解决，我们把它留给未来的工作**。此外，我们目前没有将任何正则化应用于我们的优化;这样做将有助于处理看不见的区域和弹出的工件。

虽然我们在完整的评估中使用了相同的超参数，但早期的实验表明，降低位置学习率对于在非常大的场景(例如城市数据集)中收敛是必要的。
尽管与以前的基于点的解决方案相比，我们非常紧凑，但我们的**内存消耗**明显高于基于nerf的解决方案。在大场景的训练过程中，在我们未优化的原型中，GPU内存消耗峰值可以超过20gb。然而，这个数字可以通过仔细的底层优化逻辑实现(类似于InstantNGP)而显著降低。**渲染经过训练的场景需要足够的GPU内存来存储完整的模型**(大型场景需要几百兆字节)，**光栅化需要额外的30-500兆内存**，具体取决于场景大小和图像分辨率。我们注意到有很多机会可以进一步减少我们的方法的内存消耗。点云的压缩技术是一个研究得很好的领域[De Queiroz and Chou 2016];看看这些方法如何适应我们的表现将是很有趣的。

## DISCUSSION AND CONCLUSIONS

我们已经提出了第一种方法，真正允许实时，高质量的辐射场渲染，在各种场景和捕获风格，同时需要训练时间与最快的以前的方法竞争。
我们选择的3D高斯原语保留了优化体渲染的属性，同时直接允许快速基于飞溅的栅格化。我们的工作表明，与广泛接受的观点相反，**连续表示并不是严格必要的，以允许快速和高质量的辐射场训练**。
我们的大部分(~ 80%)训练时间都花在Python代码上，因为我们在PyTorch中构建了我们的解决方案，以允许其他人轻松使用我们的方法。只有栅格化例程是作为优化的CUDA内核实现的。我们期望将剩余的优化完全移植到CUDA，例如，在InstantNGP [Müller et al . 2022]，可以为性能至关重要的应用程序提供进一步的显着加速。
我们还展示了构建实时渲染原则的重要性，利用GPU的功能和软件光栅化管道架构的速度。这些设计选择是训练和实时渲染性能的关键，提供了比以前的体积射线marching性能的竞争优势。
看看我们的高斯函数是否可以用来对捕获的场景进行网格mesh重建，这将是很有趣的。除了广泛使用网格的实际意义之外，这将使我们能够更好地理解我们的方法在体积和表面表示之间的连续体中的位置。
总之，我们已经提出了第一个实时渲染解决方案的亮度场，与渲染质量匹配最昂贵的以前的方法，训练时间与最快的现有解决方案竞争。

# AIR

辐射场方法最近彻底改变了用多张照片或视频捕获的场景的新颖视图合成。然而，实现高视觉质量仍然需要训练和渲染成本高昂的神经网络，而最近更快的方法不可避免地要牺牲速度来换取质量。
对于无界和完整的场景(而不是孤立的对象)和1080p分辨率的渲染，**目前没有任何方法可以实现实时显示速率**。
我们介绍了三个关键要素，使我们能够在保持有竞争力的训练时间的同时实现最先进的视觉质量，重要的是能够**在1080p分辨率下**实现**高质量的实时**(≥30 fps)新视图合成。
- 首先，从相机校准过程中产生的稀疏点开始，我们用3D高斯分布表示场景，该分布保留了连续体辐射场的理想属性，用于场景优化，同时避免了在空白空间中不必要的计算;
- 其次，我们对三维高斯函数进行交错优化/密度控制，特别是优化各向异性协方差，以实现对场景的准确表示;
- 第三，我们开发了一种快速的可视性感知渲染算法，该算法支持各向异性飞溅splatting，既加速了训练，又允许实时渲染。

我们在几个已建立的数据集上展示了最先进的视觉质量和实时渲染。

## Introduction

网格和点是最常见的3D场景表示，因为它们是显式的，非常适合基于GPU/ cuda的快速光栅化。相比之下，最近的神经辐射场(NeRF)方法建立在连续场景表示的基础上，通常使用体积射线推进优化多层感知器(MLP)，用于捕获场景的新视图合成。
同样，迄今为止最有效的辐射场解决方案**建立在连续表示的基础上**，通过插值存储的值，例如，体素Plenoxels[fridovic - keil and Yu et al. 2022]或哈希InstantNGP[Müller et al . 2022]网格或点Point-nerf:[Xu et al .2022]。虽然这些方法的连续性有助于优化，但渲染所需的随机采样成本很高，并且可能导致噪声。
我们引入了一种结合了两种世界最佳的新方法: 
- 我们的3D高斯表示允许优化最先进的(SOTA)视觉质量和竞争性训练时间，
- 而我们的tile-based splatting解决方案确保在几个先前发布的数据集上以1080p分辨率以SOTA质量实时渲染Mip-NeRF 360[Barron等人2022;海德曼等人2018;Knapitsch et al. 2017]。

我们的目标是允许对多张照片拍摄的场景进行实时渲染，并以优化时间创建表示，速度与之前针对典型真实场景的最有效方法一样快。近期实现快速训练的方法Plenoxels、InstantNGP，但难以达到当前SOTA NeRF方法(即Mip-NeRF360)所获得的视觉质量[Barron et al.2022]，这需要长达48小时的培训时间。快速但质量较低的亮度场方法可以根据场景实现交互式渲染时间(每秒10-15帧)，但**在高分辨率下无法实现实时渲染**。

我们的解决方案基于三个主要组件：
- 我们首先引入三维高斯函数作为灵活和富有表现力的场景表示。我们从与以前类似nerf的方法相同的输入开始，即使用运动结构(SfM)校准的相机[Snavely等人]。并使用作为SfM过程的一部分免费生成的稀疏点云初始化3D高斯集。与大多数需要多视图立体(MVS)数据的基于点的解决方案相反[Aliev等人，2020;Kopanas等人，2021;ckert et al. 2022]，我们**仅使用SfM点作为输入就获得了高质量的结果**。
    - 请注意，**对于nerf合成数据集，我们的方法即使在随机初始化的情况下也能达到高质量**。我们展示了3D高斯函数是一个很好的选择，因为它们是一种可微的体积表示，但它们也可以be rasterized通过将它们投影到2D并应用标准$\alpha$-混合，使用等效的图像形成模型作为NeRF。
- 我们方法的第二个组成部分是优化三维高斯函数的属性-三维位置，不透明度$\alpha$，各向异性协方差和球面谐波(SH)系数-与自适应密度控制步骤交错，在**优化过程中我们添加和偶尔删除3D高斯**。优化过程产生了一个相当紧凑、非结构化和精确的场景表示(所有测试场景的1-5百万高斯)。
- 我们方法的第三个也是最后一个元素是我们的实时渲染解决方案，**它使用快速GPU排序算法**，并受到tile-based rasterization的启发，遵循最近的工作[Lassner和Zollhofer 2021]。
- 然而，由于我们的3D高斯表示，我们可以执行respects visibility ordering的各向异性splatting——这要归功于排序和𝛼-blending ，并通过跟踪尽可能多的splats的遍历实现快速准确的backward。 

综上所述，我们提供了以下贡献:
•引入各向异性3D高斯作为高质量，非结构化的辐射场表示。
•3D高斯属性的优化方法，与自适应密度控制交错，为捕获的场景创建高质量的表示。
•GPU的快速、可微分渲染方法，具有可视性感知，允许各向异性splatting和快速反向传播，以实现高质量的新视图合成。

我们在先前发布的数据集上的结果表明，**我们可以从多视图captures中优化我们的3D高斯分布**，并获得与之前最佳质量的隐式辐射场方法相同或更好的质量。我们还可以实现与最快方法相似的训练速度和质量，重要的是为新视图合成提供高质量的实时渲染。

## Related Work

首先简要概述了传统的重建，然后讨论了基于点的渲染和基于RF的渲染，讨论了它们的相似性; 辐射场是一个广阔的领域，所以我们只关注直接相关的工作。有关该领域的完整覆盖，请参阅最近的优秀调查[Tewari et al. 2022;谢等。2022]。

### Traditional Scene Reconstruction and Rendering

第一种新颖的视图合成方法是基于光场，首先密集采样[Gortler等人，1996;Levoy和Hanrahan 1996]，然后允许非结构化捕获[Buehler等，2001]。结构-从运动(SfM)的出现[Snavely等。2006]开创了一个全新的领域，一组照片可以用来合成新的views。SfM在相机校准期间估计稀疏点云，最初用于简单的3D空间可视化。随后的多视图立体(MVS)多年来产生了令人印象深刻的全3D重建算法[Goesele等人，2007]，使几种视图合成算法得以发展[Chaurasia等人，2013;Eisemann et al. 2008;海德曼等人。2018;Kopanas等人。2021]。所有这些方法都是将输入图像重新投影和融合到新的视图摄像机中，并利用几何结构来指导这种重新投影。**这些方法**在许多情况下产生了很好的结果，但**当MVS生成不存在的几何形状时，通常不能从未重建的区域或“过度重建”中完全恢复**。最近的神经渲染算法[Tewari et al. 2022]大大减少了这种伪影，避免了在GPU上存储所有输入图像的巨大成本，在大多数方面都优于这些方法。

### Neural Rendering and Radiance Fields

深度学习技术很早就被用于新颖视图合成[Flynn et al. 2016;Zhou et al. 2016]; cnn用于估计混合权重[Hedman等人，2018]，或用于纹理空间解决方案[Riegler和Koltun 2020;他们等人。2019]。**使用基于mvs的几何是大多数这些方法的主要缺点**; 此外，使用cnn进行最终渲染经常会导致时间闪烁temporal flickering.。
新视角合成的体积表示由Soft3D提出[Penner and Zhang 2017]; 随后提出了与体积ray-marching相结合的深度学习技术[Henzler等人，2019;Sitzmann et al. 2019]基于连续可微密度场来表示几何。**由于查询体积需要大量的样本，因此使用体积射线行进渲染具有显着的成本**。神经辐射场(Neural Radiance Fields, nerf) [Mildenhall et al. 2020]引入了重要性采样和位置编码来提高质量，**但使用了大型多层感知器，对速度产生了负面影响**。NeRF的成功导致了后续方法的爆炸式增长，这些方法通常是通过引入规范化策略来解决质量和速度问题; 目前最先进的新视角合成图像质量是Mip-NeRF360 [Barron et al. 2022]。**虽然渲染质量非常出色，但训练和渲染时间仍然非常高**; 在提供快速培训和实时渲染的同时，我们能够达到或在某些情况下超过这种质量。

最近的方法主要集中在更快的训练和/或渲染上，主要是通过利用三种设计选择: **使用空间数据结构来存储(神经)特征**，这些特征随后在体射线行进期间被插值，**不同的编码**和**MLP容量**。这些方法包括空间离散化的不同变体different variants of space discretization[Chen et al. 2022b,a;fridovic - keil和Yu等人。2022;Garbin et al. 2021;海德曼等人。2021;Reiser等人。2021;Takikawa等人。2021;Wu等人。2022;Yu et al. 2021]，码本codebooks[Takikawa et al. 2022]，以及哈希表等编码encodings such as hash tables[m<s:1> ller et al. 2022]，允许完全使用较小的MLP或前述foregoing神经网络[friovich - keil and Yu et al. 2022;Sun et al. 2022]。
这些方法中最值得注意的是：
InstantNGP ，使用哈希网格和占用网格来加速计算，并使用较小的MLP来表示密度和外观;
Plenoxels [friovich - keil and Yu et al. 2022]使用稀疏体素网格来插值连续密度场，并且能够完全放弃神经网络。
两者都依赖于**球面谐波**: 前者直接表示方向效果，后者将其输入编码到颜色网络。虽然两者都提供了出色的结果，**但这些方法仍然难以有效地表示空白空间**，这部分取决于场景/捕获类型。此外，**图像质量**在很大程度上**受到用于加速的结构化网格的选择的限制**，而**渲染速度**则**受到需要为给定的射线行进步骤查询许多样本的阻碍**。
我们使用的非结构化，显式gpu友好的**3D高斯函数**实现了更快的渲染速度和更好的质量，而**不需要神经组件**。

### Point-Based Rendering and Radiance Fields

基于点的方法有效地渲染不连贯和非结构化的几何样本(即点云)[Gross and Pfister 2011]。在其最简单的形式中，点样本渲染[Grossman and Dally 1998]栅格化具有固定大小的非结构化点集，为此它可以利用本地支持的点类型图形api [Sainz and Pajarola 2004]或GPU上的并行软件栅格化[Laine and Karras 2011; Schütz et al . 2022]。虽然对底层数据是真实的，**但点样本渲染存在漏洞，导致混叠，并且是严格不连续的**。高质量的基于点的渲染开创性工作解决了这些问题，通过“Splatting”点基元的范围大于一个像素，例如圆形或椭圆形圆盘，椭球体或冲浪[Botsch等人，2005;菲斯特等人。2000;Ren et al. 2002;Zwicker et al. 2001b]。

最近人们对基于点的可微分渲染技术产生了兴趣[Wiles等人2020;Yifan et al. 2019]。用神经特征增强点并使用CNN进行渲染[Aliev等人2020;ckert et al. 2022]导致快速甚至实时的视图合成; **然而，它们仍然依赖于初始几何形状的MVS**，因此继承了它的artifacts，最明显的是在无特征/闪亮区域或薄结构等困难情况下的过度或重建不足。

**Point-based 𝛼-blending and NeRF-style volumetric rendering** 图像形成模式基本相同. 具体来说，颜色c是由沿射线的体绘制给出的：
$C=\sum_{i=1}^{N}T_{i}(1-\exp(-\sigma_{i}\delta_{i}))\mathbf{c}_{i}\quad\mathrm{with}\quad T_{i}=\exp\left(-\sum_{j=1}^{i-1}\sigma_{j}\delta_{j}\right),$ Eq.1
where 
- samples of density 𝜎, 
- transmittance 𝑇 , 
- and color c 
are taken along the ray with intervals $𝛿_𝑖$ . 

**This can be re-written as**
$C=\sum_{i=1}^{N}T_{i}\alpha_{i}\mathbf{c}_{i},$ 其中 $\alpha_i=(1-\exp(-\sigma_i\delta_i))\text{and}T_i=\prod_{j=1}^{i-1}(1-\alpha_i).$Eq.2

一种典型的基于神经点的方法(例如，[Kopanas et al. 2022, 2021])计算颜色c通过将重叠在像素上的N个有序点混合在一起，得到一个像素;
$C=\sum_{i\in\mathcal{N}}c_{i}\alpha_i\prod_{j=1}^{i-1}(1-\alpha_{j}),$Eq.3
where 
- $c_𝑖$ is the color of each point 
- $𝛼_𝑖$ is given 通过计算二维高斯函数的协方差 Σ [Yifan et al . 2019] 乘以学习到的每个点的不透明度。

从Eq. 2和Eq. 3可以清楚地看到，**图像的形成模型是相同的**。**然而，渲染算法是非常不同的**。
- nerf是一个连续的表示，隐含地表示空/占用空间; **为了找到**Eq. 2中的**样本，需要进行昂贵的随机抽样**，由此产生噪声和计算开销。
- 相比之下，**点是一种非结构化的离散表示，它具有足够的灵活性，可以创建、破坏和替换类似于NeRF的几何体**。这是通过优化不透明度和位置来实现的，正如之前的工作[Kopanas等人，2021]所示，同时避免了完整体积表示的缺点。

Pulsar[Lassner and Zollhofer 2021]实现了快速球体光栅化，这启发了我们tile-based and sorting renderer。然而，根据上面的分析，我们想要保持(近似)传统𝛼-blending on sorted splats，以具有体积表示的优势: 与顺序无关的方法相比order-independent，我们的**栅格化respects visibility order**。
此外，我们在一个像素的所有splats上反向传播梯度，并对各向异性splats进行光栅化。这些元素都有助于我们的高视觉质量的结果(见7.3节)。此外，前面提到的方法也使用cnn进行渲染，导致时间不稳定temporal instability。尽管如此，Pulsar [Lassner and Zollhofer 2021]和ADOP [Rückert et al . 2022] 的渲染速度成为我们开发快速渲染解决方案的动力。

在关注镜面效果的同时，Neural Point Catacaustics [Kopanas等人，2022]的基于漫射点的渲染轨迹通过使用MLP克服了这种时间不稳定性，**但仍然需要MVS几何作为输入**。
该类别中最新的方法[Zhang et al. 2022]不需要MVS，并且还使用SH作为方向; **然而，它只能处理一个对象的场景，并且需要掩码进行初始化**。虽然对于小分辨率和低点计数来说速度很快，但尚不清楚它如何扩展到典型数据集的场景[Barron et al. 2022;海德曼等人。2018;Knapitsch et al. 2017]。
我们使用3D高斯图像进行更灵活的场景表示，**避免了对MVS几何图形的需要**，**并实现了实时渲染**，这要归功于我们对投影高斯图像的tile-based rendering algorithm。

最近的一种方法[Xu et al .2022]采用径向基函数方法，用点来表示辐射场。他们在优化过程中采用点修剪和致密化技术，**但使用体积射线推进，无法实现实时显示速率**。

在人体行为捕捉领域，已经使用三维高斯函数来表示捕获的人体[Rhodin et al. 2015;Stoll et al. 2011]; 最近，它们被用于视觉任务的体积射线行进[Wang等人2023]。在类似的背景下也提出了神经体积原语Neural volumetric primitives[Lombardi等人，2021]。虽然这些方法启发了选择3D高斯作为我们的场景表示，**但它们侧重于重建和渲染单个孤立对象(人体或面部)的具体情况**，**从而产生深度复杂性较小的场景**。

相比之下，我们**对各向异性协方差的优化**、**交错优化/密度控制**以及**高效的深度排序渲染**使我们**能够处理完整、复杂的场景，包括室内和室外的背景，以及深度复杂性很大的场景**。

# OVERVIEW

我们方法的输入是一组静态场景图像，以及通过 SfM [Schönberger 和 Frahm 2016] 校准的相应摄像机，该方法会产生稀疏点云作为side-effect。
我们**从这些点中创建了一组三维高斯**（第 4 章），由位置（均值）、协方差矩阵和不透明度𝛼 定义，可以实现非常灵活的优化机制。这使得三维场景的表示相当紧凑，部分原因是高度各向异性的体积斑块可以用来紧凑地表示精细结构。辐射场的**方向性外观分量**（颜色）**通过球面谐波（SH）表示**，遵循标准做法[Fridovich-Keil 和 Yu 等人，2022；Müller 等人，2022]。
我们的算法**通过一系列三维高斯参数的优化步骤**，即位置、协方差、𝛼 和 SH 系数，以及高斯密度自适应控制的交错操作，来**创建辐射场表示**（第 5 节）。
我们的方法之所以高效，关键在于我们**tile-based rasterizer**（第 6 章），它可以对各向异性的splats进行𝛼 混合，并通过快速排序遵守可见性顺序。快速栅格化器还包括通过跟踪累积的𝛼值实现的fast backward，对可接收梯度的高斯数量没有限制。图 2 展示了我们的方法概览。

![image.png](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230817210209.png)
*优化从稀疏的SfM点云开始，并创建一组3D高斯点云。然后，我们优化和自适应控制这组高斯函数的密度。在优化过程中，我们使用快速的基于tile的渲染器，与SOTA快速辐射场方法相比，允许有竞争力的训练时间。一旦训练，我们的渲染器允许实时导航的各种场景。*

# DIFFERENTIABLE 3D GAUSSIAN SPLATTING

我们的目标是从没有法线的稀疏点集(SfM)开始，优化一个场景表示，允许高质量的新视图合成。要做到这一点，我们需要**一个原语，它继承了可微分体积表示的属性，同时是非结构化和显式的，以允许非常快速的渲染**。我们选择3D高斯分布，它是可微的，可以很容易地投影到2D分布，允许快速$\alpha$-混合渲染。

我们的表示方法与之前使用二维点的方法有相似之处[Kopanas 等人，2021 年；Yifan 等人，2019 年]，并假定每个点都是具有法线的平面小圆。鉴于 SfM 点的极度稀疏性，估计法线非常困难。同样，从这样的估计中优化噪声非常大的法线也非常具有挑战性。相反，**我们将几何建模为一组无需法线的 3D 高斯**。我们的高斯是由以点（平均值） 𝜇 为中心的全三维协方差矩阵 Σ 在世界空间中定义的[Zwicker 等人，2001a]：

$G(x)=e^{-\frac{1}{2}(x)^{T}\Sigma^{-1}(x)}$
这个高斯函数在我们的混合过程中乘以$\alpha$。
不过，我们需要将三维高斯投影到二维空间，以便进行渲染。Zwicker 等人 [2001a] 演示了如何将三维高斯投影到图像空间。给定一个视图变换 W，摄像机坐标中的协方差矩阵 Σ′ 如下所示：

$\Sigma'=JW\Sigma W^{T}J^{T}$
其中，𝐽 是投影变换仿射近似的雅各比。Zwicker 等人[2001a]的研究还表明，如果我们跳过 Σ′ 的第三行和第三列，就会得到一个 2×2 的方差矩阵，其结构和性质与以前的研究[Kopanas 等人，2021]中从有法线的平面点出发的方差矩阵相同。
一个明显的方法是直接优化协方差矩阵Σ来获得代表辐射场的三维高斯函数。**然而，协方差矩阵只有在正半定时才有物理意义**。对于我们的所有参数的优化，我们使用梯度下降，不能轻易地约束产生这样的有效矩阵，更新步骤和梯度可以很容易地创建无效的协方差矩阵。
**因此，我们选择了一种更直观，但同样具有表现力的表示来进行优化**。三维高斯函数的协方差矩阵Σ类似于描述椭球体的结构。给定一个缩放矩阵S和旋转矩阵R，我们可以找到相应的Σ:
$\Sigma=RSS^{T}R^{T}$

**为了允许独立优化这两个因素**，我们将它们分开存储:一个3D矢量s缩放和四元数q表示旋转。这些可以简单地转换成它们各自的矩阵并组合起来，确保归一化q得到一个有效的单位四元数。
**为了避免在训练过程中由于自动微分造成的巨大开销**，我们明确地推导了所有参数的梯度。具体的导数计算细节见附录A。
这种各向异性协方差的表示-适合于优化-允许我们优化3D高斯函数以适应捕获场景中不同形状的几何形状，从而**产生相当紧凑的表示**。图3说明了这些情况。

![image.png](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230818133754.png)

# OPTIMIZATION WITH ADAPTIVE DENSITY CONTROL OF 3D GAUSSIANS

**我们方法的核心**是优化步骤，它创建一个密集的3D高斯集，准确地代表自由视图合成的场景。除了positions 𝑝, 𝛼, and covarianceΣ，我们还优化了代表颜色c的SH系数，每一个高斯的正确捕获视图依赖的场景外观。这些参数的优化与控制高斯密度的步骤交织在一起，以更好地表示场景。

## Optimization

优化是基于渲染的连续迭代，并将结果图像与捕获数据集中的训练视图进行比较。不可避免地，由于3D到2D投影的模糊性，几何体可能会被错误地放置。因此，我**们的优化需要能够创建几何体，并在几何体位置不正确时破坏或移动它**。三维高斯分布的协方差参数的质量对于表示的紧凑性至关重要，因为大的均匀区域可以用少量的大各向异性高斯分布来捕获。
我们使用随机梯度下降技术进行优化，充分利用标准gpu加速框架，以及为某些操作添加自定义CUDA内核的能力，遵循最近的最佳实践[fridovic - keil和Yu等人。2022;Sun et al. 2022]。特别是，我们的**快速栅格化**(参见第6节)对于优化的效率至关重要，因为它是优化的主要计算瓶颈。
我们使用s型激活函数为了将$\alpha$其约束在`[0−1)`范围内并获得平滑的梯度，与用指数激活函数表示协方差的尺度原因类似。
我们将初始协方差矩阵估计为各向同性高斯矩阵，其轴等于到最近的三个点的距离的平均值。我们使用类似于Plenoxels的标准指数衰减调度技术[friovich - keil and Yu et al. 2022]，但仅用于位置。损失函数为L1结合D-SSIM项:
$\mathcal{L}=(1-\lambda)\mathcal{L}_{1}+\lambda\mathcal{L}_{\mathrm{D-SSIM}}$
我们用$\lambda= 0.2$在我们所有的测试中我们在第7.1节中提供了学习时间表和其他要素的详细信息

## Adaptive Control of Gaussians

我们从SfM的初始稀疏点集开始，然后应用我们的方法**自适应地控制单位体积上的高斯点的数量及其密度**，使我们能够从初始的高斯稀疏集到更密集的集，更好地代表场景，并具有正确的参数。在优化预热之后(参见7.1节)，我们每100次迭代致密化一次，并删除任何本质上透明的高斯分布，即$\alpha$低于阈值$\epsilon_{\alpha}$ 。
我们对高斯函数的自适应控制需要填充空白区域。它侧重于缺少几何特征的区域(“欠重建”)，但也适用于场景中高斯分布覆盖大面积的区域(通常对应于“过度重建”)。我们观察到两者都具有较大的视图空间位置梯度。直观地说，这可能是因为它们对应的区域还没有很好地重建，而优化试图移动高斯函数来纠正这一点。
由于这两种情况都是致密化的良好候选者，因此我们对具有高于阈值的视图空间位置梯度的平均幅度的高斯函数进行致密化$\tau_{\mathrm{pos}}$，我们在测试中将其设置为0.0002。
We next present details of this process, illustrated in Fig. 4.
![image.png](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230818134813.png)
*我们的自适应高斯致密化方案。顶行(重建不足):当小尺度几何(黑色轮廓)没有被充分覆盖时，我们克隆相应的高斯。下行(过度重建):如果小尺度几何由一个大的碎片表示，我们将其分成两部分。*

对于处于重建不足区域的小高斯函数，我们需要覆盖必须创建的新几何。对于这种情况，最好是克隆高斯函数，通过简单地创建一个相同大小的副本，并沿着位置梯度的方向移动它。
另一方面，高方差区域中的大高斯分布需要被分割成小高斯分布。我们用两个新的高斯函数替换这些高斯函数，并将它们的尺度除以$\phi=1.6$，我们通过实验确定。我们还通过使用原始的3D高斯作为采样的PDF来初始化它们的位置。

在第一种情况下，我们检测并处理增加系统的总体积和高斯数的需要，而在第二种情况下，我们保留总体积，但增加高斯数。
与其他体积表示类似，我们的优化可能会遇到靠近输入摄像头的**漂浮物**; 在我们的例子中，这可能会导致高斯密度的不合理增加。调节高斯函数数量增加的有效方法是每迭代𝑁 = 3000 次，将𝛼 值设置为接近零。优化后，我们会在需要的地方增加高斯的 𝛼，同时允许我们的剔除方法删除 𝛼 小于 𝜖𝛼 的高斯。如上所述。高斯分布可能会缩小或增长，并与其他分布有相当大的重叠，但我们会定期删除在世界空间中非常大的高斯分布和在视图空间中占用空间很大的高斯分布。这种策略可以很好地控制高斯函数的总数。我们模型中的高斯函数在欧几里得空间中始终保持原函数;与其他方法不同[Barron et al. 2022;fridovitch - keil和Yu等人2022]，我**们不需要空间压缩、翘翘或投影策略来处理远距离或大型高斯分布**。

# FAST DIFFERENTIABLE RASTERIZER FOR GAUSSIANS

我们的目标是有快速的整体渲染和快速排序，以允许近似𝛼-blending-包括各向异性splats-并避免对splats数量的硬限制，这些splats可以接收以前工作中存在的梯度[Lassner和Zollhofer 2021]。
为了实现这些目标，我们设计了一个tile-based rasterizer Gaussian splats，灵感来自于最近的软件光栅化方法[Lassner和Zollhofer 2021]，一次对整个图像进行预排序，避免了以往𝛼 混合解决方案中每个像素的排序费用[Kopanas等人，2022,2021]。我们的快速光栅化器允许在任意数量的混合高斯上进行有效的反向传播，并且额外的内存消耗很低，每像素只需要恒定的开销。我们的栅格化管道是完全可微的，并且给定2D投影(第4节)，可以栅格化各向异性splats，类似于以前的2D splats方法[Kopanas et al. 2021]。
我们的方法首先将屏幕分割为16×16块tiles，然后针对视锥体和每个块继续剔除3D高斯。具体来说，我们**只保留与视锥体相交的具有99%置信区间的高斯函数**。此外，我们使用保护带在极端位置trivially reject Gaussians(即，那些接近近平面和远离视锥台的平均值)，因为计算它们的投影二维协方差将是不稳定的。然后，我们根据重叠的贴图数量实例化每个高斯函数，并为每个实例分配一个键，该键结合了视图空间深度和贴图ID。然后，我们使用一个快速的GPU基数排序(Merrill and Grimshaw 2010)，基于这些键对高斯函数进行排序。需要注意的是，没有额外的按像素排序的点，混合是基于这种初始排序执行的。因此，在某些配置中，我们的𝛼混合可能是近似的。不过，当斑点的大小接近单个像素时，这些近似值就可以忽略不计了。我们发现这种选择大大提高了训练和渲染性能，而不会在融合场景中产生可见的工件artifacts。
在对高斯进行排序后，我们会通过识别第一个和最后一个深度排序的条目来为每个瓦片生成一个列表，这些条目会溅射到给定的瓦片上。在光栅化过程中，我们为每个瓦片启动一个线程块。每个线程块首先协同将高斯数据包加载到共享内存中，然后针对给定像素，通过前后遍历列表来累积颜色和𝛼 值，从而最大限度地提高数据加载/共享和处理的并行性。**当某个像素达到目标饱和度𝛼 时，相应的线程就会停止**。每隔一段时间，我们就会对一个平铺中的线程进行查询，当所有像素都达到饱和（即𝛼 变为 1）时，整个平铺的处理就会终止。有关排序的详细信息和整个光栅化方法的高级概述见附录 C。
在光栅化过程中，𝛼 的饱和度是唯一的停止标准。与之前的工作不同，我们**不限制接受梯度更新的混合基元的数量。我们强制执行这一特性，是为了让我们的方法能够处理具有任意不同深度复杂度的场景**，并准确地学习它们，而无需诉诸特定场景的超参数调整。因此，在后向处理过程中，我们必须恢复前向处理过程中每个像素混合点的完整序列。一种解决方案是在全局内存中存储每个像素任意长的混合点列表[Kopanas 等人，2021 年]。为了避免隐含的动态内存管理开销，我们选择再次遍历每个tile列表；我们可以重复使用前向遍历中的高斯排序数组和瓦片范围。为了方便梯度计算，我们现在从后向前遍历它们
遍历从影响tile中任何像素的最后一个点开始，并且再次协作地将点加载到共享内存中。此外，每个像素只有在深度低于或等于前向传递过程中对其颜色做出贡献的最后一个点的深度时，才会开始(昂贵的)重叠测试和处理。第4节中描述的梯度计算需要原始混合过程中每一步的不透明度值的累积。而不是遍历一个明确的列表逐步缩小的不透明度在向后传递，我们可以恢复这些中间的不透明度，只存储总累积的不透明度在向前传递结束。具体来说，每个点都存储了前向过程中最终累积的不透明度𝛼；我们在从后到前的遍历中将其除以每个点的𝛼，以获得梯度计算所需的系数。

# IMPLEMENTATION, RESULTS AND EVALUATION

接下来，我们将讨论实现的一些细节，目前的结果以及与以前的工作和消融研究相比我们的算法的评估。


# 实验

## 环境配置

实例镜像
PyTorch  2.0.0
Python  3.8(ubuntu20.04)
Cuda  11.8

- `git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive`
    - 修改environment.yml中的环境名称：gaussian_splatting
- conda env create --file environment.yml
- conda activate gaussian_splatting

安装colmap用于convert.py

```bash
sudo apt update
sudo apt upgrade
sudo apt-get install \
    git \
    cmake \
    build-essential \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libboost-test-dev \
    libeigen3-dev \
    libsuitesparse-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev
# 如果一次安装不上，可以继续下步，缺什么装什么
```

```bash
sudo apt install colmap

The following packages have unmet dependencies:  
colmap : Depends: libqt5gui5 (>= 5.4.0) but it is not going to be installed  
Depends: libqt5widgets5 (>= 5.4.0) but it is not going to be installed  
E: Unable to correct problems, you have held broken packages.
    sudo apt install libqt5gui5
    Depends: libegl1 but it is not going to be installed  
    Depends: libxkbcommon-x11-0 (>= 0.5.0) but it is not going to be installed  
        sudo apt install libegl1
        Depends: libglvnd0 (= 1.0.0-2ubuntu2.3) but 1.3.2-1~ubuntu0.20.04.2 is to be installed  
        Depends: libegl-mesa0 but it is not going to be installed 
            sudo apt install libglvnd0=1.0.0-2ubuntu2.3
            sudo apt install libegl-mesa0
                sudo apt install libglapi-mesa=20.0.8-0ubuntu1~18.04.1
            sudo apt install libegl-mesa0
        sudo apt install libxkbcommon-x11-0
        Depends: libxkbcommon0 (= 0.8.2-1~ubuntu18.04.1) but 0.10.0-1 is to be installed
            sudo apt install libxkbcommon0=0.8.2-1~ubuntu18.04.1
        sudo apt install libxkbcommon-x11-0
    sudo apt install libqt5widgets5
```

## 运行

[gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)

> [remote viewer cannot find the model path · Issue #35 · graphdeco-inria/gaussian-splatting (github.com)](https://github.com/graphdeco-inria/gaussian-splatting/issues/35) 服务端和本地端命令

```
- ./data/db/playroom
- ./data/db/drjohnson
- ./data/Miku
# remote
ssh -p 22938 root@connect.beijinga.seetacloud.com
zZF/91WLl1

conda activate gaussian_splatting
python train.py -s ./data/Miku  # --port 6009 # 在服务端127.0.0.1:6009上运行程序

# new terminal 将远程6009映射到本地6009
ssh -L 6009:localhost:6009 root@connect.beijinga.seetacloud.com -p 22938

# new terminal 运行本地SIBR GUI程序
SIBR_remoteGaussian_app -s E:\Download\tandt_db\db\drjohnson --port 6009 --rendering-size 480 240 --force-aspect-ratio
```

SIBR_remoteGaussian_app中操作：
- W, A, S, D, Q, E控制相机移动
- I, K, J, L, U, O控制相机旋转

自定义数据集- ./data/Miku
- `python convert.py -s <location> [--resize] #If not resizing, ImageMagick is not needed`
- python convert.py -s ./data/Miku
