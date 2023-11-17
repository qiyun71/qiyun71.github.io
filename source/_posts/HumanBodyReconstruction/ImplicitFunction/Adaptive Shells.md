---
title: Adaptive Shells
date: 2023-11-17 15:07:48
tags:
  - NeRF
  - 3DReconstruction
categories: HumanBodyReconstruction/ImplicitFunction
---

| Title     | Adaptive Shells for Efficient Neural Radiance Field Rendering                                                                                                |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Author    | Zian Wang and Tianchang Shen and Merlin Nimier-David and Nicholas Sharp and Jun Gao and Alexander Keller and Sanja Fidler and Thomas M\"uller and Zan Gojcic |
| Conf/Jour | ACM Trans. On Graph. (SIGGRAPH Asia 2023)                                                                                                                    |
| Year      | 2023                                                                                                                                                         |
| Project   | [Adaptive Shells for Efficient Neural Radiance Field Rendering (nvidia.com)](https://research.nvidia.com/labs/toronto-ai/adaptive-shells/)                   |
| Paper     | [Adaptive Shells for Efficient Neural Radiance Field Rendering (readpaper.com)](https://readpaper.com/pdf-annotate/note?pdfId=2053478190589858048&noteId=2053479143116332288)                                                                                                                                                             |

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20231117150239.png)

<!-- more -->

# Abstract

神经辐射场在新视图合成中实现了前所未有的质量，但它们的体积公式仍然昂贵，需要大量的样本来渲染高分辨率图像。体积编码对于表示模糊几何(如树叶和毛发)是必不可少的，它们非常适合于随机优化。然而，许多场景最终主要由固体表面组成，这些表面可以通过每个像素的单个样本精确地渲染。基于这一见解，我们提出了一种神经辐射公式，可以在基于体积和基于表面的渲染之间平滑过渡，大大加快渲染速度，甚至提高视觉保真度。
我们的方法构建了一个明确的网格包络，该包络在空间上约束了神经体积表示。在固体区域，包络几乎收敛到一个表面，通常可以用单个样本渲染。为此，我们推广了 NeuS [Wang et al. 2021]公式，**使用学习的空间变化核大小来编码密度的传播，将宽核拟合到类体积区域，将紧核拟合到类表面区域**。然后，我们在表面周围提取一个窄带的显式网格，宽度由核尺寸决定，并微调该波段内的辐射场。在推断时，我们将光线投射到网格上，并仅在封闭区域内评估辐射场，大大减少了所需的样本数量。实验表明，我们的方法可以在非常高的保真度下实现高效的渲染。我们还演示了提取的包络支持下游应用程序，如动画和仿真。

## Introduction

使用 InstantNGP 加速：nerf 已经可以渲染每条射线具有不同数量的样本的图像
- 首先，基于网格的加速结构的内存占用随着分辨率的增加而减少
- 其次，mlp 的平滑感应偏置阻碍了学习体积密度的尖锐脉冲或阶跃函数，即使学习了这样的脉冲，也很难有效地对其进行采样
- 最后，由于缺乏约束，隐式体积密度场不能准确地表示下表面 underlying surfaces—— NeuS [Wang et al. 2021]，这往往限制了它们在依赖网格提取的下游任务中的应用

为了弥补最后一点，[Wang et al. 2021, 2022a;Yariv 等人]提出优化带符号距离函数(SDF)以及编码密度分布的核大小，而不是直接优化密度。虽然这对于改善表面表示是有效的，但使用**全局核大小**与场景的不同区域需要自适应处理的观察相矛盾。

为了解决上述挑战，我们提出了一种新的体积神经辐射场表示方法。特别是:
I)我们概括了 NeuS[Wang et al .2021]的公式，该公式具有空间变化的核宽度，对于模糊表面收敛为宽核，而对于没有额外监督的固体不透明表面则坍缩为脉冲函数。在我们的实验中，**仅这一改进就可以提高所有场景的渲染质量**。
Ii)我们使用学习到的空间变化核宽度来提取表面周围窄带的网格包络。提取的包络的宽度可以适应场景的复杂性，并作为一种有效的辅助加速数据结构。
Iii)在推断时，我们将光线投射到包络层上，以便跳过空白区域，并仅在对渲染有重要贡献的区域对辐射场进行采样。在类表面区域，窄带可以从单个样本进行渲染，而对于模糊表面则可以进行更宽的核和局部体渲染。

## Related Work

Neural Radiance Fields (NeRFs).

Implicit surface representation.
我们的方法是建立在 NeuS 公式之上，我们的主要目标不是提高提取表面的准确性。相反，我们利用 SDF 提取一个狭窄的外壳，使我们能够适应场景的局部复杂性，从而加速渲染

Accelerating neural volume rendering.
我们研究了一种加速(体积)渲染的替代方法，通过调整渲染每个像素所需的样本数量来适应场景的潜在局部复杂性。请注意，我们的公式是对“烘焙”方法的补充，我们认为两者的结合是未来研究的有趣途径。

# Method

我们的方法(见图 3)建立在 NeRF 和 NeuS 的基础上。具体来说，我们概括了 NeuS 使用新的空间变化核(章节 3.2)，提高了质量并指导窄带壳的提取(章节 3.3)。然后，在 shell 内对神经表示进行微调(第 3.5 节)，从而显著加速渲染(第 3.4 节)。

## Preliminaries

直观上，**一个小的𝑠会得到一个具有模糊密度的宽核**，而在限界 $\lim_{\mathfrak{s}\to0}d\Phi_\mathbf{s}/d\tau$ 中，则近似于一个尖锐的脉冲函数(见插图)。这种基于 sdf 的公式允许在训练期间使用 Eikonal 正则化器，这鼓励学习的𝑓成为实际的距离函数，从而产生更准确的表面重建。相关的损失将在第 3.5 节中讨论。

(NeuS 对 $\Phi_s(f)=(1+\exp(-f/s))^{-1},$ 中 s 的修改即核大小，修改密度变化 $\sigma=\max\left(-\frac{\frac{d\Phi_s}{d\tau}(f)}{\Phi_s(f)},0\right)$) [趋势](https://www.desmos.com/calculator/owxqvpotdc)
![image.png|333](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20231117160846.png)

==后面的 s 都是核大小==

## Spatially-Varying Kernel Size

NeuS SDF 公式是非常有效的，然而，它依赖于全局核大小。结合 Eikonal 正则化，这意味着在整个场景中体积密度的恒定分布。然而，**一刀切的方法并不能很好地适应**包含“尖锐”表面(如家具或汽车)和“模糊”体积区域(如头发或草)混合的场景。

我们的第一个贡献是用一个空间变化的、局部学习的核大小𝑠作为依赖于输入 3D 位置 x 的额外神经输出来增强 NeuS 公式。扩展的网络变成 $(\mathbf{c},f,s)=\mathrm{NN}_{\theta}(\mathbf{x},\mathbf{d})$ (参见第 4.1 节中的实现细节)。在训练过程中，我们还加入了一个正则化器来提高核大小 field 的平滑度(第 3.5 节)。该神经场仍然可以仅从彩色图像监督中进行拟合，并且由此产生的随空间变化的核大小会自动适应场景内容的清晰度(图 7)。这种增强的表示本身是有价值的，可以提高困难场景中的重建质量，但重要的是它将指导我们在 3.3 节中明确的 Shell 提取，从而大大加快渲染速度。

## Extracting an Explicit Shell

自适应壳划分了对渲染外观有重要影响的空间区域，并由两个显式三角形网格表示。当𝑠较大时，外壳较厚，对应于体积场景内容，当𝑠较小时，外壳较薄，对应于表面。在隐式 field 𝑠和𝑓按照 3.2 节的描述进行拟合之后，我们作为后处理提取这个自适应 shell 一次。

在方程3中 $\Phi_s(f)=(1+\exp(-f/s))^{-1},$ $\sigma=\max\left(-\frac{\frac{d\Phi_s}{d\tau}(f)}{\Phi_s(f)},0\right)$
S 形指数中数量𝑓/𝑠的大小决定了沿着一条射线的渲染贡献(参见第 3.1 节的插图)。简单地提取|𝑓/𝑠| <𝜂(对于某些𝜂)作为对呈现有重要贡献的区域是很有诱惑力的。然而，学习到的函数很快就会在远离𝑓= 0 水平集的地方变得有噪声，并且在不破坏精细细节的情况下无法充分正则化。我们的解决方案是分别提取内部边界作为𝑓= 0 水平集的侵蚀，并将外部边界作为其膨胀(图 4)，两者都通过针对任务定制的正则化约束水平集进化来实现

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20231117160902.png)

详细地说，我们首先对规则网格顶点处的 field 𝑓和𝑠进行采样。然后，我们将水平集进化应用于𝑓，产生新的侵蚀场 SDF -，并通过 marching cubes 提取 SDF - = 0 水平集作为内壳边界。一个独立的、类似的演化产生了膨胀场 SDF+，而 SDF+ = 0 的能级集形成了外壳边界。
我们分别定义这两个 level 集:
- 膨胀的外表面应该是光滑的，以避免可见的边界伪影，
- 而侵蚀的内表面不需要光滑，但必须只排除那些肯定对渲染外观没有贡献的区域。

Recall field 𝑎的基本水平集演化由 $\partial a/\partial t=-\left|\nabla a\right|v$ 给定，其中𝑣是水平集的所需标量向外法向速度。我们在𝑓上的限制的正则化流是：$\frac{\partial f}{\partial t}=-|\nabla f|\left(v(f_{0},s)+\lambda_{\mathrm{curv}}\nabla\cdot\frac{\nabla f}{|\nabla f|}\right)\omega(f),$
其中 $𝑓_{0}$ 表示初始学习的 SDF，散度项是一个权值为 $𝜆_{curv}$ 的曲率平滑正则化器。软衰减𝜔(见插图)将流量限制在水平集周围的窗口:
窗口宽度𝜁。
![image.png|333](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20231117163131.png)

为了 dilate 水平集，对于法线方向入射的射线，**选择速率v**用密度为 $\sigma>\sigma_{\min}$ 填充所有区域

$v_\text{dilate}(f_0,s)=\begin{cases}\beta_d\sigma(f_0,s)&\sigma(f_0,s)>\sigma_{\min}\\0&\sigma(f_0,s)\leq\sigma_{\min}\end{cases},$

$\beta_{d}$ 是scaling coefficient。我们使用𝜁= 0.1，$𝜆_{curv} = 0.01$。

为了侵蚀水平集，速度与密度成反比，因此在低密度区域，壳向内膨胀得快，而在高密度区域，壳向内膨胀得慢

$v_{\mathrm{erode}}(f_0,s)=\min{(v_{\mathrm{max}},\beta_e}\frac{1}{\sigma(f_0,s)}),$

这里我们使用𝜁= 0.05，$𝜆_{curv}$ = 0。These velocities导致了短距离的流动，因此形成了一个狭窄的壳，其中𝑠很小，内容物呈表面状。它们导致长距离流动，因此形成一个宽壳，其中𝑠很大，内容物呈体积状。

我们通过前向欧拉积分法在网格上对该流进行50步积分，通过空间有限差分计算导数，分别计算膨胀场SDF+和侵蚀场SDF -。我们认为没有必要进行数值上的距离调整。最后，我们将结果SDF−←max(𝑓0,SDF−)和SDF+←min(𝑓0,SDF+)夹紧，以确保侵蚀场只缩小水平集，而膨胀流只增大水平集。SDF+ = 0和SDF−= 0的水平集通过行进立方体分别作为外壳外边界网格M+和内壳边界网格M−提取。图5显示了结果字段。进一步详情载于附录的程序1及2。
