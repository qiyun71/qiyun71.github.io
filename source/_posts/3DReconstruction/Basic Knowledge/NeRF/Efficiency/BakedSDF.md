---
title: BakedSDF
date: 2023-09-13 15:44:04
tags:
  - Efficiency
  - Real-time
categories: 3DReconstruction/Basic Knowledge/NeRF/Efficiency
---

| Title     | BakedSDF: Meshing Neural SDFs for Real-Time View Synthesis                                                                                                                 |
| --------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Author    | Lior Yariv and Peter Hedman and Christian Reiser and Dor Verbin and Pratul P. Srinivasan and Richard Szeliski and	Jonathan T. Barron and Ben Mildenhall                                                                                                                                                                           |
| Conf/Jour | SIGGRAPH                                                                                                                                                                           |
| Year      | 2023                                                                                                                                                                           |
| Project   | [BakedSDF](https://bakedsdf.github.io/)                                                                                                                                    |
| Paper     | [BakedSDF: Meshing Neural SDFs for Real-Time View Synthesis (readpaper.com)](https://readpaper.com/pdf-annotate/note?pdfId=4762044324229677057&noteId=1959289127671095296) |

![image.png](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230913154905.png)

- 对前景物体采用类似VolSDF的方法训练*Modeling density with an SDF*
- 使用Marching Cube 的方法来提取高分辨率网格*Baking a high-resolution mesh*
- *Modeling view-dependent appearance*，对baked后的高分辨率网格上顶点：采用漫反射颜色和球形高斯叶（前景3个波瓣，远处背景使用1个波瓣）
    - $\mathbf{C}=\mathbf{c}_{d}+\sum_{i=1}^{N}\mathbf{c}_{i}\exp\left(\lambda_{i}\left(\mu_{i}\cdot\mathbf{d}-1\right)\right).$ 

<!-- more -->

# Limitations

尽管我们的模型在实时渲染无界场景的任务中达到了最先进的速度和准确性，但仍有一些限制，这些限制代表了未来改进的机会:
- 我们使用完全不透明的网格表示来表示场景，**因此我们的模型可能难以表示半透明内容**(玻璃，雾等)。与基于网格的方法一样，我们的模型**有时无法准确地表示具有小或详细几何形状**(茂密的树叶，薄结构等)的区域。图6描绘了额外提取的网格可视化，展示了我们的表面重建限制及其对渲染重建的影响。**这些问题也许可以通过增加网格的不透明度值来解决，但是允许连续的不透明度将需要一个复杂的多边形排序过程，这很难集成到实时光栅化管道中**。
- 我们技术的另一个限制是，我们模型的输出网格**占用了大量的磁盘空间**(每个场景约430兆字节)，这可能对某些应用程序的存储或流式传输具有挑战性。这可以通过网格简化和UV绘图来改善。**然而，我们发现现有的简化和绘图工具主要是为艺术家制作的3D资产而设计的，对于我们通过移动立方体提取的网格并不适用**。

![image.png](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230913204555.png)

# Conclusion

我们提出了一个系统，产生一个高质量的网格实时渲染的大型无界现实世界场景。我们的技术首先优化了用于精确表面重建的场景的混合神经体-表面表示。从这种混合表示中，我们提取了一个三角形网格，其顶点包含与视图相关的外观的有效表示，然后优化这个网格表示以最佳地再现捕获的输入图像。这使得网格在速度和准确性方面都能产生最先进的实时视图合成结果，并且具有足够高的质量以支持下游应用。

# AIR

我们提出了一种适合于真实感**新视图**合成的大型**无界现实场景**的**高质量网格**重建方法。我们首先优化了一个**混合神经体-表面场景表示**，它被设计成具有与场景中的表面对应的行为良好的水平集。然后，我们将这种表示烘烤成一个高质量的三角形网格，我们配备了一个简单而快速的基于球面高斯的视图依赖外观模型。最后，我们优化了这种烘焙表示，以最好地再现捕获的视点，从而产生一个可以利用加速多边形光栅化管道在消费级硬件上进行实时视图合成的模型。我们的方法在准确性、速度和功耗方面优于以前的实时渲染场景表示，并产生高质量的网格，使外观编辑和物理模拟等应用成为可能。

Key Words：
- Reconstruction; Neural networks; Volumetric models.
- Neural Radiance Fields, Signed Distance Function, Surface Reconstruction, Image Synthesis, Real-Time Rendering, Deep Learning.

## Introduction

目前用于新视图合成(使用捕获的图像恢复从未观察到的视点渲染的3D表示)的最佳性能方法主要基于神经辐射场(NeRF) [Mildenhall等人，2020]。通过将场景表示为由多层感知器(MLP)参数化的连续体积函数，NeRF能够生成逼真的效果图，展示详细的几何图形和依赖于视图的效果。**由于计算NeRF下的MLP成本很高，并且每个像素必须查询数百次，因此从NeRF渲染高分辨率图像通常很慢**。

最近的工作通过将计算繁重的mlp替换为离散的体积表示(如体素网格)，提高了NeRF渲染性能。然而，这些方法需要大量的GPU内存和自定义的体积射线行进代码，并且不适合在商用硬件上进行实时渲染，因为现代图形硬件和软件面向的是呈现多边形表面而不是体积场。

虽然目前类似nerf的方法能够恢复具有简单几何形状的单个对象的高质量实时可渲染网格[Boss等人，2022]，但从现实世界无界场景的捕获(例如Barron等人的“360度捕获”)中重建详细且行为良好的网格被证明更加困难。最近，MobileNeRF [Chen et al. 2022a]通过训练NeRF来解决这个问题，该NeRF的体积内容被限制在多边形网格的表面上，然后将NeRF烘焙成纹理图。虽然这种方法产生了合理的图像质量，但MobileNeRF将场景几何初始化为一组axis-aligned tiles，优化后会变成一个纹理多边形“汤soup”。生成的几何图形不太适合常见的图形应用程序，如纹理编辑、重光照和物理模拟。

**在这项工作中，我们演示了如何从类似nerf的神经体积表示中提取高质量的网格**。我们的系统，我们称之为BakedSDF，扩展了VolSDF的混合体面神经表示[Yariv等人，2021]，以表示无界的现实世界场景。**这种表示被设计成具有与场景中的表面相对应的良好表现的零水平集，这让我们可以使用行进立方体提取高分辨率的三角形网格**。

我们的关键思想是在收缩contracted坐标空间中定义SDF [Barron et al. 2022]，因为它具有以下优点:它更强地正则化远距离内容，并且它还允许我们在收缩空间中提取网格，从而更好地分配三角形预算(中心多，外围少)。

然后，我们为这个网格配备了一个基于球面高斯的快速有效的视图依赖外观模型，该模型经过微调以再现场景的输入图像。我们系统的输出可以在商品设备上以实时帧速率呈现，并且我们表明，我们的实时渲染系统在真实感，速度和功耗方面优于先前的工作。此外，我们表明(不像以前的工作)，我们的模型产生的网格是准确和详细的，使标准的图形应用程序，如外观编辑和物理模拟。

总而言之，我们的主要贡献是:
- 无界现实场景的高质量神经表面重建;
- 一个在浏览器中实时渲染这些场景的框架，以及
- 我们证明了球面高斯函数是视图合成中视图依赖外观的实际表示(practical representation)。

## RELATED WORK

View synthesis, i.e.，在给定一组捕获的图像的情况下渲染场景的新视图，是计算机视觉和图形学领域的一个长期存在的问题。
- 在观测视点采样密集的场景中，可以通过光场渲染来合成新的视点——直接插值到观测光线集合中
- 然而，在实际环境中，观察到的视点被捕获得更稀疏，重建场景的3D表示对于呈现令人信服的新视图至关重要。大多数经典的视图合成方法使用三角形网格(通常使用由多视图立体MVS组成的管道重建。泊松地表重建，以及Marching Cube作为底层3D场景表示，并通过将观察到的图像重新投影到每个新视点中并使用启发式定义或learned 混合权重将它们混合在一起来呈现新视图。
- 尽管基于网格的表示非常适合使用加速图形管道进行实时渲染，但这些方法产生的网格在具有精细细节或复杂材料的区域中往往具有不准确的几何形状，从而导致渲染新视图时出现错误。
- Alternatively，基于点的表示更适合于建模薄几何，但如果没有可见的裂缝或相机移动时不稳定的结果，则无法有效地渲染。

最近的视图合成方法通过使用几何和外观的体积表示(如体素网格)来回避高质量网格重建的困难或多平面图像。这些表示非常适合基于梯度的渲染损失优化，因此它们可以有效地优化以重建输入图像中看到的详细几何形状。这些体积方法中最成功的是神经辐射场(NeRF) ，它构成了许多最先进的视图合成方法的基础(参见Tewari等人[2022]进行综述)。NeRF将场景表示为发射和吸收光的连续的物质体积场，并使用体积光线跟踪渲染图像。NeRF使用MLP从空间坐标映射到体积密度和发射亮度，并且MLP必须沿着射线在一组采样坐标上进行评估，以产生最终的颜色。

随后的工作建议修改NeRF的场景几何和外观的表示，以提高质量和可编辑性。
- Ref-NeRF [Verbin et al. 2022]重新参数化NeRF的视图依赖外观，以实现外观编辑并改进镜面材料的重建和渲染。
- 其他作品[Boss et al. 2021;Kuang等。2022;Srinivasan等人。2021;Zhang等。2021a,b]尝试将场景的视图依赖外观分解为材料和照明属性。
- 除了修改NeRF的外观表示外，包括UNISURF [Oechsle等人]在内的论文。[2021]， VolSDF [Yariv等。2021]，neus [Wang等。2021]，MetaNLR++和NeuMesh用混合体面模型增强NeRF的全体积表示，**但不以实时渲染为目标，只显示对象和有界场景的结果**

用于表示场景的MLP NeRF通常是大型且昂贵的评估，这意味着NeRF的训练速度很慢(每个场景数小时或数天)，渲染速度也很慢(每百万像素数秒或数分钟)。虽然可以通过减少每条光线的MLP查询的采样网络来加速渲染，最近的方法通过用体素网格取代大型MLP来改善训练和渲染时间，小型mlp网格， low-rank或sparse grid表示，或者使用小MLP进行多尺度哈希编码。

虽然这些表示减少了训练和渲染所需的计算(以增加存储为代价)，但渲染可以通过预计算和存储进一步加速，即“烘烤”，训练好的NeRF变为更有效的表示。FastNeRF ，Plenoctrees 和可扩展神经室内场景渲染(Scalable Neural Indoor Scene Rendering)[Wu et al. 2022]都将训练过的nerf烤成稀疏的体积结构，并使用简化的视图依赖外观模型，以避免沿着每条光线对每个样本进行MLP评估。这些方法可以在高端硬件上实现nerf的实时渲染，**但它们对体积射线推进的使用妨碍了普通硬件的实时性能**。在我们工作的同时，开发了内存高效辐射场(MERF)，这是一种用于无界场景的压缩表示体积，有助于在商用硬件上快速渲染。与我们的网格相比，这种体积表示达到了更高的质量分数，**但需要更多的内存，需要一个复杂的渲染器，并且不直接用于下游图形应用程序，如物理模拟**。请参考MERF论文与我们的方法进行直接比较。

# PRELIMINARIES

在本节中，我们描述了神经容量表示，NeRF 用于视图合成以及mip-NeRF 360引入的改进，因为它代表了unbounded “360度”场景。

NeRF的渲染方程与loss函数
$\mathbf{C}=\sum_{i}\exp\left(-\sum_{j<i}\tau_{j}\delta_{j}\right)\left(1-\exp\left(-\tau_{i}\delta_{i}\right)\right)\mathbf{c}_{i},\quad\delta_{i}=t_{i}-t_{i-1}.$ Eq.1

$\mathcal{L}_{\mathrm{data}}=\mathbb{E}\left[\left\|\mathbf{C}-\mathbf{C}_{\mathrm{gt}}\right\|^{2}\right].$ Eq.2

Mip-NeRF 360
将无界的x位置contract到有界域
$\operatorname{contract}(\mathbf{x})=\begin{cases}\mathbf{x}&\|\mathbf{x}\|\leq1\\\left(2-\frac{1}{\|\mathbf{x}\|}\right)\frac{\mathbf{x}}{\|\mathbf{x}\|}&\|\mathbf{x}\|>1\end{cases},$ Eq.3

# METHOD

我们的方法由三个阶段组成，如图2所示。
- 首先，我们使用类似nerf的体渲染优化基于表面的几何形状表示和场景外观。
- 然后，我们将几何形状“烘烤”到一个网格中，我们证明它足够精确，可以支持令人信服的外观编辑和物理模拟。
- 最后，我们训练了一个新的外观模型，该模型使用球面高斯(SGs)嵌入到网格的每个顶点中，取代了第一步中昂贵的类似nerf的外观模型。

这种方法产生的3D表示可以在商品设备上实时渲染，因为渲染只需要对网格进行栅格化并查询少量球面高斯函数。

![image.png](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230913164451.png)

## Modeling density with an SDF

我们的表示结合了mip-NeRF 360表示无界场景的优点，以及VolSDF混合体-面表示的良好表面属性[Yariv等人。2021]。VolSDF将场景的体积密度建模为mlp参数化有符号距离函数(SDF)的参数函数f，它返回带符号的距离$f(\mathbf{x})$从每一点$\mathbf{x}\in\mathbb{R}^3$到曲面。因为我们的重点是重建无界的现实世界场景，我们参数化f在收缩空间(公式3)而不是世界空间。场景的下表面是f的零水平集，即距离曲面为零的点的集合:
$\mathcal{S}=\{\mathbf{x}:f(\mathbf{x})=0\}.$ Eq.4
与VolSDF一样，将体密度定义为：
$\tau(\mathbf{x})=\alpha\Psi_{\beta}\left(f(\mathbf{x})\right),$ Eq.5
其中$\Psi_{\beta}$是带尺度参数$\beta>0$的零均值拉普拉斯分布的累积分布函数。注意随着$\beta$趋近于0，体积密度接近于一个函数，它返回$\alpha$在任意对象内，自由空间为0。鼓励f近似一个有效的带符号距离函数(即一个f(x)返回对所有x的f水平集的有符号欧氏距离)。，我们惩罚f的偏差满足Eikonal方程：
$\mathcal{L}_{\mathrm{SDF}}=\mathbb{E}_{\mathrm{x}}\left[(\|\nabla f(\mathrm{x})\|-1)^{2}\right].$ Eq.6
注意f在收缩空间中定义，这个约束也作用于收缩空间。

最近，Ref-NeRF [Verbin et al. 2022]通过将其参数化为反映表面法线的视图方向的函数来改进视图依赖外观。我们使用sdf参数化的密度使得这很容易被采用因为sdf具有定义良好的表面法线:$\mathbf{n}(\mathbf{x})=\nabla f(\mathbf{x})/\|\nabla\hat{f}(\mathbf{x})\|.$ 因此，在训练我们模型的这一阶段时，我们采用Ref-NeRF的外观模型，并使用单独的漫射和镜面分量来计算颜色，其中镜面分量是由法线方向反射的视图方向的拼接、法线方向与视图方向之间的点积以及MLP输出的256个元素的bottleneck向量来参数化的。

我们使用mip-NeRF 360的变体作为我们的模型(具体训练细节见补充材料中的附录a)。与VolSDF [Yariv et al. 2021]类似，我们将密度比例因子参数化为$\alpha=\beta^{-1}$式5。然而，我们find调度$\beta$而不是把它作为一个自由的可优化参数，结果是更稳定的训练。我们因此退火$\beta$根据$\beta_{0}\left(1+\frac{\beta_{0}-\beta_{1}}{\beta_{1}}t^{0.8}\right)^{-1},$其中$t$在训练过程中从0到1，$\beta_{0}=0.1$，和$\beta_{1}$三个分层采样阶段分别为0.015、0.003和0.001。由于密度SDF参数化所需的Eikonal正则化已经消除了floaters体并导致了良好的常态，我们发现没有必要使用Ref-NeRF的orientation损失或预测法向量，或mip - nerf 360的distortion损失。

## Baking a high-resolution mesh

优化神经体积表示后，我们通过在常规3D网格上查询恢复的mlp参数化SDF创建三角形网格，然后运行Marching Cubes [Lorensen and Cline 1987]。请注意，VolSDF使用超出SDF零交叉点的密度下降来建模边界(参数化为$\beta$)。我们在提取网格时考虑到这种扩散，并选择0.001作为表面交叉的等值，否则我们会发现场景几何形状会被轻微侵蚀。

Visibility and free-space culling.当运行Marching Cubes时，MLP参数化的SDF可能在被观测视点遮挡的区域以及建议MLP标记为“自由空间”的区域中包含虚假的表面交叉。在训练期间，这两种类型区域中的SDF MLP值都不受监督，因此我们必须剔除任何可能在重建网格中显示为虚假内容的表面交叉点。为了解决这个问题，我们检查沿着我们的训练数据中的射线拍摄的3D样本。我们计算每个样本的体积渲染权重，即它对训练像素颜色的贡献。然后，我们将任何具有足够大的渲染权重(> 0.005)的样本放入3D网格中，并将相应的单元标记为表面提取的候选单元。

Mesh extraction.**我们在收缩空间中以均匀间隔的坐标对SDF网格进行采样**，从而在世界空间中产生不均匀间隔的非轴向坐标。**这具有为靠近原点的前景内容创建较小三角形(在世界空间中)和为较远的内容创建较大三角形的理想属性**。有效地，我们利用收缩算子作为细节级策略:因为我们想要渲染的视图接近场景原点，并且因为收缩的形状被设计为撤销透视投影的效果，所以所有三角形在投影到图像平面上时将具有大约相等的面积。
Region growing.在提取三角形网格后，我们使用区域生长过程来填充可能存在于输入视点未观察到或在烘烤过程中被提议MLP遗漏的区域中的小洞。我们在当前网格周围的邻域中迭代标记体素，并提取这些新活动体素中存在的任何表面交叉点。**这种区域增长策略有效地解决了SDF MLP中存在曲面但由于训练视图覆盖不足或提议MLP中存在错误而无法通过行进立方体提取的情况**。然后我们将网格转换为世界空间，这样它就可以通过在欧几里德空间中操作的传统渲染引擎进行光栅化。
Implementation.我们使用$2048^{3}$网格进行可见性和自由空间筛选和移动立方体。最初，我们只在未被剔除的体素上运行行进立方体，即可见且非空的体素。然后我们用32个区域增长迭代来完成网格，在那里我们在当前网格中的顶点周围的$8^{3}$个体素附近重新运行行进立方体。最后，我们使用顶点顺序优化对网格进行后处理[Sander等2007]，**它通过允许顶点着色器输出在相邻三角形之间缓存和重用来加快现代硬件上的渲染性能**。在附录B中，我们详细介绍了网格提取的其他步骤，这些步骤并不严格提高重建精度，但可以提供更令人愉悦的交互式观看体验。

## Modeling view-dependent appearance

上面描述的烘焙过程从我们基于mlp的场景表示中提取高质量的三角形网格几何。为了模拟场景的外观，包括与视图相关的效果，如镜面，我们为每个网格顶点配备漫射颜色$c_{d}$和一组球形高斯叶。由于遥远的区域只能从有限的一组视图方向观察到，我们不需要在场景中以相同的保真度对视图依赖性进行建模。在我们的实验中，我们在中心区域$(\|\mathbf{x}\|\leq1)$使用三个球形高斯波瓣，在外围区域使用一个波瓣。图3展示了我们的外观分解。

![image.png](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230913202957.png)

这种外观表示满足我们对计算和内存的效率目标，因此可以实时呈现。每个球面高斯叶有七个参数:一个三维单位向量$\mu$对于瓣的均值，一个三维矢量c表示瓣的颜色，和一个标量$\lambda$对于叶的宽度。这些叶是由视图方向向量d参数化的，所以一条射线与任何给定顶点相交的渲染颜色C可以计算为:
$\mathbf{C}=\mathbf{c}_{d}+\sum_{i=1}^{N}\mathbf{c}_{i}\exp\left(\lambda_{i}\left(\mu_{i}\cdot\mathbf{d}-1\right)\right).$ Eq.7

为了优化这种表示，我们首先将网格栅格化到所有训练视图中，并存储与每个像素相关的顶点索引和质心坐标。在此预处理之后，我们可以通过将重心插值应用于学习到的每个顶点参数，然后运行我们的视图依赖外观模型(模拟fragment着色器的操作)来轻松渲染像素。因此，我们可以通过最小化每个像素的颜色损失来优化每个顶点的参数，如公式2所示。如附录B所述，我们还优化了背景清晰的颜色，以便为交互式查看器提供更愉悦的体验。为了防止这种优化被网格几何形状没有很好建模的像素(例如，软对象边界和半透明对象的像素)所影响，我们使用鲁棒损失$\rho(\cdot,\alpha,c)$来代替VolSDF最小化的L2损失。鲁棒损失超参数$\alpha=0,c={}^{1}/5$，这使得优化对离群值更具鲁棒性[Barron 2019]。我们还使用直通估计器对量化建模[Bengio等人]。[2013]，确保8位精度很好地表示视图相关外观的优化值。
我们发现直接优化这种逐顶点表示会使GPU内存饱和，这阻碍了我们扩展到高分辨率网格。我们转而优化了一个基于Instant NGP的压缩神经哈希网格模型。在优化过程中，我们在训练批内的每个3D顶点位置查询该模型，以产生我们的漫反射颜色和球面高斯参数。
优化完成后，我们通过在每个顶点位置查询NGP模型以获得与外观相关的参数，烘烤出哈希网格中包含的压缩场景表示。最后，我们使用gLTF格式[ISO/IEC 12113:2022 2022]导出生成的网格和逐顶点外观参数，并使用gzip压缩它，这是web协议原生支持的格式。

# EXPERIMENTS

我们通过输出效果图的准确性以及速度、能量energy和内存需求来评估方法的性能。
对于准确性，我们测试了两个版本的模型: 第4.1节中描述的中间体绘制结果，我们称之为“离线”模型，以及第4.2节和4.3节中描述的烘焙实时模型，我们称之为“实时”模型。作为基线，我们使用先前的离线模型[Barron等人，2022;米尔登霍尔等人。2020;Müller等。2022;Riegler和Koltun 2021;Zhang等人。2020]designed for fidelity,，以及之前的实时方法[Chen等人。2022年;海德曼等人。2018]designed for performance. 
我们还将我们的方法恢复的网格与COLMAP [Schönberger等人]，mip-NeRF 360 [Barron等。2022]，和MobileNeRF [Chen等。2022年]提取的网格进行了比较。所有FPS(帧/秒)测量都是在1920 × 1080分辨率下渲染的。

## Real-time rendering of unbounded scenes

## Mesh extraction

## Appearance model ablation


# 实验(非官方code)

## 环境配置

```bash
pip install torch torchvision
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install -r requirements.txt
```

## 运行

colmap生成pose
```bash
python scripts/imgs2poses.py ./load/bmvs_dog # images are in ./load/bmvs_dog/images
``` 

run:
```bash
python launch.py --config configs/neus-colmap.yaml --gpu 0 --train dataset.root_dir=$1
python launch.py --config configs/bakedsdf-colmap.yaml --gpu 0 --train dataset.root_dir=$1 --resume_weights_only --resume latest
```


