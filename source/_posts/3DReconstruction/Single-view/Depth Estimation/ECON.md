---
title: ECON
date: 2023-09-30 17:27:17
tags:
  - ClothedHumans
  - DepthEstimation
categories: 3DReconstruction/Single-view/Depth Estimation
---

| Title     | ECON: Explicit Clothed humans Obtained from Normals                                                                                                                 |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Author    | Yuliang Xiu1 Jinlong Yang1 Xu Cao2 Dimitrios Tzionas3 Michael J. Black1                                                                                             |
| Conf/Jour | CVPR                                                                                                                                                                |
| Year      | 2023                                                                                                                                                                |
| Project   | [ECON: Explicit Clothed humans Optimized via Normal integration (xiuyuliang.cn)](https://xiuyuliang.cn/econ/)                                                       |
| Paper     | [ECON: Explicit Clothed humans Obtained from Normals (readpaper.com)](https://readpaper.com/pdf-annotate/note?pdfId=4700954831381069826&noteId=1983981033620573952) |

![image.png](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230930173026.png)

姿态稳定(**ICON在难的姿势下较好地重建**)+灵活拓扑(**ECON还可以较好地重建宽松的衣服**)

缺陷：
![image.png](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230930204752.png)

<!-- more -->

# Abstract

艺术家策展的扫描与深层隐式函数（IF）的结合，使得可以从图像中创建详细的、穿着衣物的3D人物成为可能。然而，现有方法远非完美。基于IF的方法可以恢复自由形式的几何形状，但在看不见的姿势或服装下会产生脱离身体的肢体或退化的形状。为了增加这些情况的稳健性，现有的工作使用显式参数化身体模型来限制表面重建，但这限制了自由形式表面（如与身体不符的宽松服装）的恢复。我们想要的是一种结合了隐式和显式方法的最佳特性的方法。为此，我们提出了两个关键观察点：（1）当前的网络在推断详细的2D maps方面表现更好，而不是完整的3D表面，以及（2）参数化模型可以被看作是将详细的表面片段拼接在一起的“画布”。ECON即使在宽松的服装和具有挑战性的姿势下也可以推断出高保真度的3D人物，同时具有逼真的面部和手指。这超越了以前的方法。对CAPE和Renderpeople数据集的定量评估表明，ECON比现有技术更精确。感知研究还表明，ECON的感知逼真度明显更高。

# Method

## Detailed normal map prediction

$\mathcal{L}_{\mathrm{SMPL-X}}=\mathcal{L}_{\mathrm{N.diff}}+\mathcal{L}_{\mathrm{S.diff}}+\mathcal{L}_{\mathrm{J.diff}},$ 
- 在ICON基础上添加了(2D body landmarks)二维地标间的关节损失(L2): $\mathcal{L}_\mathrm{J\_diff}=\lambda_\mathrm{J\_diff}|\mathcal{J}^\mathrm{b}-\widehat{\mathcal{J}^\mathrm{c}}|,$

## Front and back surface reconstruction

将覆盖的法线贴图提升到2.5D表面。我们期望这些2.5D表面满足三个条件:
(1)高频表面细节与预测的覆盖法线图一致;
(2)低频表面变化(包括不连续面)与SMPL-X的一致;
(3)前后轮廓的深度彼此接近。

> [xucao-42/bilateral_normal_integration: Official implementation of "Bilateral Normal Integration" (BiNI), ECCV 2022. (github.com)](https://github.com/xucao-42/bilateral_normal_integration)

利用bilateral normal integration (BiNI)方法，利用**粗糙先验、深度图和轮廓一致性**进行全身网格重建。
本文提出了一种深度感知轮廓一致的双边法向积分(d-BiNI)方法
$\mathrm{d-BiNI}(\widehat{\mathcal{N}}_{\mathrm{F}}^{\mathrm{c}},\widehat{\mathcal{N}}_{\mathrm{B}}^{\mathrm{c}},\mathcal{Z}_{\mathrm{F}}^{\mathrm{b}},\mathcal{Z}_{\mathrm{B}}^{\mathrm{b}})\to\widehat{\mathcal{Z}}_{\mathrm{F}}^{\mathrm{c}},\widehat{\mathcal{Z}}_{\mathrm{B}}^{\mathrm{c}}.$

优化方法：
![image.png](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230930184708.png)
- $\mathcal{L}_{\mathrm{n}}$ 由BiNI引入的前后BiNI项
- $\mathcal{L}_{\mathrm{d}}$ 前后深度prior项 ，$\mathcal{L}_{\mathrm{d}}(\widehat{\cal Z}_{i}^{\mathrm{c}};\mathcal{Z}_{i}^{\mathrm{b}})=|\widehat{\cal Z}_{i}^{\mathrm{c}}-\mathcal{Z}_{i}^{\mathrm{b}}|\quad i\in\{F,B\}.$
- $\mathcal{L}_{\mathrm{s}}$ 前后轮廓一致性项，$\mathcal{L}_{\mathrm{s}}(\widehat{\mathcal{Z}_{\mathrm{F}}^{\mathrm{c}}},\widehat{\mathcal{Z}_{\mathrm{B}}^{\mathrm{c}}})=|\widehat{\mathcal{Z}_{\mathrm{F}}^{\mathrm{c}}}-\widehat{\mathcal{Z}_{\mathrm{B}}^{\mathrm{c}}}|_{\mathrm{silhouette}}.$

## Human shape completion

sPSR(Screened poisson surface reconstruction) completion with SMPL-X (ECONEX).
在SMPL-X的mesh中将前后摄像头可以看到的三角形网格移除，留下的三角形soup包括侧面和遮挡区域，将sPSR应用到soup和d-BiNI曲面$\{\mathcal{M}_{\mathrm{F}},\mathcal{M}_{\mathrm{B}}\}$的并集，得到一个水密重建。*(这种方法称为ECONEX。虽然ECONEX避免了四肢或侧面的缺失，但由于SMPL-X与实际的衣服或头发之间的差异，它不能为原来缺失的衣服和头发表面产生连贯的表面;见图4中的ECONEX)*

![image.png](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230930203744.png)

Inpainting with IF-Nets+ (RIF)
为了提高重建一致性，我们使用学习的隐式函数(IF)模型来“补绘”给定的前后d-BiNI表面缺失的几何形状
IF-Nets+以体素化的前、后地真深度图$\{\mathcal{Z}_{\mathrm{F}}^{\mathfrak{c}},\mathcal{Z}_{\mathrm{B}}^{\mathfrak{c}}\}$和体素化(估计)的身体网格$\mathcal{M}^{\mathrm{b}}$作为输入进行训练，并以地真3D形状进行监督

sPSR completion with SMPL-X and RIF (ECONIF).
为了获得最终的网格R，我们应用sPSR来缝合
(1)d-BiNI表面，
(2)来自Rif的侧面和闭塞的三角形汤纹，
(3)从估计的SMPL-X体裁剪的脸或手

- 虽然RIF已经是一个完整的人体网格，但我们只使用它的侧面和遮挡部分，因为与d-BiNI表面相比，它的正面和背面区域缺乏清晰的细节
- 此外，我们使用从$\mathcal{M}^{\mathrm{b}}$裁剪的脸部或手，因为这些部分在RIF中通常重建得很差

IF-Nets+ ：
![image.png](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231001100118.png)

# Experiments

Training on THuman2.0. and 我们使用 THuman2.0 来训练 ICON 变体，IF-Nets+、IF-Nets、PIFu 和 PaMIR。

# Discussion

Limitations 
ECON 将 RGB 图像和估计的 SMPL-X 身体作为输入。然而，从单个图像中恢复 SMPL-X 身体（或类似的模型）仍然是一个悬而未决的问题，不能完全解决。这中的任何故障都会导致 ECON 故障，如图 10-A 和图 10-B 所示。ECON的重建质量主要依赖于预测法线图的准确性。如图 10-C 和图 10-D 所示，糟糕的法线贴图可能会导致过于接近甚至相交的前表面和后表面。
![image.png](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230930204752.png)

Future work.
除了解决上述限制之外，其他几个方向对于实际应用很有用。目前，ECON只重建3D几何图形。还可以恢复底层骨架和蒙皮权重，例如，使用SSDR[40]，以获得完全动画的化身。此外，推断反向视图纹理将导致完全纹理的化身。从恢复的几何图形中解开服装、头发或配件将使这些样式的合成、编辑和转移成为可能。最后，ECON 的重建可用作训练神经化身的伪地面实况 [16, 19, 30]。

# Conclusion

我们提出了 ECON，一种从彩色图像重建详细的穿着衣服 3D 人体的方法。ECON结合了显式参数模型和深度隐函数的优点;它估计人体和服装的详细3D表面，而不局限于特定的拓扑，同时对具有挑战性的看不见的姿势和服装具有鲁棒性。为此，它采用了**变分正态积分**和**形状补全**的最新进展，并有效地将这些扩展到从彩色图像重建人体的任务。我们相信这项工作可以导致 3D 视觉社区的实际应用和有用的数据增强，因此，我们发布了我们的模型和代码

