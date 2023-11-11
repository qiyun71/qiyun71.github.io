---
title: Generative approach Review
date: 2023-10-24T11:56:04.000Z
tags:
  - Diffusion
  - GAN
  - Flow
  - Review
categories: HumanBodyReconstruction
top: true
date updated: 2023-11-10T16:32:59.000Z
---

| Paper     | Model                          | Input        | Parameter/Pnum | GPU         |
| --------- | ------------------------------ | ------------ | -------------- | ----------- |
| DiT-3D    | Diffusion Transformers         | Voxelized PC |                |             |
| PointFlow | AE flow-based                  | PointCloud   | 1.61M          |             |
| FlowGAN   | GAN flow-based                 | Single Image | N = 2500       | A40 45GB    |
| BuilDiff  | Diffusion models               | Single Image | 1024 to 4096   | A40 45GB    |
| CCD-3DR   | CDPM                           | Single Image | 8192           | 3090Ti 24GB |
| SG-GAN    | SG-GAN                         | Single Image |                |             |
| **HaP**   | Diffusion+SMPL+DepthEstimation | Single Image | 10000          | 4x3090Ti    |

<!-- more -->

# Generative approach(Img2PC)

## Network Framework

![Image.png|555](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231021163349.png)

- GAN(generative adversarial networks)
- VAE(variational auto-encoders)
- Auto-regressive models
- Normalized flows(flow-based models), PointFlow
  - 相当于多个生成器，并且可逆
  - [Flow-based Generative Model - YouTube](https://www.youtube.com/watch?v=uXY18nzdSsM&list=PLJV_el3uVTsOK_ZK5L0Iv_EQoL1JefRL4&index=60)
  - [Flow-based Generative Model 笔记整理 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/267305869)
  - [Normalization Flow (标准化流) 总结 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/59615785)
- DiT(Diffusion Transformers), DiT-3D

Flow-based：
![image.png|333](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231021100952.png)
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231021101055.png)

## Loss

点云倒角距离 CD ↓
$\begin{aligned}\mathcal{L}_{CD}&=\sum_{y'\in Y'}min_{y\in Y}||y'-y||_2^2+\sum_{y\in Y}min_{y'\in Y'}||y-y'||_2^2,\end{aligned}$

推土距离 EMD (Earth Mover's distance)↓
$\mathcal{L}_{EMD}=min_{\phi:Y\rightarrow Y^{\prime}}\sum_{x\in Y}||x-\phi(x)||_{2}$ , φ indicates a parameter of bijection.

## Diffusion Models

### Human as Points(HaP)

[yztang4/HaP (github.com)](https://github.com/yztang4/HaP)
[Human as Points: Explicit Point-based 3D Human Reconstruction from Single-view RGB Images(arxiv.org)](https://arxiv.org/pdf/2311.02892.pdf)
[Human as Points—— Explicit Point-based 3D Human Reconstruction from Single-view RGB Images.pdf (readpaper.com)](https://readpaper.com/pdf-annotate/note?pdfId=2039003253707810304&noteId=2039256760006808832)

深度估计+SMPL 估计得到两个稀疏点云，输入进 Diffusion Model 进行精细化生成

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20231107193647.png)

### DiffuStereo

[DiffuStereo Project Page (liuyebin.com)](https://liuyebin.com/diffustereo/diffustereo.html)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231111120652.png)

### BuilDiff

预计 2023.11 release
[BuilDiff论文阅读笔记](/HumanBodyReconstruction/Generative%20approach/BuilDiff)
[weiyao1996/BuilDiff: BuilDiff: 3D Building Shape Generation using Single-Image Conditional Point Cloud Diffusion Models (github.com)](https://github.com/weiyao1996/BuilDiff)
[BuilDiff: 3D Building Shape Generation using Single-Image Conditional Point Cloud Diffusion Models (readpaper.com)](https://readpaper.com/pdf-annotate/note?pdfId=4796303379219349505&noteId=2014132586369911808)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231021114740.png)

### RODIN

[RODIN Diffusion (microsoft.com)](https://3d-avatar-diffusion.microsoft.com/)
[Rodin: A Generative Model for Sculpting 3D Digital Avatars Using Diffusion (readpaper.com)](https://readpaper.com/pdf-annotate/note?pdfId=4700249091733454849&noteId=2024885528733762560)

微软大数据集 + Diffusion + NeRF Tri-plane
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231028214552.png)

### CCD-3DR

[CCD-3DR: Consistent Conditioning in Diffusion for Single-Image 3D Reconstruction (readpaper.com)](https://readpaper.com/pdf-annotate/note?pdfId=4789424767274844161&noteId=2014146864821066240)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231021115606.png)

### PC^2

[PC2: Projection-Conditioned Point Cloud Diffusion for Single-Image 3D Reconstruction (lukemelas.github.io)](https://lukemelas.github.io/projection-conditioned-point-cloud-diffusion/)
[$PC^2$: Projection-Conditioned Point Cloud Diffusion for Single-Image 3D Reconstruction (readpaper.com)](https://readpaper.com/pdf-annotate/note?pdfId=4726675454702911489&noteId=2014671879272477696)

相机位姿???
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231021203845.png)

### DiT-3D

[DiT-3D论文阅读笔记](/HumanBodyReconstruction/Generative%20approach/DiT-3D)
[DiT-3D: Exploring Plain Diffusion Transformers for 3D Shape Generation](https://dit-3d.github.io/)
[DiT-3D: Exploring Plain Diffusion Transformers for 3D Shape Generation (readpaper.com)](https://readpaper.com/pdf-annotate/note?pdfId=4776143720479195137&noteId=2011558450133224704)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231019170328.png)

### Make-It-3D

[Make-It-3D: High-Fidelity 3D Creation from A Single Image with Diffusion Prior](https://make-it-3d.github.io/)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231023154817.png)

### DreamGaussian

[DreamGaussian](https://dreamgaussian.github.io/)

Gaussian Splatting + Diffusion
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231024164334.png)

### Wonder3D

[Wonder3D: Single Image to 3D using Cross-Domain Diffusion (xxlong.site)](https://www.xxlong.site/Wonder3D/)

Diffusion 一致性出图 + Geometry Fusion (novel geometric-aware optimization scheme)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231024205005.png)

#### CLIP

[openai/CLIP: CLIP (Contrastive Language-Image Pretraining), Predict the most relevant text snippet given an image (github.com)](https://github.com/openai/CLIP)

对比语言-图片预训练模型
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231025093839.png)

### GenNeRF

[Generative Neural Fields by Mixtures of Neural Implicit Functions (arxiv.org)](https://arxiv.org/abs/2310.19464)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20231031184253.png)

### LDM3D-VR

[LDM3D-VR: Latent Diffusion Model for 3D VR (arxiv.org)](https://arxiv.org/abs/2311.03226)
[视频演示T.LY URL Shortener](https://t.ly/tdi2)

从给定的文本提示生成图像和深度图数据，此外开发了一个 DepthFusion 的应用程序，它使用生成的 RGB 图像和深度图来使用 TouchDesigner 创建身临其境的交互式 360°视图体验
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20231108223206.png)

### Control3D

[Control3D: Towards Controllable Text-to-3D Generation](https://arxiv.org/pdf/2311.05461.pdf)

草图+文本条件生成 3D

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20231110160231.png)

### Etc

Colored PC [3D Colored Shape Reconstruction from a Single RGB Image through Diffusion (readpaper.com)](https://readpaper.com/pdf-annotate/note?pdfId=4723055356805120001&noteId=2014681937165010176)

## GAN

### SE-MD

[SE-MD: A Single-encoder multiple-decoder deep network for point cloud generation from 2D images. (readpaper.com)](https://readpaper.com/pdf-annotate/note?pdfId=4546363961368010753&noteId=2015501024673795072)
单编码器-->多解码器
每个解码器生成某些固定视点，然后融合所有视点来生成密集的点云

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231022105308.png)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20231031152551.png)

### FlowGAN

[FlowGAN论文阅读笔记](/HumanBodyReconstruction/Generative%20approach/FlowGAN)
[Flow-based GAN for 3D Point Cloud Generation from a Single Image (mpg.de)](https://bmvc2022.mpi-inf.mpg.de/569/)
[Flow-based GAN for 3D Point Cloud Generation from a Single Image (readpaper.com)](https://readpaper.com/pdf-annotate/note?pdfId=4677468038203719681&noteId=2011515461854903552)

![image.png|555](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231019162233.png)

### SG-GAN

[SG-GAN论文阅读笔记](/HumanBodyReconstruction/Generative%20approach/SG-GAN)
[SG-GAN: Fine Stereoscopic-Aware Generation for 3D Brain Point Cloud Up-sampling from a Single Image (readpaper.com)](https://readpaper.com/pdf-annotate/note?pdfId=4762223320204574721&noteId=2014610279544785920)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231021193539.png)

### 3D Brain Reconstruction and Complete

[3D Brain Reconstruction by Hierarchical Shape-Perception Network from a Single Incomplete Image. (readpaper.com)](https://readpaper.com/pdf-annotate/note?pdfId=4552080679232348161&noteId=2015496104888593408)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231022101838.png)

### Deep3DSketch+

[Deep3DSketch+: Rapid 3D Modeling from Single Free-Hand Sketches (readpaper.com)](https://readpaper.com/pdf-annotate/note?pdfId=4803935841323843585&noteId=2028490687877420544)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20231031092339.png)

### Reality3DSketch

[[2310.18148] Reality3DSketch: Rapid 3D Modeling of Objects from Single Freehand Sketches (arxiv.Org)](https://arxiv.org/abs/2310.18148)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20231031092703.png)

### Deep3DSketch+\\+

[[2310.18178] Deep3DSketch++: High-Fidelity 3D Modeling from Single Free-hand Sketches (arxiv.Org)](https://arxiv.org/abs/2310.18178)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20231031093421.png)

### GET3D

[nv-tlabs/GET3D (github.com)](https://github.com/nv-tlabs/GET3D)
[GET3D: A Generative Model of High Quality 3D Textured Shapes Learned from Images (nv-tlabs.github.io)](https://nv-tlabs.github.io/GET3D/)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231023154445.png)

### AG3D

[AG3D: Learning to Generate 3D Avatars from 2D Image Collections (zj-dong.github.io)](https://zj-dong.github.io/AG3D/)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231023160511.png)

## NFs (Normalizing Flows)

### PointFlow

[stevenygd/PointFlow: PointFlow : 3D Point Cloud Generation with Continuous Normalizing Flows (github.com)](https://github.com/stevenygd/PointFlow?tab=readme-ov-file)
[PointFlow: 3D Point Cloud Generation with Continuous Normalizing Flows (readpaper.com)](https://readpaper.com/pdf-annotate/note?pdfId=4500221890681004033&noteId=2011462053836873984)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231019154043.png)

## Other

### Automatic Reverse Engineering

[Automatic Reverse Engineering: Creating computer-aided design (CAD) models from multi-view images (readpaper.com)](https://readpaper.com/pdf-annotate/note?pdfId=4804297472033685505&noteId=2014396749893884416)

多视图图像生成 CAD 命令序列
局限性：

- CAD 序列的长度仍然局限于 60 个命令，因此只支持相对简单的对象
- 表示仅限于平面和圆柱表面，而许多现实世界的对象可能包括更灵活的三角形网格或样条表示

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231021160431.png)

### SimIPU

[zhyever/SimIPU: [AAAI 2021] Official Implementation of "SimIPU: Simple 2D Image and 3D Point Cloud Unsupervised Pre-Training for Spatial-Aware Visual Representations" (github. Com)](https://github.com/zhyever/SimIPU)

雷达点云+图片
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231022103332.png)

### 3DRIMR

[3DRIMR: 3D Reconstruction and Imaging via mmWave Radar based on Deep Learning. (readpaper.com)](https://readpaper.com/pdf-annotate/note?pdfId=4542232848617857025&noteId=2015491146701999104)
MmWave Radar + GAN
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231022102437.png)

### TransHuman

[TransHuman论文阅读笔记](TransHuman.md)
[TransHuman: A Transformer-based Human Representation for Generalizable Neural Human Rendering (pansanity666.github.io)](https://pansanity666.github.io/TransHuman/)

ImplicitFunction(NeRF)
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231022103225.png)

### Nvdiffrec

网格优化比 mlp 优化难，速度慢 from NeRF wechat

- 还有一个 nvdiffrcmc，效果可能好一些 Shape, Light, and Material Decomposition from Images using Monte Carlo Rendering and Denoising
- 后续还有个 NeuManifold: Neural Watertight Manifold Reconstruction with Efficient and High-Quality Rendering Support，应该比 nvdiffrec 要好

[NVlabs/nvdiffrec: Official code for the CVPR 2022 (oral) paper "Extracting Triangular 3D Models, Materials, and Lighting From Images". (github.com)](https://github.com/NVlabs/nvdiffrec)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231023154513.png)

### HyperHuman

高质量人类图片
[HyperHuman: Hyper-Realistic Human Generation with Latent Structural Diffusion (snap-research.github.io)](https://snap-research.github.io/HyperHuman/)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231023155025.png)

### SyncDreamer

[SyncDreamer: Generating Multiview-consistent Images from a Single-view Image (liuyuan-pal.github.io)](https://liuyuan-pal.github.io/SyncDreamer/)

多视图一致的图片生成
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231023165400.png)

### One-2-3-45

[One-2-3-45](https://one-2-3-45.github.io/)

MVS+NeRF
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231023165804.png)

### Zero-1-to-3

[Zero-1-to-3: Zero-shot One Image to 3D Object (columbia.edu)](https://zero123.cs.columbia.edu/)

多视图一致 Diffusion Model + NeRF

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231023171814.png)

### Pix2pix3D

[pix2pix3D: 3D-aware Conditional Image Synthesis (cmu.edu)](http://www.cs.cmu.edu/~pix2pix3D/)

一致性图像生成+NeRF

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231105173051.png)

### Consistent4D

[Consistent4D (consistent4d.github.io)](https://consistent4d.github.io/)

单目视频生成 4D 动态物体，Diffusion Model 生成多视图(时空)一致性的图像
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20231107192857.png)

### ConRad

[ConRad: Image Constrained Radiance Fields for 3D Generation from a Single Image](https://arxiv.org/pdf/2311.05230.pdf)
[ConRad: Image Constrained Radiance Fields for 3D Generation from a Single Image (readpaper.com)](https://readpaper.com/pdf-annotate/note?pdfId=2043334481936337152&noteId=2043393675895077376)

多视图一致
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20231110160631.png)

### LRM

[LRM: Large Reconstruction Model for Single Image to 3D (scalei3d.github.io)](https://scalei3d.github.io/LRM/)

大模型 Transformer(5 亿个可学习参数) + 5s 单视图生成 3D

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20231110162856.png)
