---
title: Anime Image 3D Reconstruction
date: 2024-08-22 21:06:26
tags:
  - 3DReconstruction
categories: 3DReconstruction
---

重建动漫角色模型

> [CharacterGen: Efficient 3D Character Generation from Single Images](https://charactergen.github.io/)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240825220242.png)


<!-- more -->

# 目前的生产方式(手办/模型)

> [女武神脉冲 葵来栖 建模过程](https://www.bilibili.com/video/BV1SJ411t7TK/?spm_id_from=333.999.0.0&vd_source=1dba7493016a36a32b27a14ed2891088)

建立数字化3D模型(mesh)-->**模型拆件**
- 软件：Blender, ZBrush

## 3D打印(小批量，未来趋势)

> [如何把3d模型变成实物 葵来栖手办打磨篇_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1AJ411a7q4/?spm_id_from=333.788.recommend_more_video.4&vd_source=1dba7493016a36a32b27a14ed2891088)

使用3D打印工艺，进行模型拆件要除了便于组装外，还要有一定厚度(3D打印的模型必须要有足够的厚度)，可以融合使用一些其他的组装件如铜柱等，拆件完后，在3D打印机软件中做支撑，然后进行打印(光固化树脂)

**整体流程**：
建模-->拆件-->(做支撑)-->3D打印-->打磨-->组装-->上色(涂装)

1. 建模师根据2D图像进行建模(需要大量经验) ==> 多视图三维重建，且根据2D虚拟图像进行重建才有意义，如stable diffusion等文生图AI生成出来的图像(可以做到多视图)，或者画师手工绘制的图像(单视图较为方便)。*需要保证做出来的模型具有一定的稳定性，不能一碰就倒，要找好重心*
2. 拆件也需要一定的经验，哪里做凹陷，哪里连接用铜柱等等问题需要考虑
3. 做支撑可以防止零件被悬空打印
4. 3D打印有很多种方式**FDM常见成本低**、**SLA成本高但精度高**、SLS、SLM。未来趋势全彩3D打印机：[2023年5款最佳全彩3D打印机](https://www.bilibili.com/read/cv25588645/)

展望：
- **使用全彩3D打印机一体化地打印模型，省去了拆件、组装、上色等步骤**，通过改进还可以省去做支撑(或者自动做支撑)
- 使用**三维重建**算法可以更高效地获取三维模型，三维重建算法的最高形态是可以根据物体几张图片甚至一张图片，重建出高质量的模型。图片来源可以是画师手绘的一张图、随便拍的真实物体照片、网站上的各种物体图片等等

> [赛纳三维 J402PLUS](https://www.sailner.com/web/product/41/detail) SLA 立体光固化打印机
> [SLA/DLP/LCD 3D 打印支撑添加全解](https://www.chitubox.com/zh-Hans/academy/advanced/a-complete-guide-about-adding-supports-in-sladlplcd-3d-printing)

## 钢模+注塑(大批量)

> [手办制作工厂大揭秘!~来看自己的老婆是如何诞生的!!!](https://www.bilibili.com/video/BV1eJ411E7Et/?spm_id_from=333.337.search-card.all.click&vd_source=1dba7493016a36a32b27a14ed2891088)

使用钢模+注塑的话，拆件

**整体流程**：
建模-->拆件-->开模(做钢模)-->批量注塑-->筛选-->打磨-->上色(手工+喷涂)-->分件分类+组装-->除尘+打包

# Theory

## 手办图像生成

> [[PVC Style Model]Movable figure model Pony - Pony1.60 | Stable Diffusion Checkpoint | Civitai](https://civitai.com/models/400329?modelVersionId=675843) Stable Diffusion PVC model

![image.png|222](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240824144446.png) 

选择PVC，外挂VAE选择None

- Prompt: 
  - (score_9, score_8_up, score_7_up), source anime, figure
  - surreal,amazing quality,masterpiece,best quality,awesome,inspiring,cinematic composition,soft shadows,Film grain,shallow depth of field,highly detailed,high budget,cinemascope,epic,color graded cinematic,atmospheric lighting,natural,figure,natural lighting,exqusite visual effect,delicate details
- Negative: 
  - engrish text, low quality, worst quality, score_4,score_3,score_2,score_1,ugly,bad feet, bad hands
  - lowres,(bad),text,error,fewer,extra,missing,worst quality,jpeg artifacts,low quality,watermark,unfinished,displeasing,oldest,early,chromatic aberration,artistic error,username,english text,scan,[abstract],

```
1.更倾向于PVC的起始标签（Starting tags that lean more towards PVC）：
figure,natural lighting,exqusite visual effect,delicate details,fully body

Negative prompt: 
NSFW, cleavage, breasts, bad image, worst dquality

2.更倾向于2.5d的标签（Tags that lean more towards 2.5d）：

surreal,amazing quality,masterpiece,best quality,awesome,inspiring,cinematic composition,soft shadows,Film grain,shallow depth of field,highly detailed,high budget,cinemascope,epic,color graded cinematic,atmospheric lighting,natural,figure,natural lighting,exqusite visual effect,delicate details

Negative prompt：
lowres,(bad),text,error,fewer,extra,missing,worst quality,jpeg artifacts,low quality,watermark,unfinished,displeasing,oldest,early,chromatic aberration,artistic error,username,english text,scan,[abstract],

Suggest drawing parameters
sampler:

Euler a: 30 steps

hires.fix: 4x-ultrasharp ,

High score iteration steps 10

Redraw amplitude 0.3

cfg scale: 7.0  

Resolution of portrait characters： 768x1280 ，1024x1024

clip skip: 2

Local map generation can be enabled：ADetailer
```

ControlNet: 
- Unit0 Canny， Pixel Perfect，Allow Preview，Canny，预处理器选canny，My prompt is more important
- Unit1 tilecolorfix， Pixel Perfect，Allow Preview，tilecolorfix，预处理器选tilecolorfix，My prompt is more important

难点：
- 现有的图生图模型无法根据手绘图片合理的(保留原始风格)生成手办(PVC)风格图片
- 必须要有针对Full Body的强控制 

![image.png|444](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240824201522.png)


## 背景移除

> [Bilateral Reference for High-Resolution Dichotomous Image Segmentation | PDF](https://arxiv.org/pdf/2401.03407)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240823152932.png)

> [(SAM) facebookresearch/segment-anything: The repository provides code for running inference with the SegmentAnything Model](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file)
> [(SAM 2) facebookresearch/segment-anything-2: The repository provides code for running inference with the Meta Segment Anything Model 2](https://github.com/facebookresearch/segment-anything-2?tab=readme-ov-file)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240824160034.png)


## 额外先验估计

### Normal Image

> [Stable-X/StableNormal: StableNormal: Reducing Diffusion Variance for Stable and Sharp Normal](https://github.com/Stable-X/StableNormal)

### Depth Image

> [Shakes on a Plane: Unsupervised Depth Estimation from Unstabilized Photography – Princeton Computing Imaging Lab](https://light.princeton.edu/publication/soap/) 手持微抖的图像序列


> [Fine-Tuning Image-Conditional Diffusion Models is Easier than You Think](https://gonzalomartingarcia.github.io/diffusion-e2e-ft/)扩散模型


## 一致性多视图生成

> [VideoMV](https://aigc3d.github.io/VideoMV/)

与训练的视频生成模型+位姿(微调)生成多视图图像

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240823202814.png)


>[EscherNet: A Generative Model for Scalable View Synthesis](https://kxhit.github.io/EscherNet)

它可以在单个消费级 GPU 上同时生成 100 多个一致的目标视图，以任何相机姿势的任意数量的参考视图为条件。

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240914150918.png)

> [Cycle3D: High-quality and Consistent Image-to-3D Generation via Generation-Reconstruction Cycle](https://pku-yuangroup.github.io/Cycle3D/)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240920110142.png)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240920110225.png)

## 多视图三维重建

NeRF-based 的 NeuS 系列
3DGS-based 的 显示高斯体优化

# Related Work

some github repo:

>[emanuelevivoli/aswesome-comics-understanding: The official repo of the Comics Survey: "A missing piece in Vision and Language: A Survey on Comics Understanding"](https://github.com/emanuelevivoli/awesome-comics-understanding)


## Dataset

> [kangyeolk/AnimeCeleb: Official implementation of "AnimeCeleb: Large-Scale Animation CelebHeads Dataset for Head Reenactment" (ECCV 2022)](https://github.com/kangyeolk/AnimeCeleb?tab=readme-ov-file)

![raw.githubusercontent.com/kangyeolk/AnimeCeleb/main/assets/teaser.png](https://raw.githubusercontent.com/kangyeolk/AnimeCeleb/main/assets/teaser.png)


## PAniC-3D

> [ShuhongChen/panic3d-anime-reconstruction: CVPR 2023: PAniC-3D Stylized Single-view 3D Reconstruction from Portraits of Anime Characters](https://github.com/shuhongchen/panic3d-anime-reconstruction)

用去除线条后的2D image进行建模，得到3D radiance field

![image.png](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230806132746.png)

## NOVA-3D 

> [NOVA-3D: Non-overlapped Views for 3D Anime Character Reconstruction](https://wanghongsheng01.github.io/NOVA-3D/)

GAN网络，从Non-overlapped Views(Sparse-view)中重建模型

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240822174904.png)

## CharacterGen

> [CharacterGen: Efficient 3D Character Generation from Single Images](https://charactergen.github.io/)

- 多视图图像生成，由于特殊姿势的一致性多视图很难，本文提出了将输入的图像转换到canonical space中
- 三维重建：基于NeRF和Transformer参考了[LRM: Large Reconstruction Model for Single Image to 3D](https://yiconghong.me/LRM/)
  - 在Objaverse dataset上pre-train，然后再本文数据集Anime3D上Fine-tune，为了引入更多的人体先验
  - 为了得到跟好的模型，使用 Triplane **SDF** 替换密度场
- 网络提取和UV map获取，参考了[Deep Marching Tetrahedra: a Hybrid Representation for High-Resolution 3D Shape Synthesis](https://research.nvidia.com/labs/toronto-ai/DMTet/)，但是DMTet的UV展开过程中会损失外观信息。此外，UV的分辨率比直接渲染的分辨率大，多个texels可能投影到相同的像素上，为了解决这一问题，本文将四个视图投影到UV map上(*通过depth test消除occluded texels*没理解)
  - 单纯的叠加还不行(会导致角色身体轮廓出现噪声texels)，本文使用相机方向向量与normal texture map作内积，内积大于-0.2的texels抛弃掉
  - 对于overlapping的texels，选择back投影后RGB最接近coarse texture的值
  - 最后使用Poisson Blending将projected texels与原始texels混合，以减少seams
- 使用了[Modular Primitives for High-Performance Differentiable Rendering | PDF](https://arxiv.org/pdf/2011.03277) NvDiffRast渲染器进行高效光栅化

texels(纹理元素，纹素)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240822202233.png)
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240822202244.png)


## CoNR

> [CoNR：二次元视频生成AI背后的故事 - 知乎](https://zhuanlan.zhihu.com/p/565391665) **不算Reconstruction，因为最终目标不是模型，而是用模型得到的新视图图像**
> [Collaborative Neural Rendering using Anime Character Sheets (CoNR)](https://transpchan.github.io/live3d/)

- 网络1是UDP Detector，相当于给一张手绘图，可以粗略估计3D模型 (UDP检测器后接泊松表面重建算法，可以得到一个粗略灰模)。
  - 这样静态的“单张”3D灰模可以变成一个运动的“序列”。(比如，可以通过现有的自动化绑定技术（RigNet或基于模板的方法）或者其他网上的半自动绑定工具（比如Adobe Mixamo）得到一个可操作的模型，然后通过现有的基于物理的仿真技术或基于视觉的动作捕捉（如OpenPose，Posenet等）来套用各种真人视频中动作等等。)
- 网络2是CoNR，相当于从手绘人设图和3D模型上进行“转描”

特色是可以处理张数较少的手绘的二次元人物（不像3D扫描、Nerf那样需要转台和上百个视角的图像），并且输出的结果是人物在A-Pose下的灰模

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240822195911.png)


## Toon3D

> [Toon3D](https://toon3d.studio/)

动漫中的环境场景，动漫中的手绘图像没有3D一致性，本文致力解决这一问题

![teaser.png (1828×464)](https://toon3d.studio/static/images/teaser.png)

数据集生成
- Marigold 进行深度估计
- SAM 进行mask获取
- 自制的Toon3D Labeler进行

## Sketch-A-Shape

>[Sketch-A-Shape: Zero-Shot Sketch-to-3D Shape Generation](https://arxiv.org/pdf/2307.03869)

从草图中生成3D shape (across voxel, implicit, and CAD representations and synthesize consistent 3D shapes)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20241011110424.png)

## ShapeFromSketches

[3D Shape Reconstruction from Sketches via Multi-view Convolutional Networks:](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8374559&tag=1)

很简单的思路，就是直接输入sketches 然后使用卷积构建单个编码器和两个解码器，生成多个视图的depth 和 normal maps，然后出点云

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20241011141941.png)
