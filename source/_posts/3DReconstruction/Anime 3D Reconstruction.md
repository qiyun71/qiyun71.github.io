---
title: Anime 3D Reconstruction
date: 2024-08-22 21:06:26
tags:
  - 3DReconstruction
categories: 3DReconstruction
---

重建动漫角色模型

<!-- more -->

# Theory

> [ShuhongChen/panic3d-anime-reconstruction: CVPR 2023: PAniC-3D Stylized Single-view 3D Reconstruction from Portraits of Anime Characters](https://github.com/shuhongchen/panic3d-anime-reconstruction)

用去除线条后的2D image进行建模，得到3D radiance field

![image.png](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230806132746.png)


> [NOVA-3D: Non-overlapped Views for 3D Anime Character Reconstruction](https://wanghongsheng01.github.io/NOVA-3D/)

GAN网络，从Non-overlapped Views(Sparse-view)中重建模型

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240822174904.png)

> [CharacterGen: Efficient 3D Character Generation from Single Images](https://charactergen.github.io/)

- 多视图图像生成
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


> [CoNR：二次元视频生成AI背后的故事 - 知乎](https://zhuanlan.zhihu.com/p/565391665) **不算Reconstruction，因为最终目标不是模型，而是用模型得到的新视图图像**
> [Collaborative Neural Rendering using Anime Character Sheets (CoNR)](https://transpchan.github.io/live3d/)

- 网络1是UDP Detector，相当于给一张手绘图，可以粗略估计3D模型 (UDP检测器后接泊松表面重建算法，可以得到一个粗略灰模)。
  - 这样静态的“单张”3D灰模可以变成一个运动的“序列”。(比如，可以通过现有的自动化绑定技术（RigNet或基于模板的方法）或者其他网上的半自动绑定工具（比如Adobe Mixamo）得到一个可操作的模型，然后通过现有的基于物理的仿真技术或基于视觉的动作捕捉（如OpenPose，Posenet等）来套用各种真人视频中动作等等。)
- 网络2是CoNR，相当于从手绘人设图和3D模型上进行“转描”

特色是可以处理张数较少的手绘的二次元人物（不像3D扫描、Nerf那样需要转台和上百个视角的图像），并且输出的结果是人物在A-Pose下的灰模

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240822195911.png)


