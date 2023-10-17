---
title: Datasets
date: 2023-09-23 11:16:00
tags:
  - Datasets
categories: NeRF/NeRF
---

数据集

<!-- more -->

# Custom Datasets

COLMAP + Blender(neuralangelo)

## COLMAP

> [Learn-Colmap](/Learn/Learn-Colmap)

# DTU

[DTU Robot Image Data Sets | Data for Evaluating Computer Vision Methods etc.](http://roboimagedata.compute.dtu.dk/)
![image.png](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230923112720.png)
1600x1200

在黑盒空间中，使用 6 轴工业机器人手部的结构光相机，structured light scanner 可以捕获所观察场景/对象的参考 3D 表面几何形状

![image.png|555](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230923112501.png)

![image.png](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230923112539.png)

我们的机器人的**定位精度难以控制**，但重复性非常高，随机性很小。这意味着多次运行相同的定位脚本，每次定位几乎都是相同的。为了解决这个定位问题，我们不直接使用（或报告）发送给机器人的相机位置，而是确定并报告我们**获取的相对相机位置**。这是通过 MatLab 的相机校准工具箱完成的。
![image.png|555](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230923112650.png)

# BlenderMVS

[YoYo000/BlendedMVS: BlendedMVS: A Large-scale Dataset for Generalized Multi-view Stereo Networks (github.com)](https://github.com/YoYo000/BlendedMVS)
[BlendedMVS: A Large-scale Dataset for Generalized Multi-view Stereo Networks (readpaper.com)](https://readpaper.com/pdf-annotate/note?pdfId=4545062112983670785&noteId=1973517268736296192)
![image.png](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230923143752.png)
H ×W = 1536 × 2048

![image.png](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230923111934.png)

应用 3D 重建管道从精心选择的场景图像中恢复高质量的纹理网格。然后，我们将这些网格模型渲染为彩色图像和深度图。为了在训练期间引入环境照明信息，将渲染的彩色图像与输入图像进一步混合以生成训练输入

- Altizure 平台进行纹理网格重建，该软件将执行完整的 3D 重建管道并返回纹理网格和相机姿势作为最终输出
- 然后**将网格模型渲染到每个相机视图点**以生成渲染图像和渲染的深度图，**渲染的深度图**将用作 GT 深度图
- 由于渲染图像不包含与视图相关的照明，使用高通滤波器用于从渲染图像中提取图像视觉线索，而低通滤波器用于从输入中提取环境照明。最后线性混合生成混合图，**混合图**用作 GT 监督

![image.png|333](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230923112434.png)

混合图像与输入图像具有相似的背景照明，同时继承了渲染图像的纹理细节。
![image.png|555](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230923112838.png)
与 DTU 数据集[2]不同的是，所有场景都由固定的机械臂捕获，BlendedMVS 中的场景包含各种不同的摄像机轨迹。非结构化摄像机轨迹可以更好地模拟不同的图像捕获风格，并能够使网络更一般化到真实世界的重建

![image.png](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230923113119.png)

# Tanks and Temples

[Tanks and Temples Benchmark](https://www.tanksandtemples.org/)

- 工业激光扫描仪(industrial laser scanner.)捕获

# Human Body

> [rlczddl/awesome-3d-human-reconstruction (github.com)](https://github.com/rlczddl/awesome-3d-human-reconstruction#body-1)

| Name              | Link                                                                                                                                              |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2K2K              | [ketiVision/2K2K (github.com)](https://github.com/ketiVision/2K2K)                                                                                |
| ZJU-Mocap         | [neuralbody/INSTALL.md at master · zju3dv/neuralbody (github.com)](https://github.com/zju3dv/neuralbody/blob/master/INSTALL.md#zju-mocap-dataset) |
| _People-Snapshot_ | [People-Snapshot](https://github.com/zju3dv/neuralbody/blob/master/INSTALL.md#people-snapshot-dataset)                                            |
| THUman            | [THUmanDataset](https://github.com/ZhengZerong/DeepHuman/tree/master/THUmanDataset)                                                               |
| THuman2.0         | [ytrock/THuman2.0-Dataset (github.com)](https://github.com/ytrock/THuman2.0-Dataset)                                                              |
| THuman3.0         | [fwbx529/THuman3.0-Dataset (github.com)](https://github.com/fwbx529/THuman3.0-Dataset)                                                            |
| **THUman4.0**     | [ZhengZerong/THUman4.0-Dataset (github.com)](https://github.com/ZhengZerong/THUman4.0-Dataset)                                                    |
| THUman 来源       | [Yebin Liu (刘烨斌) (liuyebin.com)](http://liuyebin.com/dataset.html)                                                                             |
| renderpeople      | https://renderpeople.com/                                                                                                                         |
| MVPHuman          | [MVP-Human](https://github.com/TingtingLiao/MVPHuman)                                                                                             |
| **HuMMan**        | [caizhongang/humman_toolbox: Toolbox for HuMMan Dataset (github.com)](https://github.com/caizhongang/humman_toolbox)                              |
| CLOTH4D           | [AemikaChow/CLOTH4D (github.com)](https://github.com/AemikaChow/CLOTH4D)                                                                          |

- UltraStage：多视角和多光照条件下捕获的高质量的人体资源
    - [Relightable Neural Human Assets from Multi-view Gradient Illuminations (miaoing.github.io)](https://miaoing.github.io/RNHA/)
    - [IHe-KaiI/RNHA_Dataset: The dataset of the paper "Relightable Neural Human Assets from Multi-view Gradient Illuminations". (github.com)](https://github.com/IHe-KaiI/RNHA_Dataset)


## Human-Art

[Human-Art: A Versatile Human-Centric Dataset Bridging Natural and Artificial Scenes (idea-research.github.io)](https://idea-research.github.io/HumanArt/)

# NeuralLabeling 工具箱

> [NeuralLabeling: A versatile toolset for labeling vision datasets using Neural Radiance Fields](NeuralLabeling.md) > [NeuralLabeling: A versatile toolset for labeling vision datasets using Neural Radiance Fields (florise.github.io)](https://florise.github.io/neural_labeling_web/)

![image.png](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230925144627.png)
