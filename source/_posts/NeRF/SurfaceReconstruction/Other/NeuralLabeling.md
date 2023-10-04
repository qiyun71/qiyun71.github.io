---
title: NeuralLabeling
date: 2023-09-26 14:22:46
tags:
  - Datasets
categories: NeRF/SurfaceReconstruction/Other
---

| Title     | NeuralLabeling: A versatile toolset for labeling vision datasets using Neural Radiance Fields                                                                                                                                                                                    |
| --------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Author    | Floris Erich1, Naoya Chiba2, Yusuke Yoshiyasu1, Noriaki Ando1, Ryo Hanai1, Yukiyasu Domae1                                                                                                                                                                                       |
| Conf/Jour |                                                                                                                                                                                                                                                                                  |
| Year      | 2023                                                                                                                                                                                                                                                                             |
| Project   | [NeuralLabeling: A versatile toolset for labeling vision datasets using Neural Radiance Fields (florise.github.io)](https://florise.github.io/neural_labeling_web/)                                                                                                              |
| Paper     | [2309.11966.pdf (arxiv.org)](https://arxiv.org/pdf/2309.11966.pdf) [NeuralLabeling: A versatile toolset for labeling vision datasets using Neural Radiance Fields (readpaper.com)](https://readpaper.com/pdf-annotate/note?pdfId=4802775927150870529&noteId=1976626410711696896) |

![image.png](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230925144627.png)

图像物体标记方法和工具集

<!-- more -->

# Abstract

我们提出了neurallabelling，这是一种标记方法和工具集，用于使用边界框或网格来注释场景，并生成分割蒙版、功能图、2D边界框、3D边界框、6DOF对象姿势、深度图和对象网格。neurallabelling使用神经辐射场(NeRF)作为渲染器，允许使用3D空间工具进行标记，同时结合遮挡等几何线索，仅依赖于从多个视点捕获的图像作为输入。为了证明NeuralLabeling在机器人技术中的实际问题中的适用性，我们将地面真实深度图添加到30000帧透明物体RGB中，并使用RGBD传感器捕获放置在洗碗机中的眼镜的噪声深度图，从而产生洗碗机30k数据集。我们表明，**使用带注释的深度图训练具有监督的简单深度神经网络比使用先前应用的弱监督方法训练产生更高的重建性能**。

![image.png](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230925144508.png)

支持使用3D边界框或网格进行标记
- 基于3D边界框的标签：当场景整洁和/或高质量的对象网格无法应用标签到场景时
- 基于网格的标签：当场景混乱或我们已经有对象网格可用时

# Discussion And Conclusion

我们提出了neurallabelling，这是**一种标记方法和工具集**，用于注释NeRF渲染和为下游深度学习应用生成数据集。通过使用neurallabelling，我们能够在复杂的环境中快速创建透明对象的新数据集，并使用该数据集大大提高透明对象深度补全的性能。neurallabelling的主要限制是记录场景和为每个捕获帧生成相机外部结构需要大量时间，然而这大部分是自动化的，未来可能会进一步自动化。此外，在迭代最近点可以执行之前，一个粗略的对齐是必要的。改进自动校准对于实现大型数据集的快速标记至关重要。在未来，我们计划将neurallabelling应用于更大的场景，如超市和便利店，以生成用于微调视觉语言模型的数据集。我们还计划研究如何将neurallabelling应用于动态场景，以及如何使用高质量的对象网格将对象插入到对象原本不位于的场景中

