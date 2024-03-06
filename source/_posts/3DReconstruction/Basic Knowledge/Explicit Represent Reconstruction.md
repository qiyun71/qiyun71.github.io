---
title: Explicit Represent Reconstruction
date: 2024-01-30 14:00:59
tags:
  - 
categories: 3DReconstruction/Basic Knowledge
---

显式表示：
- 点云 PointCloud
- 网格 Mesh
- 体素 Voxel
- 深度图 **[DE](/3DReconstruction/Basic%20Knowledge/Other%20Paper%20About%20Reconstruction)**

<!-- more -->

# Mesh

## Pixel2Mesh

[Pixel2Mesh (nywang16.github.io)](https://nywang16.github.io/p2m/index.html)
[Pixel2Mesh: Generating 3D Mesh Models from Single RGB Images (readpaper.com)](https://readpaper.com/pdf-annotate/note?pdfId=4518061932765929473&noteId=2082454818072408064)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20240130141333.png)

我们提出了一种端到端深度学习架构，可以从单色图像中生成三角形网格中的3D形状。受深度神经网络特性的限制，以前的方法通常是用体积或点云表示三维形状，将它们转换为更易于使用的网格模型是很困难的。与现有方法不同，我们的网络在基于图的卷积神经网络中表示3D网格，并通过逐步变形椭球来产生正确的几何形状，利用从输入图像中提取的感知特征。我们采用了从粗到精的策略，使整个变形过程稳定，并定义了各种网格相关的损失来捕捉不同层次的属性，以保证视觉上的吸引力和物理上的精确3D几何。大量的实验表明，我们的方法不仅可以定性地产生具有更好细节的网格模型，而且与目前的技术相比，可以实现更高的3D形状估计精度。


## Pixel2Mesh++ 

[Pixel2Mesh++ (walsvid.github.io)](https://walsvid.github.io/Pixel2MeshPlusPlus/)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20240204211248.png)
