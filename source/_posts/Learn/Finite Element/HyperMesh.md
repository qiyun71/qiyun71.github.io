---
title: HyperMesh
date: 2025-01-11 22:50:07
tags:
  - 
categories: Learn/Finite Element
---

网格划分
[new-hyperworks-experience](https://web.altair.com/zh-cn/new-hyperworks-experience)

<!-- more -->

# 官方教程

> [Altair Course: Getting Started with HyperMesh v2024](https://learn.altair.com/course/view.php?id=668)

Based on the Finite Element Method (FEM), analysts (engineers and physicists) make use of FEA and FEM software to predict the behavior of varied types of physical laws in real-life scenarios with precision, versatility, and practicality.

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20250111231533.png)


# Others

>  [HyperMesh网格划分简要流程小试_hypermesh划分网格-CSDN博客](https://blog.csdn.net/qq_39784672/article/details/127437426)

**拓扑检查和修复**

![4bf5f5e6b3860f9f14e1864db3245c5e.png (1026×362)|666](https://i-blog.csdnimg.cn/blog_migrate/4bf5f5e6b3860f9f14e1864db3245c5e.png#pic_center)

![2de1c1e3308500a3169e2a6a98009e1b.png (878×392)|666](https://i-blog.csdnimg.cn/blog_migrate/2de1c1e3308500a3169e2a6a98009e1b.png#pic_center)

**网格划分** --> 网格质量检查 --> 网格文件导出


# 

[六面体网格划分_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1zi42117wc/?spm_id_from=333.1387.collection.video_card.click&vd_source=1dba7493016a36a32b27a14ed2891088)

15:00中网格划分方法：
- 先2d panel mesh将圆柱(轴)底面网格与孔的网格对齐
- 然后3d hex 路径引导划分圆柱上的网格
- 然后需要共节点：validate中的equivalence 
- 检查face，隐藏几何，隐藏表面的mesh，观察内部网格质量

[hypermesh2023_四面体网格三种画法总结_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1Pm421M7XS/?vd_source=1dba7493016a36a32b27a14ed2891088)

