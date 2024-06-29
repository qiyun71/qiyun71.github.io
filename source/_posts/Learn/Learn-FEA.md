---
title: Learn-FEA
date: 2024-05-25 22:10:37
tags:
  - 
categories: Learn
---

Basic of Finite Element Analysis

<!-- more -->

# Finite Element


> [有限元讨论班大纲 - 哔哩哔哩](https://www.bilibili.com/read/cv15083732/?spm_id_from=333.788.0.0)

有限元分析（Finite Element Analysis，FEA）有时又被称作有限元方法（Finite Element Method，FEM），二者基本可以混用，微小的区别在于，FEM 侧重于整套理论而 FEA 侧重于应用 FEM 于具体工程问题。

## Basic

> [有限元入门--Understanding the Finite Element Method | The Efficient Engineer的部分笔记 - 知乎](https://zhuanlan.zhihu.com/p/681621600)
> [网格划分：六面体和四面体该怎么选择？ - 知乎](https://zhuanlan.zhihu.com/p/348729395) 六面体计算收敛速度快，四面体计算收敛慢。
> [有限元分析（FEA）是个什么东东 - 知乎](https://zhuanlan.zhihu.com/p/56326567) **有限元法就是为了求解偏微分方程！** **获得微分方程的“弱形式”解**
>     [有限元法（FEM）详解](https://cn.comsol.com/multiphysics/finite-element-method) 
> [(72 封私信 / 80 条消息) 数值pde与深度学习结合是未来发展方向吗？ - 知乎](https://www.zhihu.com/question/523893840) PINN [​内嵌物理知识神经网络（PINN）是个坑吗？ - 知乎](https://zhuanlan.zhihu.com/p/468748367)

有限元模型：
- 节点nodes和元素/单元elements的集合称为有限元网格(finite element mesh)
  - 元素elements：线元素line elements用于建模杆等一维结构、表面元素surface elements用于模拟薄壳等薄表面、实体元素solid elements用于建模三维实体，如四面体Tet，六面体Hex
  - **复杂的几何体通常只能使用tri三角形和tet四面体元素进行网格划分。四边形quad和六面体hex元素最适合规则几何图形，在规则几何图形中，它们优于三角形和三角形元素，因为它们效率更高且需要的节点更少**
  - 元素可以是线性的（也称为一阶元素，linear or first-order elements）或二次的（二阶元素，quadratic or second-order elements）。**二次元素在元素的每一侧都有附加的中间节点**。它们需要更多的计算能力，但通常比线性元素产生更准确的结果。线性四面体元素（例如**TET4**，一种4节点四面体元素）可能表现出过度僵硬的行为，应避免使用它们-通常应使用**TET10**二次四面体元素
- 自由度：有限元网格中的每个节点都有一定数量的自由度，*在二维应力分析中，每个节点有3个自由度——在X轴和Y轴上平移，以及绕Z轴旋转。对于热分析，每个节点都有一个自由度，即节点温度。*









# 有限元分析软件

## Solidworks

将Solid 转换成Surface：
- **Delete Face**
- Offset

> [How To Convert Solid To Surface Body In SolidWorks - YouTube](https://www.youtube.com/watch?v=Fx0jX_7aJHM)

## Patran

Preferences --> Geometry 单位制 inches/m/mm，不同的单位设置材料特性时会有差异

eg(钢板): E = 210GPa，$\rho = 2700Kg/m^{3}$
mm制：
- Elastic Modulus/Shear Modulus ：(210000)MPa
- Poisson Ratio：注意设置了E和G后，Poisson Ratio会自动计算(根据钢板算例测试出来的)
- Density：(2.7E-9)$Kgf \cdot s^{2} /mm^{4} = (Kg/mm^{3},1Kgf = 9.8N)$  

[patran中数据的输入输出单位 - 百度文库](https://wenku.baidu.com/view/9f2572f37c1cfad6195fa7d6.html?fr=income1-doc-search&_wkts_=1714641483975&wkQuery=patran%E4%B8%AD%E6%95%B0%E6%8D%AE%E7%9A%84%E8%BE%93%E5%85%A5%E8%BE%93%E5%87%BA%E5%8D%95%E4%BD%8D)


## Nastran

[如何用matlab被nastran给整的明明白白 PART 1 KNOW YOUR ENEMY——.bdf文件 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/33538970)

Nastran的Python库：[Welcome to pyNastran’s documentation for v1.3! — pyNastran 1.3 1.3 documentation (pynastran-git.readthedocs.io)](https://pynastran-git.readthedocs.io/en/1.3/index.html)

Debug: [Nastran Error List 1. | PDF](https://www.scribd.com/doc/70652924/Nastran-Error-List-1)

### 卫星算例.bdf

不同结构参数生成结构特征量FR

```bdf file
$ Elements and Element Properties for region : Shear_Panels
PSHELL   1       1      .003     1               1

- 36  行 .003 Shear_Panels 厚度 theta5
- 429 行 .002 Central_Cylinder 厚度 theta3
- 666 行 .001 Adapter 厚度 theta2 本来应该是密度2.7
- 723 行 .002 Upper_platform 厚度 theta6
- 864 行 .001 Lower_platform 厚度 theta4
- 1020行 7.   mat_N 弹性模量  theta1  
- 1023行 7.   mat_CC
- 1026行 7.   mat_L
- 1029行 7.   mat_SP
- 1032行 7.   mat_U
- 
```

- **主弹性模量**$\theta_1$ 70Gpa，
- **主密度** $\theta_2$  ，密度2.7x $10^{3} kg/m^{3}$ (英文论文) or 适配器厚度 1mm(本 1)
- **中心筒厚度**$\theta_3$ 2mm
- 底板厚度 $\theta_4$ 1mm
- **剪切板厚度**$\theta_5$ 2mm
- 顶板厚度 $\theta_6$ 2.5mm