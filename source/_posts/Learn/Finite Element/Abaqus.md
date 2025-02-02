---
title: Abaqus
date: 2024-12-16 10:03:06
tags:
  - 
categories: Learn/Finite Element
---

Abaqus Learning note

<!-- more -->

# Basic

ABAQUS/CAE 由以下功能模块构成: Part (部件)、 Property (特性)、 Assembly (装配)、 Step (分析g;- ), Interaction (相互作用)、 Load (载荷)、 Mesh (网格)、 Job (分析作业)、 Visualization (后处理〉、 Sketch (绘图)

每个 ABAQUS 模型中只能有→个装配件(assembly) ，它是由一个或多个实体( instance) 组成的，一个部件( part) 可以对应多个实体。

Interactions 的主要作用
- 定义接触：模拟两个或多个表面之间的接触行为（如滑动、分离、摩擦等）。
- 定义约束：限制模型各部分之间的相对运动（如绑定、刚体约束等）。
- 定义连接：模拟部件之间的连接行为（如螺栓、铰链、弹簧等）。
- 定义热传导：模拟热分析中的热接触或热交换行为。

Abaqus中，S、U、V、E、CF
- S (Stress) 应力 Pa
- U (Displacement)
- V (Velocity)
- E (Strain) 应变 无量纲
- CF (Concentrated Force) 集中力 用于分析节点处的力和力矩分布，常用于边界条件或载荷分析 力单位（如N）或力矩单位（如N·m

> [我的Abaqus CAE分析基础入门百科教程（结合案例讲解）_Abaqus_结构基础_静力学_非线性_通用_理论_科普-仿真秀视频课程](https://www.fangzhenxiu.com/course/6865306/?uri=728_d3eFfJ3S4bB)
> [ABAQUS/Standard 有限元软件入门指南](https://oss.jishulink.com/upload/201901/6b0bc1174afc4d8981999dadc3a37fd1.pdf#page=159.11)


动力学分析：
特征频率 
[ABAQUS Tutorial 3 : Frequency - Dynamic Harmonic loading on a cylindric fatigue specimen - YouTube](https://www.youtube.com/watch?v=7keHd1KeGjQ&list=PLockPWcLoFp9xTR-7s6bDngohn1qBb_KY) 

## 共节点

>  [ABAQUS中常用共节点的方法_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1XV4y1q7im?buvid=XXA7157A2595D8DAA7D27D13F5911BB415F26&from_spmid=search.search-result.0.0&is_story_h5=false&mid=5E%2FE0HONObjFbvgpVZnCxw%3D%3D&plat_id=116&share_from=ugc&share_medium=android&share_plat=android&share_session_id=d115e833-5b12-450f-b711-51f199326232&share_source=WEIXIN&share_tag=s_i&spmid=united.player-video-detail.0.0&timestamp=1734506574&unique_k=ERTboHq&up_id=373921637&vd_source=1dba7493016a36a32b27a14ed2891088)

通过几何连接关系使网格共节点(基于构造几何)
- merge 合并实体， 选择solid

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20241218160111.png)


- partition 分割实体(原来是一整个实体，需要分为多个部分，每个部分赋予不同的材料属性)

通过合并网格节点使网格共节点（基于网格）
- merge 节点, 选择 mesh

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20241218160111.png)

mesh node 编辑 merge，但是需要保证网格没有与几何连接(使用Delete Mesh Associativity)

# 动力学分析

## 频响函数

>  [ABAQUS直接法频率响应分析的正常流程——像建模这种事情吧一旦开始就很难结束尤其是前边留下问题的时候_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1Ka4y1p7c5/?vd_source=1dba7493016a36a32b27a14ed2891088) 
>  [ABAQUS模态法频率响应分析——我也只是个成长中的小菜鸟，一起进步吧同志们_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV14a411c7kg?spm_id_from=333.788.recommend_more_video.-1&vd_source=1dba7493016a36a32b27a14ed2891088)
>  [压电悬臂梁振动能量收集仿真与试验验证<sup>*</sup>](https://html.rhhz.net/BJHKHTDXXBZRB/20170626.htm) 2+3维悬臂梁模态频率和直接法FRFs

![bjhkhtdxxb-43-6-1271-5.jpg (700×510)](https://html.rhhz.net/BJHKHTDXXBZRB/PIC/bjhkhtdxxb-43-6-1271-5.jpg)


# 模态分析

>  [结构模态分析详解](https://www.tup.com.cn/upload/books/yz/082156-01.pdf)


# Python

>  [abqpy 2025.7.10.dev9+g0175c7325 documentation](https://hailin.wang/abqpy/en/dev/)