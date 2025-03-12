
>  [北京科技大学研究生学位论文模板_下载中心_北京科技大学研究生院](https://gs.ustb.edu.cn/index.php/cms/item-view-id-3319.shtml)

北科提供了两种格式模板：
- 《北京科技大学研究生学位论文书写指南》.docx
- 《北京科技大学博士学位论文模板》.docx
- 《北京科技大学硕士学位论文模板》.docx
- 北京科技大学博士学位论文Latex模板.zip
- 北京科技大学硕士学位论文Latex模板.zip
  - 注：本模板要求使用者具备一定的LaTeX基础知识。安装好相关软件，例如Texlive2021以上+TexStudio（Windows），TeXShop（MAC）。
  - 推荐XeLaTex

>[在 WSL 中安装 TexLive 2023 记录 踩坑总结（openSUSE Tumbleweed）tlmgr更新 清华源-CSDN博客](https://blog.csdn.net/m0_73248035/article/details/130565440) 给wsl挂载windows字体 sudo ln -s /mnt/c/Windows/Fonts /usr/share/fonts/font
>  [WSL+Texlive+Vscode（Texstudio）完美配置 - 知乎](https://zhuanlan.zhihu.com/p/9322988213)

本地环境配置：
- 首先在本地wsl2(Ubuntu)下安装并配置texlive2024编译环境。
- 然后在win11中安装texstudio，并配置好texlive的相关路径。
- 通过texstudio进行编写latex文档，并通过在wsl2下（XeLaTex）编译pdf

可能会有查重问题，网上查询后可以直接在知网提交pdf，但texlive的版本必须在2023以上，不然会出问题(参考文献被查重)：

@- 以前 macOS 有人报告过查重问题，升级到 TeXlive 2023 或更新可解决。
https://github.com/BITNP/BIThesis/issues/326#issuecomment-1541271508
https://github.com/BITNP/BIThesis/discussions/317?sort=new#discussioncomment-5855586
如果不方便升级，但内容不敏感，可以用在线平台编译查重用的版本。
https://github.com/BITNP/BIThesis/discussions/536

---

latex格式论文 tips

```
\figref{fig:fig1s5}
```

$(\sin(2^0\pi p)$

$\eta | \mathbf{\eta} | \boldsymbol{\eta}$


|                   3D Reconstruction                    |     Single-view      |       Multi-view        |
| :----------------------------------------------------: | :------------------: | :---------------------: |
|                           特点                           | **简单但信息不足，未见区域很难重建** |   **多视图信息互补但一致性很难保证**   |
| 深度估计 **[DE](Paper%20About%203D%20Reconstruction.md)**  |      2K2K, ECON      |    MVS, MVSNet-based    |
| 隐式函数 **[IF](Paper%20About%203D%20Reconstruction.md)**  |      PIFu, ICON      |    NeuS, DoubleField    |
| 生成模型 **[GM](Generative%20Models%20Reconstruction.md)** |   BuilDIff, SG-GAN   |       DiffuStereo       |
|                      混合方法 **HM**                       |         HaP          |          DMV3D          |
|                        显式表示 ER                         |     Pixel2Mesh++     | 3DGS, SuGaR, Pixel2Mesh |
|                                                        |                      |                         |

Follow: [NeRF and Beyond日报](https://www.zhihu.com/column/c_1710703836652716032) | [nerf and beyond docs](https://github.com/yangjiheng/nerf_and_beyond_docs) | **[ventusff/neurecon](https://github.com/ventusff/neurecon)** | [Surface Reconstruction](https://paperswithcode.com/task/surface-reconstruction) | [传统3D Reconstruction](https://github.com/openMVG/awesome_3DReconstruction_list) | [Jianfei Guo](https://longtimenohack.com/) | [Nerf Tags | Yin的笔记本](http://www.yindaheng98.top/tag/Nerf/)

应用：[快手智能3D物体重建系统解析](https://mp.weixin.qq.com/s/-VU-OBpdmU0DLiEgtTFEeg) | [三维重建如今有什么很现实的应用吗？](https://www.zhihu.com/question/449185693) | [LumaAI](https://lumalabs.ai/)
资讯：[“三维AIGC与视觉大模型”十五问](https://mp.weixin.qq.com/s?__biz=MzI0MTY1NTk1Nw==&mid=2247495573&idx=1&sn=968b2d4fe20e1ab21e139f943b3cce71&chksm=e90ae66fde7d6f79cc842d9cde6b928605e3d360d17e1fdf9bde7c854058f1649a1bc45e53a7&scene=132&exptype=timeline_recommend_article_extendread_samebiz#wechat_redirect)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20231219125844.png)

**单目多视图重建--NeRF方法基本流程**

多视图三维重建，目前较好的方法是在NeuS和HashGrid方法基础上的改进。
NeRF基本流程为从相机位姿出发，得到多条从相机原点到图片像素的光线(**像素选取方法**)，在光线上进行采样得到一系列空间点(**采样方式**)，然后对采样点坐标进行编码(**编码方式**)，输入密度MLP网络进行计算(**神经网络结构**)，得到采样点位置的密度值，同时对该点的方向进行编码，输入颜色MLP网络计算得到该点的颜色值。然后根据体渲染函数沿着光线积分(**体渲染函数**)，得到像素预测的颜色值并与真实的颜色值作损失(**损失函数**)，优化MLP网络参数，最后得到一个用MLP参数隐式表达的三维模型。为了从隐式函数中提取显示模型，需要使用**MarchingCube**得到物体表面的点云和网格。

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20231219125859.png)

---

研究任务与目的：设计一套快速高精度的低成本无接触三维重建系统，用以快速地在日常生活领域生成三维模型，然后进行3D打印，满足用户定制化模型的需求

# Abstract

# Introduction+RelatedWork

## 传统的多视图三维重建方法

- 基于点云PointCloud **SFM**
- 基于网格Surface Grid
- 基于体素Voxel
- 基于深度图Depth **MVS**
  - [MVSNet: Depth Inference for Unstructured Multi-view Stereo (readpaper.com)](https://readpaper.com/pdf-annotate/note?pdfId=4518062699161739265&noteId=1986540055632613120)
  - [MVS: Multi-View Stereo based on deep learning. | Learning notes, codes and more. (github.com)](https://github.com/doubleZ0108/MVS)
  - [XYZ-qiyh/multi-view-3d-reconstruction: 📷 基于多视角图像的三维重建 (github.com)](https://github.com/XYZ-qiyh/multi-view-3d-reconstruction)

对场景显式的表征形式：
- 优点是能够对场景进行显示建模从而合成照片级的虚拟视角
- 缺点是这种离散表示因为不够精细化会造成重叠等伪影，而且最重要的，它们对内存的消耗限制了高分辨率场景的应用

## 基于NeRF的重建方法

### 基于隐式表示的三维重建方法
- [occupancy_networks: This repository contains the code for the paper "Occupancy Networks - Learning 3D Reconstruction in Function Space" (github.com)](https://github.com/autonomousvision/occupancy_networks)
- [facebookresearch/DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation (github.com)](https://github.com/facebookresearch/DeepSDF?tab=readme-ov-file)

Occupancy Networks 与 DeepSDF 依然需要显示的三维模型作监督

### 基于神经辐射场重建的三维重建方法

**NeRF被提出(2020 by UC Berkeley)**[NeRF: Neural Radiance Fields (matthewtancik.com)](https://www.matthewtancik.com/nerf)
![Network.png|666](https://raw.githubusercontent.com/yq010105/Blog_images/main/Network.png)

- 优点：隐式表示低内存、自监督方法(成本低)、单个场景进行训练可以重建任意物体=(优点or缺点)=泛化性差
- 缺点：重建速度慢、重建精度差、所需图像数量多、适用场景单一(限于有界场景、远处模糊，出现伪影)

### NeRF的不足

重建速度+重建精度
- 更快：Plenoxels、**InstantNGP**
- 更好：[UNISURF](https://github.com/autonomousvision/unisurf)、VolSDF、**NeuS**
- 快+好(InstantNGP+NeuS)：Neuralangelo、PermutoSDF、NeuS2、NeuDA、Instant-NSR、BakedSDF

重建所需图像数量（稀疏视图重建）
- SparseNeuS、NeuSurf、FORGE、FreeNeRF、ZeroRF、ColNeRF、SparseNeRF、pixelNeRF

远近细节比例不平衡（物体不在相机景深导致的模糊）
- Mip-NeRF、Mip-NeRF 360、Zip-NeRF

相机参数有误差

照片质量不好（高光、阴影、HDR|LDR）

# 中文硕博论文参考

> [基于神经隐式学习的多视图三维重建算法研究_付前程](file/基于神经隐式学习的多视图三维重建算法研究_付前程.pdf) Geo-NeuS

- 神经隐式三维重建问题分析
  - 视图合成中的神经辐射场 简单介绍NeRF
  - 基于隐式学习的三维重建 占据概率（Occupancy Probability）和符号距离场（Signed Distance Field 即 SDF）
  - 利用体渲染学习神经隐式表面 NeuS、NeuralWarp
- 几何一致的神经隐式三维重建算法 Geo-NeuS的内容
  - 颜色渲染中的偏差分析
  - 对几何场建模的显式监督
  - 基于多视图约束的几何一致性监督
- 高速的神经隐式三维重建算法 Geo-NeuS的展望，加速算法
  - 多分辨率哈希编码

DTU + BlendedMVS 数据集
分析讨论：消融分析、体积分带来的几何偏差、网络收敛速度、GPU 显存消耗、更高分辨率下进行重建、对含有噪声的 SFM 稀疏点的鲁棒性测试、视图感知的 SDF 损失、运动恢复结构中的稀疏点云讨论、表面定位中的线性插值、光度一致性约束中的灰度图像、光度一致性的其他度量

> [基于改进神经辐射场的三维重建技术研究_张志铭](file/基于改进神经辐射场的三维重建技术研究_张志铭.pdf)

- 相关概念与理论基础
  - 体渲染
  - 神经辐射场
  - 球谐函数
- 基于神经辐射场模型的少样本重建方法
  - 针对稀疏点云表征能力不足的问题：引入了用于提取深度信息的深度估计网络，**通过深度估计网络获取比稀疏点云更密集更连续的深度信息**，并用二者共同作为深度监督的来源
  - 针对少样本场景下有效采样点数量不足的问题：本章提出了利用深度信息来指导采样的过程，**通过深度信息感知物体表面的位置，使得采样点可以集中在物体表面**，提高了有效采样点的数量
  - 针对少样本场景下，仅有的少数量图像因不同拍摄角度的光照条件不同而带来的重建模糊问题，**对每张输入的图片增加了额外的嵌入向量模拟光照条件**。嵌入向量的增加为模型提供了额外的场景表示，不但能帮助模型避免过于依赖有限的训练样本导致的过拟合问题，还能提高少样本场景下对输入图像光照条件的宽容度。
- 基于神经辐射场的训练与渲染加速方法
  - 在采样方面，本章引入了结合**有效采样与关键采样**的方法，通过动态更新的采样点密度值来筛除无效采样点，从而减少了需要训练的采样点数量
  - 在渲染方面，本章设计了**一种基于双层线性表的高速缓存结构。该结构缓存了体密度和颜色信息**，使得在重建新视角时，只需查询缓存中的信息，避免了传统 NeRF 模型渲染时需要访问 MLP 神经网络获取体密度信息与颜色信息而带来的时间开销。
  - 最后，本章还使用了球谐系数来表示采样点的颜色信息，实现了渲染质量的提升。
- 基于神经辐射场的去遮挡方法 **不确定性**
  - 增加了一个用于学习暂态物体的暂态 MLP 感知机网络。暂态 MLP 网络输出的密度信息允许在不同观察视角下不同，并输出一个额外的不确定度，实现了对暂态物体的学习和筛选，减小了遮挡物对重建造成的影响

![image.png|444](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20241024101337.png)

深度损失计算时将像素点分为两类：
- 通过SFM获取深度的点
- 通过深度估计获取深度的点

![image.png|444](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20241024103120.png)

>[基于神经辐射场的三维重建算法研究_夏月](file/基于神经辐射场的三维重建算法研究_夏月.pdf) 

- 相关理论基础
  - 三维重建任务相关理论：视觉系统坐标系、像素坐标系和图像坐标系的变换、相机坐标系与图像坐标系转换、世界坐标系与相机坐标系转换
  - 神经辐射场 NeRF 相关理论：可微渲染理论、经典 NeRF 理论框架、体渲染技术、位置编码
  - 主要评价指标：PSNR、SSIM、LPIPS
- 用于复杂自然场景新视角图像重建的神经辐射场
- 神经辐射场的计算性能优化算法研究

# IDEA

- 使用先验(改进精度)
  - 分析体渲染过程中几何与颜色的偏差，使用改进的体渲染函数，提出了一个两阶段重建的框架
  - 添加先验监督：用估计的深度图和SFM获取的点监督渲染的深度图信息，用估计的法向量图来监督渲染的法向量信息
- 采样方式选择像素(改进训练速度)
  - 使用深度先验图 (其他深度估计方法) 引导图像像素选择
  - 深度先验的
- NeuS中的不确定性分析与量化
  - 通过分析量化未见视图中的不确定性，指导数据集扩充，来提高重建的质量



展望：
- 不确定性量化，可以为下一个最优视图提供指导，畅想这样的流程：
  - 用机器人拍摄多视图进行重建，并同时量化不确定性，然后对不确定性大的新视图进行重新拍摄,增加信息来指导更好的表面重建


# Old IDEA/Method

## 大论文章节

开题报告
- 三维重建算法的模块化框架构建(已有)
- 提升重建速度的像素选取方法研究
- 提高重建精度的混合编码方式研究(已有)
- 三维重建数据采集平台的搭建(设备)


- 基于神经隐式表面和体渲染的三维重建 NeuS
  - 采样方式
  - 编码方式
  - 神经网络结构
  - 体渲染函数
  - 损失函数

两种结构：
- 分步骤的(但是步骤之间关联要很小)
- 分方法改进的


## 数据采集平台搭建

Color-NeuS: 
- 三维扫描仪[EinScan Pro 2X - Shining3D Multifunctional Handheld Scanner | EinScan](https://www.einscan.com/handheld-3d-scanner/einscan-pro-2x-2020/)

DTU数据集：[Large Scale Multi-view Stereopsis Evaluation (readpaper.com)](https://readpaper.com/pdf-annotate/note?pdfId=732026274103447552&noteId=2151039906290343424)
- binary stripe encoding投影仪+相机（自制结构光相机） [DTU Robot Image Data Sets](http://roboimagedata.compute.dtu.dk/)
- 测量点云扫描精度：保龄球

BlendedMVS数据集：[YoYo000/BlendedMVS](https://github.com/YoYo000/BlendedMVS)
- Altizure.com online platform根据图片进行网格重建和位姿获取
- 根据带纹理网格生成图像数据集，目前无法评估点云[Point cloud evaluation · Issue #4](https://github.com/YoYo000/BlendedMVS/issues/4)

Tanks and Temples：[Tanks and Temples Benchmark](https://www.tanksandtemples.org/)
- 工业激光扫描仪(FARO Focus 3D X330 HDR scanner [Laser scanner](https://www.archiexpo.com/prod/faro/product-66338-1791336.html))捕获的模型
- 评估precision、recall、F-score(重建模型与GT模型)

### 方案对比

| 方案  | 设备                                             | 场景(工况)  | 成本(预估)                                                                        |
| --- | ---------------------------------------------- | ------- | ----------------------------------------------------------------------------- |
| 方案1 | 结构光相机(4+1台)，三脚架和云台(4+1套)，同步集线器+线材(1套)          | 室内场景/物体 | 总计：¥12,360<br>¥1,999/台相机(共5台)<br>¥328/套三脚架和云台(共5套)<br>¥725/套同步设备(共1套)         |
| 方案2 | 结构光相机(4+1台)，地面相机阵列支架(1套)，手持云台(1台)，同步集线器+线材(1套) | 室内场景/物体 | 总计：¥16,320<br>¥1,999/台相机<br>¥600/台手持云台<br>¥5,000/套地面相机阵列支架<br>¥725/套同步设备(共1套) |
| 方案3 | 无人机(1架)，激光扫描仪(1台)                              | 室外大场景   | 总计：¥49,246<br>¥4,788/架无人机<br>¥44458/台激光扫描仪                                    |
| 方案4 | 相机(4台)，手持扫描仪(1台)                               | 室内场景/物体 | 总计：¥57363<br>¥1,999/台相机<br>¥43,367/台手持扫描仪                                     |

三维重建数据集的构建需要采集真实物体的照片和模型，使用结构光相机就可以同时拍摄RGB图片和深度图。但要想搭建完整的数据采集系统还需要其他的一些配套设备，如固定相机的云台和支架、计算机等设备。硬件设备准备完成后，还需要根据相机型号了解配套的软件以及数据的通讯等。

方案1: 使用单个结构光相机进行单帧多视图拍摄，并使用4个结构光相机组成阵列进行多帧多视图拍摄，可以采集得到物体的照片和完整模型。该方案三脚架和云台组装简单，便捷性高且设备成本低，此外还可以在半室外的环境中进行数据采集工作。

方案2: 与方案1相比区别在于将三脚架和云台，替换为手持云台和一整套地面相机阵列支架，手持云台用于方便移动单台相机进行单帧多视图拍摄，阵列支架用于固定多相机的多帧多视图拍摄。该方案的成本略高于方案1，且支架搬运困难，便携性差。

方案3: 使用无人机拍摄室外的大场景(如建筑物)，采集照片数据，通过激光扫描仪获取大场景的模型。该方案成本高，且无人机受外部环境影响较大。

方案4: 使用相机采集物体的照片，通过手持扫描仪得到物体的模型。该方案成本高，且手持扫描仪对体积大的物体扫描很困难。


### 设备调研

#### 相机 ¥3000左右

[经典vs新锐！奥比中光Gemini2和RealSense D435i，直接采图对比！](https://www.bilibili.com/video/BV1iv4y1L7qQ?vd_source=4298530947f40edd06a04aa52d5f01d1)

[奥比中光（ORBBEC） Gemini 2 3D双目结构光深度相机-淘宝网](https://item.taobao.com/item.htm?abbucket=18&id=701234419723&ns=1&spm=a21n57.1.0.0.1c62523c9PXosc)
结构光相机：2000~5000/个，三脚架+云台200~500
[三脚架+云台的投入应该占到你的相机+镜头总投入的10%-15%](https://forum.xitek.com/thread-535226-1-1.html)

#### 三脚架+云台 ¥300左右
三脚架主要参数：高度/节数、材料

云台可分为：
- 二维云台
- 三维云台
- 球形云台

#### 同步设备

[orbbec.com/staging/wp-content/uploads/2023/08/ORBBEC_Datasheet_Multi-Camera-Sync-Hub-0816-v01.pdf](https://www.orbbec.com/staging/wp-content/uploads/2023/08/ORBBEC_Datasheet_Multi-Camera-Sync-Hub-0816-v01.pdf)
[Sync Solutions - ORBBEC - 3D Vision for a 3D World](https://www.orbbec.com/products/camera-accessories/sync-solutions/)

#### 无人机

#### 扫描仪
- [FARO Focus 3D X 330 Laser Scanner](https://frugalindustry.com/products/FARO-Focus-3D-X-330-Laser-Scanner.html)
- [Einscan Pro 2X 2020 Handheld 3D Scanner Shining3D: price in USA](https://top3dshop.com/product/shining-3d-einscan-pro-2x-3d-scanner)

# 实验

| 实验时间             |     对象      | 方法                                 | 重建时间  |
| :--------------- | :---------: | ---------------------------------- | ----- |
| @20240108-124117 | dtu114_mine | neus + HashGrid                    |       |
| @20240108-133914 | dtu114_mine | + ProgressiveBandHashGrid          |       |
| @20240108-151934 | dtu114_mine | + loss_curvature(sdf_grad_samples) |       |
|                  |             |                                    |       |
|                  |   Miku_宿舍   | neus + HashGrid                    |       |
| @20240117-164156 |   Miku_宿舍   | + ProgressiveBandHashGrid          | 47min |
|                  |             |                                    |       |
| @20240124-165842 |  TAT_Truck  | ProgressiveBandHashGrid            | 2h    |
| @20240124-230245 |  TAT_Truck  | ProgressiveBandHashGrid            |       |
| @20240125-113410 |  TAT_Truck  | ProgressiveBandHashGrid            |       |

