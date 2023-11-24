---
title: CVPR 2023
date: 2023-09-16 09:48:45
tags:
  - Review
categories: 3DReconstruction/Basic Knowledge/NeRF/Review
---
ref:
[awesome-NeRF-papers/NeRFs-CVPR2023.md at main · lif314/awesome-NeRF-papers (github.com)](https://github.com/lif314/awesome-NeRF-papers/blob/main/NeRFs-CVPR2023.md)
[awesome-NeRF-papers/NeRFs-ICCV2023.md at main · lif314/awesome-NeRF-papers (github.com)](https://github.com/lif314/awesome-NeRF-papers/blob/main/NeRFs-ICCV2023.md)
[工作簿: CVPR 2023 Subject Areas by Team Size (tableau.com)](https://public.tableau.com/views/CVPR2023SubjectAreasbyTeamSize/Dashboard2a?:showVizHome=no)
[yangjiheng/nerf_and_beyond_docs (github.com)](https://github.com/yangjiheng/nerf_and_beyond_docs)

<!-- more -->

反射 | 低光 | 阴影
稀疏视图
编码方式
去模糊anti-alias
**无位姿情况**
快速渲染
节省内存

应用：
- 物体探测
- **分割**
- RGBD实时跟踪
- 机器人感知
- Human：人体，人脸，人手
- 其他

扩散模型
3D风格迁移
text to 3D
可编辑
动态场景

[AutoRecon: Automated 3D Object Discovery and Reconstruction (zju3dv.github.io)](https://zju3dv.github.io/autorecon/)

# 反射 | 低光 | 阴影
## 光照反射
[https://github.com/hirotong/ReNeuS](https://github.com/hirotong/ReNeuS)3D重建,
NeFII: Inverse Rendering for Reflectance Decomposition with Near-Field Indirect Illumination光照反射渲染
[https://ktiwary2.github.io/objectsascam/](https://ktiwary2.github.io/objectsascam/)辐射场相机,物体反射
[MS-NeRF (zx-yin.github.io)](https://zx-yin.github.io/msnerf/)穿过镜像物体的复杂光路，高质量场景渲染
[JiaxiongQ/NeuS-HSR: Looking Through the Glass: Neural Surface Reconstruction Against High Specular Reflections (CVPR 2023) (github.com)](https://github.com/JiaxiongQ/NeuS-HSR) 高镜面反射

## 阴影射线监督
[https://gerwang.github.io/shadowneus/](https://gerwang.github.io/shadowneus/)

## Low Light低光场景
[https://www.whyy.site/paper/llnerf](https://www.whyy.site/paper/llnerf)

## 水下或雾气弥漫场景
[SeaThru-NeRF: Neural Radiance Fields in Scattering Media (sea-thru-nerf.github.io)](https://sea-thru-nerf.github.io/)

## 光流监督
[https://mightychaos.github.io/projects/fsdnerf/](https://mightychaos.github.io/projects/fsdnerf/)

# 稀疏视图
[https://scade-spacecarving-nerfs.github.io/](https://scade-spacecarving-nerfs.github.io/)(深度监督)
[https://github.com/google-research/nerf-from-image](https://github.com/google-research/nerf-from-image)(单视图)
[https://flex-nerf.github.io/](https://flex-nerf.github.io/)(人体建模)
[http://prunetruong.com/sparf.github.io/](http://prunetruong.com/sparf.github.io/)(位姿不准)
[https://jiawei-yang.github.io/FreeNeRF/](https://jiawei-yang.github.io/FreeNeRF/)稀疏视图
[https://github.com/ShuhongChen/panic3d-anime-reconstruction](https://github.com/ShuhongChen/panic3d-anime-reconstruction)单视图3D重建
MixNeRF: Modeling a Ray with Mixture Density for Novel View Synthesis from Sparse Inputs稀疏视图,深度监督
NeRFInvertor: High Fidelity NeRF-GAN Inversion for Single-shot Real Image Animation
[https://sparsefusion.github.io/](https://sparsefusion.github.io/)扩散模型
[https://sparsenerf.github.io/](https://sparsenerf.github.io/)depth-based, Few-shot
[Behind the Scenes: Density Fields for Single View Reconstruction | fwmb.github.io](https://fwmb.github.io/bts/)
[Zero-1-to-3: Zero-shot One Image to 3D Object (columbia.edu)](https://zero123.cs.columbia.edu/)
[[2306.00965] BUOL: A Bottom-Up Framework with Occupancy-aware Lifting for Panoptic 3D Scene Reconstruction From A Single Image (arxiv.org)](https://arxiv.org/abs/2306.00965)

# 编码方式
[https://github.com/KostadinovShalon/exact-nerf](https://github.com/KostadinovShalon/exact-nerf)
[https://3d-front-future.github.io/neuda/](https://3d-front-future.github.io/neuda/)保真表面重建

# 去模糊
[https://github.com/TangZJ/able-nerf](https://github.com/TangZJ/able-nerf)(自注意力)
[https://dogyoonlee.github.io/dpnerf/](https://dogyoonlee.github.io/dpnerf/)(物理场景先验)
[https://github.com/BoifZ/VDN-NeRF](https://github.com/BoifZ/VDN-NeRF)
[https://jonbarron.info/zipnerf/](https://jonbarron.info/zipnerf/)Anti-Aliased, Grid-Based
[https://wbhu.github.io/projects/Tri-MipRF/](https://wbhu.github.io/projects/Tri-MipRF/)Anti-Aliasing, Faster
[CVPR 2023 Open Access Repository (thecvf.com)](https://openaccess.thecvf.com/content/CVPR2023/html/Isaac-Medina_Exact-NeRF_An_Exploration_of_a_Precise_Volumetric_Parameterization_for_Neural_CVPR_2023_paper.html)

# 没有位姿 NeRF without pose
[https://rust-paper.github.io/](https://rust-paper.github.io/)
[https://nope-nerf.active.vision/](https://nope-nerf.active.vision/)
[https://henry123-boy.github.io/level-s2fm/](https://henry123-boy.github.io/level-s2fm/)增量重建,无位姿SfM

大规模街景
[https://city-super.github.io/gridnerf/](https://city-super.github.io/gridnerf/)
[https://dnmp.github.io/](https://dnmp.github.io/)Urban Reconstruction
[https://astra-vision.github.io/SceneRF/](https://astra-vision.github.io/SceneRF/)Urban Reconstruction

联合估计位姿，增量重建
[https://localrf.github.io/](https://localrf.github.io/)

Bundle-Adjusting
[https://rover-xingyu.github.io/L2G-NeRF/](https://rover-xingyu.github.io/L2G-NeRF/)
[https://aibluefisher.github.io/dbarf/](https://aibluefisher.github.io/dbarf/)
[https://github.com/WU-CVGL/BAD-NeRF](https://github.com/WU-CVGL/BAD-NeRF)去模糊

可泛化
[https://xhuangcv.github.io/lirf/](https://xhuangcv.github.io/lirf/)

点云渲染
[https://arxiv.org/pdf/2303.16482.pdf](https://arxiv.org/pdf/2303.16482.pdf)
[https://jkulhanek.com/tetra-nerf/](https://jkulhanek.com/tetra-nerf/)Point-Based, Tetrahedra-Based[ALTO: Alternating Latent Topologies for Implicit 3D Reconstruction - Visual Machines Group (ucla.edu)](https://visual.ee.ucla.edu/alto.htm/)交替潜在拓扑，从嘈杂的点云中高保真地重建隐式3D表面

逆渲染
[https://haian-jin.github.io/TensoIR/](https://haian-jin.github.io/TensoIR/)

逼真合成
[https://redrock303.github.io/nerflix/](https://redrock303.github.io/nerflix/)
ContraNeRF: Generalizable Neural Radiance Fields for Synthetic-to-real Novel View Synthesis via Contrastive Learning真实渲染
[https://robustnerf.github.io/public/](https://robustnerf.github.io/public/)真实渲染，场景包含分散物体，**去除floaters**
[AligNeRF (yifanjiang19.github.io)](https://yifanjiang19.github.io/alignerf)高保真度、输入高分辨率图像和有校准误差的相机

3D边缘重建
[https://yunfan1202.github.io/NEF/](https://yunfan1202.github.io/NEF/)

3D扫描中重建2D楼层平面图
[Connecting the Dots: Floorplan Reconstruction Using Two-Level Queries (ywyue.github.io)](https://ywyue.github.io/RoomFormer/)

nerf2mesh, nerf-texture
[https://me.kiui.moe/nerf2mesh/](https://me.kiui.moe/nerf2mesh/)

高质量重建SDF Based Reconstruction / Other Geometry Based Reconstruction
[https://xmeng525.github.io/xiaoxumeng.github.io/projects/cvpr23_neat](https://xmeng525.github.io/xiaoxumeng.github.io/projects/cvpr23_neat)
[yiqun-wang/PET-NeuS: PET-NeuS: Positional Encoding Tri-Planes for Neural Surfaces (CVPR 2023) (github.com)](https://github.com/yiqun-wang/PET-NeuS)
[NeuManifold (sarahweiii.github.io)](https://sarahweiii.github.io/neumanifold/)
[NeuS2: Fast Learning of Neural Implicit Surfaces for Multi-view Reconstruction (mpg.de)](https://vcai.mpi-inf.mpg.de/projects/NeuS2/)


立体深度
[https://nerfstereo.github.io/](https://nerfstereo.github.io/)

视图合成
Enhanced Stable View Synthesis(室外)

事件相机
[https://4dqv.mpi-inf.mpg.de/EventNeRF/](https://4dqv.mpi-inf.mpg.de/EventNeRF/)

3D GAN,生成辐射场
[https://www.computationalimaging.org/publications/singraf/](https://www.computationalimaging.org/publications/singraf/)SinGRAF：学习单个场景的3D生成辐射场

# 快速渲染
[https://mobile-nerf.github.io/](https://mobile-nerf.github.io/)(移动设备)
[https://snap-research.github.io/MobileR2L/](https://snap-research.github.io/MobileR2L/)
[https://totoro97.github.io/projects/f2-nerf/](https://totoro97.github.io/projects/f2-nerf/)(任意相机路径,快速)
[https://radualexandru.github.io/permuto_sdf/](https://radualexandru.github.io/permuto_sdf/)快速渲染, 30 fps on an RTX 3090
[https://arxiv.org/pdf/2212.08476.pdf](https://arxiv.org/pdf/2212.08476.pdf)
[SurfelNeRF: Neural Surfel Radiance Fields for Online Photorealistic Reconstruction of Indoor Scenes (gymat.github.io)](https://gymat.github.io/SurfelNeRF-web/)
SteerNeRF: Accelerating NeRF Rendering via Smooth Viewpoint Trajectory
[[2305.13220] Fast Monocular Scene Reconstruction with Global-Sparse Local-Dense Grids (arxiv.org)](https://arxiv.org/abs/2305.13220) 快速重建
[Distilling Neural Fields for Real-Time Articulated Shape Reconstruction (jefftan969.github.io)](https://jefftan969.github.io/dasr/)实时从视频中重建关节式三维模型的方法

# 节省内存
[https://daniel03c1.github.io/masked_wavelet_nerf/](https://daniel03c1.github.io/masked_wavelet_nerf/)
[https://github.com/AlgoHunt/VQRF](https://github.com/AlgoHunt/VQRF)
[https://plenvdb.github.io/](https://plenvdb.github.io/)快速渲染
[https://sarafridov.github.io/K-Planes/](https://sarafridov.github.io/K-Planes/)快速渲染
[NeRFLight: Fast and Light Neural Radiance Fields Using a Shared Feature Grid | Papers With Code](https://paperswithcode.com/paper/nerflight-fast-and-light-neural-radiance)

# 应用

## 物体探测
[https://github.com/lyclyc52/NeRF_RPN](https://github.com/lyclyc52/NeRF_RPN)
[https://chenfengxu714.github.io/nerfdet/](https://chenfengxu714.github.io/nerfdet/)3D Object Detection
[AutoRecon: Automated 3D Object Discovery and Reconstruction (zju3dv.github.io)](https://zju3dv.github.io/autorecon/) 点云中发现物体并重建

## 分割
[https://rahul-goel.github.io/isrf/](https://rahul-goel.github.io/isrf/)交互式场景
[https://liuff19.github.io/S-Ray/](https://liuff19.github.io/S-Ray/)(可泛化语义分割)
Unsupervised Continual Semantic Adaptation through Neural Rendering 语义分割
[https://github.com/xxm19/jacobinerf](https://github.com/xxm19/jacobinerf)语义分割
[Panoptic Lifting (nihalsid.github.io)](https://nihalsid.github.io/panoptic-lifting/)从野外场景的图像中学习全景三维体积表示的新方法
[[2303.05251] Masked Image Modeling with Local Multi-Scale Reconstruction (arxiv.org)](https://arxiv.org/abs/2303.05251)本地多尺度重构

## RGBD实时跟踪与3D重建
[https://bundlesdf.github.io/](https://bundlesdf.github.io/)
[https://rllab-snu.github.io/projects/RNR-Map/](https://rllab-snu.github.io/projects/RNR-Map/)(视觉导航)

## 机器人感知,抓取感知
[https://bland.website/spartn/](https://bland.website/spartn/)

## 场景识别recognition
[Neural Part Priors: Learning to Optimize Part-Based Object Completion in RGB-D Scans (alexeybokhovkin.github.io)](https://alexeybokhovkin.github.io/neural-part-priors/)利用大规模合成的3D形状数据集，其中包含部分信息的注释，来学习神经部分先验（NPPs）

## 视觉重定位器
[ACE: Accelerated Coordinate Encoding (nianticlabs.github.io)](https://nianticlabs.github.io/ace/)

## 人类
### 人体重建
[https://github.com/JanaldoChen/GM-NeRF](https://github.com/JanaldoChen/GM-NeRF)(可泛化)
[https://yzmblog.github.io/projects/MonoHuman/](https://yzmblog.github.io/projects/MonoHuman/)(,文本交互)
HumanGen: Generating Human Radiance Fields with Explicit Priors
[PersonNeRF: Personalized Reconstruction from Photo Collections | Papers With Code](https://paperswithcode.com/paper/personnerf-personalized-reconstruction-from)
[https://grail.cs.washington.edu/projects/personnerf/](https://grail.cs.washington.edu/projects/personnerf/)
[https://zju3dv.github.io/mlp_maps/](https://zju3dv.github.io/mlp_maps/)动态人体建模
[https://skhu101.github.io/SHERF/](https://skhu101.github.io/SHERF/)


### 人脸渲染
[https://kunhao-liu.github.io/StyleRF/](https://kunhao-liu.github.io/StyleRF/)(高质量人脸)
[https://yudeng.github.io/GRAMInverter/](https://yudeng.github.io/GRAMInverter/)(单视图人像合成)
NeRF-Gaze: A Head-Eye Redirection Parametric Model for Gaze Estimation(视线重定向)
[https://malteprinzler.github.io/projects/diner/diner.html](https://malteprinzler.github.io/projects/diner/diner.html)人脸建模,深度监督

### Gaze redirection 视线重定向
[https://github.com/AlessandroRuzzi/GazeNeRF](https://github.com/AlessandroRuzzi/GazeNeRF)

### 手部重建
HandNeRF: Neural Radiance Fields for Animatable Interacting Hands


## 自拍VR
[https://changwoon.info/publications/EgoNeRF](https://changwoon.info/publications/EgoNeRF)

## 电影剪辑
[https://www.lix.polytechnique.fr/vista/projects/2023_cvpr_wang/](https://www.lix.polytechnique.fr/vista/projects/2023_cvpr_wang/)

## 6-DoF Video
[HyperReel: High-Fidelity 6-DoF Video with Ray-Conditioned Sampling](https://hyperreel.github.io/)

# Datasets
[OmniObject3D: Large-Vocabulary 3D Object Dataset for Realistic Perception, Reconstruction and Generation](https://omniobject3d.github.io/)
[MVImgNet (cuhk.edu.cn)](https://gaplab.cuhk.edu.cn/projects/MVImgNet/#table_stat)
[ReNé (eyecan-ai.github.io)](https://eyecan-ai.github.io/rene/)
[[2303.01932] MobileBrick: Building LEGO for 3D Reconstruction on Mobile Devices (arxiv.org)](https://arxiv.org/abs/2303.01932)

# 扩散模型
[https://sirwyver.github.io/DiffRF/](https://sirwyver.github.io/DiffRF/)
[https://github.com/nianticlabs/diffusionerf](https://github.com/nianticlabs/diffusionerf)
NeRDi: Single-View NeRF Synthesis with Language-Guided Diffusion as General Image Priors扩散模型,语言引导
[https://research.nvidia.com/labs/toronto-ai/NFLDM/](https://research.nvidia.com/labs/toronto-ai/NFLDM/)场景生成
[https://poseguided-diffusion.github.io/](https://poseguided-diffusion.github.io/)视图合成 NeRF-GAN NeRF-Diffusion
[https://make-it-3d.github.io/](https://make-it-3d.github.io/)3D Creation, NeRF-Diffusion
[https://samsunglabs.github.io/NeuralHaircut/](https://samsunglabs.github.io/NeuralHaircut/)Hair Reconstruction, NeRF-Diffusion

# 3D风格迁移
[https://github.com/sen-mao/3di2i-translation](https://github.com/sen-mao/3di2i-translation)
[https://kunhao-liu.github.io/StyleRF/](https://kunhao-liu.github.io/StyleRF/)
Transforming Radiance Field with Lipschitz Network for Photorealistic 3D Scene Stylization
[https://ref-npr.github.io/](https://ref-npr.github.io/)(3D场景风格化)

# text to 3D
[https://research.nvidia.com/labs/dir/magic3d/](https://research.nvidia.com/labs/dir/magic3d/)
[https://github.com/eladrich/latent-nerf](https://github.com/eladrich/latent-nerf)
[https://bluestyle97.github.io/dream3d/](https://bluestyle97.github.io/dream3d/)CLIP, Text-to-3D,扩散模型,零样本
[https://dreambooth3d.github.io/](https://dreambooth3d.github.io/)
[https://sked-paper.github.io/](https://sked-paper.github.io/)
[https://avatar-craft.github.io/](https://avatar-craft.github.io/)Text-to-Avatars
[[2305.02541] Catch Missing Details: Image Reconstruction with Frequency Augmented Variational Autoencoder (arxiv.org)](https://arxiv.org/abs/2305.02541)

# 可编辑
[https://ktertikas.github.io/part_nerf](https://ktertikas.github.io/part_nerf)(部分)
[https://zju3dv.github.io/sine/](https://zju3dv.github.io/sine/)
[https://spinnerf3d.github.io/](https://spinnerf3d.github.io/)(移除物体)
[https://jetd1.github.io/nerflets-web/](https://jetd1.github.io/nerflets-web/)(高效和结构感知的三维场景表示, 大规模，室内室外，场景编辑，全景分割)
[https://jingsenzhu.github.io/i2-sdf/](https://jingsenzhu.github.io/i2-sdf/)(室内重建,可编辑,重光照)
[https://github.com/yizhangphd/FreqPCR](https://github.com/yizhangphd/FreqPCR)(可编辑,点云渲染,实时)
[https://chengwei-zheng.github.io/EditableNeRF/](https://chengwei-zheng.github.io/EditableNeRF/)
[https://snap-research.github.io/discoscene/](https://snap-research.github.io/discoscene/)
[https://nianticlabs.github.io/nerf-object-removal/](https://nianticlabs.github.io/nerf-object-removal/)
[https://palettenerf.github.io/](https://palettenerf.github.io/)外观编辑
[https://zju3dv.github.io/intrinsic_nerf/](https://zju3dv.github.io/intrinsic_nerf/)


# 动态场景
[https://caoang327.github.io/HexPlane/](https://caoang327.github.io/HexPlane/)
[https://dylin2023.github.io/](https://dylin2023.github.io/)
[https://haithemturki.com/suds/](https://haithemturki.com/suds/)(城市动态场景)
[https://github.com/JokerYan/NeRF-DS](https://github.com/JokerYan/NeRF-DS)(光照反射)
[https://robust-dynrf.github.io/](https://robust-dynrf.github.io/)
[https://limacv.github.io/VideoLoop3D_web/](https://limacv.github.io/VideoLoop3D_web/)
[https://nowheretrix.github.io/Instant-NVR/](https://nowheretrix.github.io/Instant-NVR/)人机交互,动态场景
[https://sungheonpark.github.io/tempinterpnerf/](https://sungheonpark.github.io/tempinterpnerf/)
[https://aoliao12138.github.io/ReRF/](https://aoliao12138.github.io/ReRF/)
[https://fengres.github.io/mixvoxels/](https://fengres.github.io/mixvoxels/)
[RoDynRF: Robust Dynamic Radiance Fields (robust-dynrf.github.io)](https://robust-dynrf.github.io/)
[K-Plane (sarafridov.github.io)](https://sarafridov.github.io/K-Planes/)
[HexPlane (caoang327.github.io)](https://caoang327.github.io/HexPlane/)

# 多模态

[SDFusion: Multimodal 3D Shape Completion, Reconstruction, and Generation (yccyenchicheng.github.io)](https://yccyenchicheng.github.io/SDFusion/)

混合表达SDF,高效密集SLAM
[https://www.idiap.ch/paper/eslam/](https://www.idiap.ch/paper/eslam/)
[https://hengyiwang.github.io/projects/CoSLAM](https://hengyiwang.github.io/projects/CoSLAM)NeRF-based SLAM
[https://kxhit.github.io/vMAP](https://kxhit.github.io/vMAP)NeRF-based SLAM, RGBD
Audio Driven
[https://github.com/Fictionarry/ER-NeRF](https://github.com/Fictionarry/ER-NeRF)
