NeRF-based 3D Reconstruction项目

| NeRF-based 主要参考项目   | Project Page                               | 描述                                                                     |
| ------------------- | ------------------------------------------ | ---------------------------------------------------------------------- |
| NeRF                | https://github.com/yenchenlin/nerf-pytorch | NeRF原始方法                                                               |
| NeuS精度              | https://github.com/Totoro97/NeuS           | 密度替换成SDF，重建精度提高，但是速度很慢                                                 |
| INSR速度              | https://github.com/bennyguo/instant-nsr-pl | [InstantNGP](https://github.com/NVlabs/instant-ngp)+NeuS保证重建精度，提高了重建速度 |
| Instant-angelo速度+精度 | https://github.com/hugoycj/Instant-angelo  | 基于[Neuralangelo ](https://github.com/NVlabs/neuralangelo)的非官方实现        |
| Neuralangelo        | https://github.com/NVlabs/neuralangelo     | 改进了InstantNGP的编码，提高了重建精度                                               |

其他方法——精度不高：
- [19reborn/NeuS2: Official code for NeuS2](https://github.com/19reborn/NeuS2)
- [zhaofuq/Instant-NSR: Pytorch implementation of fast surface resconstructor](https://github.com/zhaofuq/Instant-NSR)
- [Colmar-zlicheng/Color-NeuS: [3DV 2024] Color-NeuS: Reconstructing Neural Implicit Surfaces with Color](https://github.com/Colmar-zlicheng/Color-NeuS)

---

学习流程：
1. 先复现了NeRF和NeuS，了解重建的基本流程
2. 根据INSR构建了一下自己的项目：**NeRF-Mine**
3. 根据Neuralangelo(Instant-angelo的代码)，给项目中添加了ProgressiveBandHashGrid方法

项目用到的库：(主要参考[INSR](https://github.com/bennyguo/instant-nsr-pl))
- 采样方式 https://github.com/nerfstudio-project/nerfacc
  - 一直用的0.3.5版本，现在说明文档好像被删了，只剩下最新版文档
- 编码方式 https://github.com/NVlabs/tiny-cuda-nn
  - Python调用InstantNGP的库
- 自制数据集(相机位姿格式...) https://github.com/Totoro97/NeuS/tree/main/preprocess_custom_data

一些工具类项目：
- colmap https://colmap.github.io/index.html
- **评价模型质量** https://github.com/jzhangbs/DTUeval-python

**NeRF-Mine**项目代码框架：
- confs/ 配置文件（数据集路径、网络结构、训练超参数等等）
  - neus-dtu.yaml（DTU数据集配置）
  - neus-dtu_like.yaml （自制数据集配置）
- encoder/ 编码方式（位置编码和方向编码）
  - get_encoding.py
  - frequency.py
  - hashgrid.py
  - spherical.py
- process_data/ 处理数据（数据预处理）
  - dtu.py
  - dtu_like.py
- models/ 网络的结构和网络的运行和方法
  - network.py 基本网络结构
  - neus.py NeuS的网络
  - utils.py 工具类
- systems/ 训练的程序
  - neus.py 训练 neus 的程序
- utils/ 工具类函数
- run.py 主程序
- inputs/ 数据集
- outputs/ 输出和 log 文件
  - logs filepath: /root/tf-logs/name_in_conf/trial_name
    - logs通过使用`tensorboard --logdir /root/tf-logs/...` 来查看


实验记录：

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
