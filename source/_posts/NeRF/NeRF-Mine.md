---
title: 基于Instant-NSR创建一个项目
date: 2023-07-06 21:17:54
tags:
    - NeRF
    - MyProject
categories: NeRF
---

自己的项目：基于Instant-NSR——[yq010105/NeRF-Mine (github.com)](https://github.com/yq010105/NeRF-Mine)

<!-- more -->

目前准备在Instant—NSR基础上，保留Neus与tcnn，并可以输出texture

文件结构：
- encoder 编码方式
- process_data 处理数据集
- models 放一些网络的结构和网络的运行和方法
- systems 训练的程序
- run.py

- inputs 数据集
- outputs 输出和log文件
    - logs filepath: /root/tf-logs

# 环境配置
选择RTX3090单卡，镜像配置：

- PyTorch  1.10.0
- Python  3.8(ubuntu20.04)
- Cuda  11.3

```
git clone https://github.com/yq010105/NeRF-Mine.git

cd NeRF-Mine

pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```


- `conda create -n nsr python=3.8`
- `conda activate nsr`
- `pip install -r requirements.txt`
- `pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch`

```
# 开始训练
python run.py "inputs/pipaxing-singleframe"  --workspace "/root/tf-logs" --downscale 1 --network sdf

# 提取网格mesh
python run.py "inputs/pipaxing-singleframe"  --workspace "/root/tf-logs" --downscale 1 --network sdf --mode mesh

# 生成特定的目标相机图片
python run.py "inputs/pipaxing-singleframe"  --workspace "/root/tf-logs" --downscale 1 --network sdf --mode render
```