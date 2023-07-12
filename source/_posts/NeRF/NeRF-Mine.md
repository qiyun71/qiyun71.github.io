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

- confs 配置文件
- inputs 数据集
- outputs 输出和log文件
    - logs filepath: /root/tf-logs

# 代码结构

## confs

```
eg: dtu.conf
general {
    base_exp_dir = /root/tf-logs
}

dataset {
    data_dir = ./inputs/CASE_NAME/
    render_cameras_name = cameras_sphere.npz
    object_cameras_name = cameras_sphere.npz
}
```

## run.py

```
import argparse

def config_parser():
    parser = argparse.ArgumentParser()
    
    args = parser.parse_args()
    
    return args
```

# 环境配置
选择RTX3090单卡，镜像配置：

- PyTorch  1.10.0
- Python  3.8(ubuntu20.04)
- Cuda  11.3

```
source /etc/network_turbo

git clone https://github.com/yq010105/NeRF-Mine.git

cd NeRF-Mine
```


- `conda create -n nsr python=3.8`
- `conda activate nsr`
- `pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple`
- 可选`pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch`

## error
```
1
Failed to build pysdf  
ERROR: Could not build wheels for pysdf, which is required to install pyproject.toml-based projects

先取消pysdf的安装，安装完其他后再进行pysdf的安装

2
加载cpp扩展时，锁住：
File "run.py", line 6, in <module>  
from models.network_sdf import NeRFNetwork  
File "/root/autodl-tmp/NeRF-Mine/models/network_sdf.py", line 8, in <module>  
from encoder.encoding import get_encoder  
File "/root/autodl-tmp/NeRF-Mine/encoder/encoding.py", line 8, in <module>  
from encoder.shencoder import SHEncoder  
File "/root/autodl-tmp/NeRF-Mine/encoder/shencoder/__init__.py", line 1, in <module>  
from .sphere_harmonics import SHEncoder  
File "/root/autodl-tmp/NeRF-Mine/encoder/shencoder/sphere_harmonics.py", line 9, in <module>  
from .backend import _backend  
File "/root/autodl-tmp/NeRF-Mine/encoder/shencoder/backend.py", line 6, in <module>  
_backend = load(name='_sh_encoder',  
File "/root/miniconda3/envs/nsr/lib/python3.8/site-packages/torch/utils/cpp_extension.py", line 1284, in load  
return _jit_compile(  
File "/root/miniconda3/envs/nsr/lib/python3.8/site-packages/torch/utils/cpp_extension.py", line 1523, in _jit_compile  
baton.wait()  
File "/root/miniconda3/envs/nsr/lib/python3.8/site-packages/torch/utils/file_baton.py", line 42, in wait  
time.sleep(self.wait_seconds)  
KeyboardInterrupt

原因：
当加载hash编码后，快速加载了shen编码，导致cpp_extension.py依然被占用，因此，只需加一个延时，让两个库导入时后者慢一步，解除程序的占用
方法：
删除/root/.cache/torch_extensions/py38_cu117/下文件_hash_encoder


from encoder.hashencoder import HashEncoder
之间添加延时 time.sleep(10)
from encoder.shencoder import SHEncoder
Note!延时在运行过第一次后可以注释掉
```

> error2:[(21条消息) torch.utils.cpp_extension.load卡住无响应_zParquet的博客-CSDN博客](https://blog.csdn.net/qq_38677322/article/details/109696077)

# 在示例数据集上训练

下载InstantNSR提供的示例数据集：a test dataset [dance](https://drive.google.com/drive/folders/180qoFqABXjBDwW2hHa14A6bmV-Sj1qqJ?usp=sharing)，放入inputs目录下

```
# 开始训练
python run.py --conf confs/dtu.conf --downscale 1 --network sdf
or
python run.py --conf confs/dtu.conf

# 提取网格mesh
python run.py --downscale 1 --network sdf --mode mesh

# 生成特定的目标相机图片
python run.py --downscale 1 --network sdf --mode render
```

训练结果不理想：猜想是由于没有去除后面的背景，可以说nsr是一个依赖mask的方法

![image.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/20230710211129.png)


# TODO

- [ ]  集成Neus自定义数据集的方式
- [ ] 添加NeRFacc的采样技术

- [ ]  添加InstantNSR的生成texture方式