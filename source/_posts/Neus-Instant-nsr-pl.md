---
title: Instant Neus
date: 2023-06-14 23:15:07
tags:
    - Neus
    - Training & Inference Efficiency
categories: NeRF
---

使用[Instant-ngp](https://github.com/NVlabs/instant-ngp)中的技术，使Neus可以更快的进行inference，大概只需要5~10min生成一个模型

<!-- more -->

无论文，来源Neus的issue：[Train NeuS in 10min using Instant-NGP acceleration techniques · Issue #78 · Totoro97/NeuS (github.com)](https://github.com/Totoro97/NeuS/issues/78)
[bennyguo/instant-nsr-pl: Neural Surface reconstruction based on Instant-NGP. Efficient and customizable boilerplate for your research projects. Train NeuS in 10min! (github.com)](https://github.com/bennyguo/instant-nsr-pl)

环境配置：
autodl镜像：
    PyTorch  1.10.0
    Python  3.8(ubuntu20.04)
    Cuda  11.3
- Install PyTorch>=1.10 [here](https://pytorch.org/get-started/locally/) based the package management tool you used and your cuda version (older PyTorch versions may work but have not been tested)
- - Install tiny-cuda-nn PyTorch extension: `pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch`
    - 这一步可以先通过clone tiny-cuda-nn，然后在bindings/torch文件夹下运行`python setup.py`
    - 需要提前准备好cuda、cmake、gcc等环境[NVlabs/tiny-cuda-nn: Lightning fast C++/CUDA neural network framework (github.com)](https://github.com/NVlabs/tiny-cuda-nn#requirements)
    - auto-dl生成实例时可以选择pytorch>=1.10.0，cuda12.0，cmake需要升级版本
        - [带你复现nerf之instant-ngp（从0开始搭环境） - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/588741279)
        - cmake编译完成后需要进行环境变量的配置 `vim 保存并退出 :wq`
            - [(18条消息) Cmake安装遇到问题_snap cmake_小布米的博客-CSDN博客](https://blog.csdn.net/xiaobumi123/article/details/109578993)
    - sudo apt-get install build-essential git
    - 然后对tiny-cuda-nn进行编译，
        - $ git clone --recursive https://github.com/nvlabs/tiny-cuda-nn
        - $ cd tiny-cuda-nn
        - tiny-cuda-nn$ cmake . -B build
        - tiny-cuda-nn$ cmake --build build --config RelWithDebInfo -j
- - `pip install -r requirements.txt`


# 数据集：
## NeRF-Synthetic，解压并放在/load文件夹下，The file structure should be like `load/nerf_synthetic/lego`.

```
# train NeRF
python launch.py --config configs/nerf-blender.yaml --gpu 0 --train dataset.scene=lego tag=example

# train NeuS with mask
python launch.py --config configs/neus-blender.yaml --gpu 0 --train dataset.scene=lego tag=example
# train NeuS without mask
python launch.py --config configs/neus-blender.yaml --gpu 0 --train dataset.scene=lego tag=example system.loss.lambda_mask=0.0
```


代码快照、ckpt和实验输出保存在 exp/[name]/[tag]@[timestamp]，而TensorBoard日志可以在 runs/[name]/[tag]@[timestamp] 找到。您可以通过在YAML文件中指定参数而无需添加--来更改任何配置，例如：
`python launch.py --config configs/nerf-blender.yaml --gpu 0 --train dataset.scene=lego tag=iter50k seed=0 trainer.max_steps=50000`

test，生成mp4和mesh的obj文件
```
python launch.py --config path/to/your/exp/config/parsed.yaml --resume path/to/your/exp/ckpt/epoch=0-step=20000.ckpt --gpu 0 --test

- eg:
python launch.py --config exp/neus-blender-lego/example@20230601-185640/config/parsed.yaml --resume exp/neus-blender-lego/example@20230601-185640/ckpt/epoch=0-step=20000.ckpt --gpu 0 --test
```

### eg: neus_chair
neus_chair_wmask
`python launch.py --config configs/neus-blender.yaml --gpu 0 --train dataset.scene=chair tag=example`
![Pasted image 20230601205022.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/Pasted%20image%2020230601205022.png)


![Pasted image 20230601205055.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/Pasted%20image%2020230601205055.png)


nerf生成mesh的obj文件很小（nerf_chair）：
`python launch.py --config configs/nerf-blender.yaml --gpu 0 --train dataset.scene=chair tag=example`
![Pasted image 20230601202815.png|300](https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/Pasted%20image%2020230601202815.png)


neus_chair_womask
`python launch.py --config configs/neus-blender.yaml --gpu 0 --train dataset.scene=chair tag=chair system.loss.lambda_mask=0.0`

![Pasted image 20230601212157.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/Pasted%20image%2020230601212157.png)


![Pasted image 20230601212246.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/Pasted%20image%2020230601212246.png)


### eg: neus_lego
neus_lego_wmask
`python launch.py --config configs/neus-blender.yaml --gpu 0 --train dataset.scene=lego tag=lego`
![Pasted image 20230601213938.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/Pasted%20image%2020230601213938.png)


![Pasted image 20230601214004.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/Pasted%20image%2020230601214004.png)



neus_lego & w/o mask
`python launch.py --config configs/neus-blender.yaml --gpu 0 --train dataset.scene=lego tag=example system.loss.lambda_mask=0.0`
迭代20000步，大致需要18分钟，生成的模型细节还是不够精细
![Pasted image 20230601201131.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/Pasted%20image%2020230601201131.png)

![Pasted image 20230601201258.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/Pasted%20image%2020230601201258.png)

![Pasted image 20230601201510.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/Pasted%20image%2020230601201510.png)



## Blended_MVS，DTU
```
# train NeuS on DTU without mask
python launch.py --config configs/neus-dtu.yaml --gpu 0 --train
# train NeuS on DTU with mask
python launch.py --config configs/neus-dtu.yaml --gpu 0 --train system.loss.lambda_mask=0.1
```

```ad-note
作者只提供了DTU数据集的加载方式，但是DTU和Bmvs相差不大，因此只需要作微小修改即可完成bmvs数据集的处理：
```

修改config/neus-dtu.yaml文件，dtu保持不变(需要用到dtu.py数据集加载文件)，修改数据集的文件路径（dtu与bmvs数据集差别不大，都是由image、mask和cameras_sphere.npz组成

在/datasets/dtu.py中修改数据集的文件名，
- dtu的是前面补0到6位数.png
- bmvs是前面补0到3位数.png

```
灵活调整dtu.py 文件
# # DTU 数据集
# img_sample = cv2.imread(os.path.join(self.config.root_dir, 'image', '000000.png'))
# BMVS数据集
img_sample = cv2.imread(os.path.join(self.config.root_dir, 'image', '000.png'))

# bmvs
img_path = os.path.join(self.config.root_dir, 'image', f'{i:03d}.png')
# DTU
# img_path = os.path.join(self.config.root_dir, 'image', f'{i:06d}.png')
```

### eg: neus_bmvs_clock_womask

`python launch.py --config configs/neus-dtu.yaml --gpu 0 --train`
![Pasted image 20230602160929.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/Pasted%20image%2020230602160929.png)

![Pasted image 20230602160959.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/Pasted%20image%2020230602160959.png)



也有噪声
![Pasted image 20230602161034.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/Pasted%20image%2020230602161034.png)



### eg: neus_bmvs_clock_wmask
`python launch.py --config configs/neus-dtu.yaml --gpu 0 --train system.loss.lambda_mask=0.1`

![Pasted image 20230602172423.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/Pasted%20image%2020230602172423.png)


![Pasted image 20230602172441.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/Pasted%20image%2020230602172441.png)

### eg: neus_bmvs_bear_womask
`python launch.py --config configs/neus-dtu.yaml --gpu 0 --train`

![Pasted image 20230602163439.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/Pasted%20image%2020230602163439.png)


![Pasted image 20230602163553.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/Pasted%20image%2020230602163553.png)



## 自定义数据集