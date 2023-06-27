---
title: Instant Neus
date: 2023-06-14 23:15:07
tags:
    - Neus
    - Training & Inference Efficiency
categories: NeRF/Surface Reconstruction
---

使用[Instant-ngp](https://github.com/NVlabs/instant-ngp)中的技术，使Neus可以更快的进行inference，大概只需要5~10min生成一个模型

<!-- more -->

无论文，来源Neus的issue：[Train NeuS in 10min using Instant-NGP acceleration techniques · Issue #78 · Totoro97/NeuS (github.com)](https://github.com/Totoro97/NeuS/issues/78)
[bennyguo/instant-nsr-pl: Neural Surface reconstruction based on Instant-NGP. Efficient and customizable boilerplate for your research projects. Train NeuS in 10min! (github.com)](https://github.com/bennyguo/instant-nsr-pl)

环境配置：
autodl镜像：
    GPU 3090
    PyTorch  1.10.0
    Python  3.8(ubuntu20.04)
    Cuda  11.3
- 安装tiny-cuda-nn扩展之前需要编译好环境
    - 安装tiny-cuda-nn`pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch`
        - 这一步可以先通过clone tiny-cuda-nn，然后在bindings/torch文件夹下运行`python setup.py`
    - 需要提前准备好cuda、cmake、gcc等环境[NVlabs/tiny-cuda-nn: Lightning fast C++/CUDA neural network framework (github.com)](https://github.com/NVlabs/tiny-cuda-nn#requirements)
    - auto-dl生成实例时可以选择pytorch>=1.10.0，cuda11.3or higher，cmake需要升级版本
        - [带你复现nerf之instant-ngp（从0开始搭环境） - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/588741279)
            - `wget https://cmake.org/files/v3.21/cmake-3.21.0.tar.gz`
            - `tar -zxvf cmake-3.21.0.tar.gz`
            - `cd cmake-3.21.0`
            - `./bootstrap && make && sudo make install`
            - `cmake --version`
        - cmake编译完成后，如果cmake -version失败需要进行环境变量的配置 `vim 保存并退出 :wq`
            - [(18条消息) Cmake安装遇到问题_snap cmake_小布米的博客-CSDN博客](https://blog.csdn.net/xiaobumi123/article/details/109578993)
    - sudo apt-get install build-essential git
    - 然后对tiny-cuda-nn进行编译，此处需要用到cuda，因此需要带GPU模式开机
        - `$ git clone --recursive https://github.com/nvlabs/tiny-cuda-nn`
        - `$ cd tiny-cuda-nn`
        - `tiny-cuda-nn$ cmake . -B build`
        - `tiny-cuda-nn$ cmake --build build --config RelWithDebInfo -j`
    - 最后运行setup.py，如果不在全局环境安装，亦可在conda虚拟环境中安装
        - `tiny-cuda-nn$ cd bindings/torch`
        - `tiny-cuda-nn/bindings/torch$ python setup.py install`
- 编译完成cmake和tiny-cuda-nn后，创建虚拟环境并利用pip安装python库
    - `conda create -n inneus python=3.8`
    - `conda init bash` 初始化bash终端
    - `conda activate inneus`
    - `pip install -r requirements.txt`
    - `pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch`
    - `pip install trimesh -i https://mirrors.ustc.edu.cn/pypi/web/simple/`


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

or：
### neus-blender-chair/example@20230624-191845/
python launch.py --config exp/neus-blender-chair/example@20230624-191845/config/parsed.yaml --resume exp/neus-blender-chair/example@20230624-191845/ckpt/epoch=0-step=20000.ckpt --gpu 0 --test
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

### eg: neus_bmvs_bear_womask 给毛绒物体添加缺陷
- 修改images中的照片数据or自己做一个照片数据（毛绒玩具）


### eg: 使用neus自定义数据集对M590三维重建

- 拍M590视频
- video2img.py，将mp4按帧拆分成png，并生成mask文件夹
- 使用colmap对image下图片进行处理，得到相机位姿和点云ply等文件

![录制_2023_06_25_16_27_37_917.gif](https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/%E5%BD%95%E5%88%B6_2023_06_25_16_27_37_917.gif)

- 使用img2poses.py得到sparse_points.ply
    - `cd colmap_preprocess       ##neus/preprocess_custom_data/colmap_preprocess/`
    - `python img2poses.py ${data_dir}`

![image.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/20230625163416.png)

- 根据sparse_points.ply在meshlab中进行clean操作，只保留interest区域的点云，并保存为${data_dir}/sparse_points_interest.ply

![image.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/20230625163527.png)


- 使用gen_cameras.py生成npz文件
    - `python gen_cameras.py ${data_dir}`
    - 在 ${data_dir}下生成preprocessed，包括image、mask和cameras_sphere.npz，但image和mask下无文件，需要自行拷贝
- 打包成zip，上传到百度云盘，然后在AutoDL的AutoPanel中下载到服务器的/root/autodl-tmp文件夹下
- 解压到/instant-nsr-pl/load文件夹下，修改config/neus-dtu.yaml文件中的数据集路径
- 运行`python launch.py --config configs/neus-dtu.yaml --gpu 0 --train`
- train结束后得到obj文件，下载到本地使用meshlab打开

![GIF 2023-6-25 16-44-37.gif](https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/GIF%202023-6-25%2016-44-37.gif)


#### 分析

- 数据集使用xiaomi13录制成视频然后跳帧选取图片得到，会有一些反光和阴影
- M590的右键部分有一块大的缺陷，这是由于数据集在右键部分信息太少(反光，阴影)
- M590的下部分没有建模出来，这是由于数据集未录制其下部分的信息
![M590.gif](https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/M590.gif)

![image.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/20230625165030.png)


## 自定义数据集

### 服务器环境配置colmap (未完成)

```text
git clone https://ceres-solver.googlesource.com/ceres-solver
cd ceres-solver
git checkout $(git describe --tags) # Checkout the latest release
mkdir build
cd build
cmake .. -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF
make -j
sudo make install
```

### 本地处理数据集(同neus)

同neus使用自定义数据集[Neus | YunQi (yq010105.github.io)](https://yq010105.github.io/2023/06/14/NeRF/Surface%20Reconstruction/Neus/)

