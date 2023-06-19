---
title: Nerfstudio——简化NeRF流程
date: 2023-06-15 12:16:19
tags:
    - NeRF Framework
categories: NeRF
---

[Nerfstudio](https://github.com/nerfstudio-project/nerfstudio/)提供了一个简单的API，可以简化创建、训练和测试NeRF的端到端过程。该库通过将每个组件模块化，支持更易于理解的NeRF实现。通过更模块化的NeRF，我们希望为探索这项技术提供更用户友好的体验。

<!-- more -->

# Autodl使用
选择实例，pytorch2.0.0，python3.8，cuda11.8

## 环境配置
[Installation - nerfstudio](https://docs.nerf.studio/en/latest/quickstart/installation.html)

```
conda create --name nerfstudio -y python=3.8
conda activate nerfstudio
python -m pip install --upgrade pip
```
for cuda11.8，需要很长时间
```
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

```shell
pip install nerfstudio

默认源不好用，使用清华源
pip install nerfstudio -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 安装FFmpeg
[在Ubuntu 20.04 中安装FFMPEG-之路教程 (onitroad.com)](https://www.onitroad.com/jc/linux/ubuntu/faq/how-to-install-ffmpeg-on-ubuntu-20-04.html)
```
sudo apt update 
sudo apt install ffmpeg

ffmpeg -version
```

### 安装Colmap

`sudo apt install colmap`

## 训练model

### Download some test data:

`ns-download-data nerfstudio --capture-name=poster`

```ad-error
AutoDL连接不了google drive，只能使用自己的数据集or：
    使用google的colab下载数据集并将其打包成zip，然后再上传到autodl
```
### use own data 

`ns-process-data {video,images,polycam,record3d} --data {DATA_PATH} --output-dir {PROCESSED_DATA_DIR}`

`ns-process-data {images, video} --data {DATA_PATH} --output-dir {PROCESSED_DATA_DIR}`

eg: 
cd autodl-tmp
`ns-process-data images --data data/images --output-dir data/nerfstudio/images_name`

### Train model
`ns-train nerfacto --data data/nerfstudio/poster`



## 使用viewer
[nerfstudio viewer](https://viewer.nerf.studio/)

### autodl
本地运行：`ssh -CNg -L 6006:127.0.0.1:6006 root@123.125.240.150 -p 42151`

```ad-important
本地端口:localhost:远程端口
```

一般本地进入服务器（ssh）
`ssh -p 23394 root@connect.beijinga.seetacloud.com`

将服务器6006端口映射到本地的6006端口上
`ssh -CNg -L 6006:127.0.0.1:6006 root@connect.beijinga.seetacloud.com -p 23394`

### viewer
一般nerfstudio的viewer运行在本地的7007端口上
`ssh -L 7007:localhost:7007 <username>@<remote-machine-ip>`

需要在本地再开一个终端，并运行，将本地的6006端口与远程的7007进行绑定
- `ssh -L 7007:localhost:7007 root@connect.beijinga.seetacloud.com -p 23394`

此时打开[nerfstudio viewer](https://viewer.nerf.studio/)，在Getting started中输入ws://localhost:7007，即可在viewer中查看

#### 更换服务器端口
- 当服务器的7007被占用时：
    默认为7007，修改端口7007为6006 并训练
    `ns-train nerfacto --data data/nerfstudio/poster --viewer.websocket-port 6006`
- 此时在本地需运行
    `ssh -L 7007:localhost:6006 root@connect.beijinga.seetacloud.com -p 23394`

