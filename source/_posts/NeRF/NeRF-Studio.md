---
title: Nerfstudio——简化NeRF流程
date: 2023-06-15 12:16:19
tags:
    - NeRF
    - Framework
categories: NeRF
---

| Title     | Nerfstudio: A Modular Framework for Neural Radiance Field Development                                                                                                                                                                       |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Author    | Tancik, Matthew and Weber, Ethan and Ng, Evonne and Li, Ruilong and Yi, Brentand Kerr, Justin and Wang, Terrance and Kristoffersen, Alexander and Austin,Jake and Salahi, Kamyar and Ahuja, Abhik and McAllister, David and Kanazawa,Angjoo |
| Conf/Jour | ACM SIGGRAPH 2023 Conference Proceedings                                                                                                                                                                                                    |
| Year      | 2023                                                                                                                                                                                                                                        |
| Project   | [nerfstudio-project/nerfstudio: A collaboration friendly studio for NeRFs (github.com)](https://github.com/nerfstudio-project/nerfstudio/)                                                                                                  |
| Paper     | [Nerfstudio: A Modular Framework for Neural Radiance Field Development (readpaper.com)](https://readpaper.com/pdf-annotate/note?pdfId=4762351665164582913&noteId=1908666225137730048)                                                       |

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

## 加载数据&训练model

`ns-train nerfacto --data data/nerfstudio/poster`

### Download some test data:

`ns-download-data nerfstudio --capture-name=poster`

```ad-error
AutoDL连接不了google drive，只能使用自己的数据集or：
    使用google的colab下载数据集并将其打包成zip，然后再上传到autodl
```
### Use Own Data 

{% note primary %} 配好环境后，可以在任意地址创建文件夹，放入需要训练的数据集 {% endnote %}

`ns-process-data {video,images,polycam,record3d} --data {DATA_PATH} --output-dir {PROCESSED_DATA_DIR}`

`ns-process-data {images, video} --data {DATA_PATH} --output-dir {PROCESSED_DATA_DIR}`

#### eg: Miku
cd autodl-tmp
`ns-process-data images --data data/images --output-dir data/nerfstudio/images_name`

跳过图像处理：复制和缩放
`ns-process-data images --data data/Miku/image/ --output-dir data/nerfstudio/Miku --skip-image-processing`

```
06.29:
(nerfstudio) root@autodl-container-7092458c99-5f01fa1c:~/autodl-tmp# ns-process-data images  --data data/Miku/image/ --output-dir data/nerfstudio/Miku --skip-image-processing --skip-colmap  
[15:37:47] Only single camera shared for all images is supported.
数据集必须是单个相机去拍照物体？？？
无所谓：无卡开机用cpu算
ns-process-data images  --data data/Miku/image/ --output-dir data/nerfstudio/Miku --skip-image-processing --no-gpu
依然不行

问题&原因：
qt.qpa.xcb: could not connect to display qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "" even though it was found.  This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application may fix this problem.
最大的可能就是 --SiftExtraction.use_gpu 1  必须要求GPU带一个显示器


06.30:
使用3090开机但是use no gpu
ns-process-data images  --data data/Miku/image/ --output-dir data/nerfstudio/Miku --skip-image-processing --no-gpu

[15:32:40] 🎉 Done extracting COLMAP features.                                                       colmap_utils.py:131
[15:49:59] 🎉 Done matching COLMAP features.                                                         colmap_utils.py:145
[15:53:28] 🎉 Done COLMAP bundle adjustment.                                                         colmap_utils.py:167
[15:53:56] 🎉 Done refining intrinsics.                                                              colmap_utils.py:176
           🎉 🎉 🎉 All DONE 🎉 🎉 🎉                                                images_to_nerfstudio_dataset.py:100
           Starting with 178 images                                                  images_to_nerfstudio_dataset.py:103
           Colmap matched 178 images                                                 images_to_nerfstudio_dataset.py:103
           COLMAP found poses for all images, CONGRATS!                              images_to_nerfstudio_dataset.py:103
train：
ns-train nerfacto --data data/nerfstudio/Miku

```

in viewer:  it is easy to view results and process

![image.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/20230630161446.png)


### Train model
`ns-train nerfacto --data data/nerfstudio/poster`

## export 
### mesh

手动调整参数得到命令：

```
ns-export poisson --load-config outputs/Miku/nerfacto/2023-06-30_155708/config.yml --output-dir exports/mesh/ --target-num-faces 50000 --num-pixels-per-side 2048 --normal-method open3d --num-points 1000000 --remove-outliers True --use-bounding-box True --bounding-box-min -0.5 -0.5 -1 --bounding-box-max 0.5 0.5 0

output: 
Loading latest checkpoint from load_dir  
✅ Done loading checkpoint from outputs/Miku/nerfacto/2023-06-30_155708/nerfstudio_models/step-000029999.ckpt  
☁ Computing Point Cloud ☁ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 00:05  
✅ Cleaning Point Cloud  
✅ Estimating Point Cloud Normals  
✅ Generated PointCloud with 1008679 points.  
Computing Mesh... this may take a while.

CPU生成mesh的速度很慢 大约用了1个小时多，效果也不是很好，因为使用的是nerfacto的方法，零水平集有很多坑洞
```



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
- eg: `ssh -L 7007:localhost:7007 root@connect.beijinga.seetacloud.com -p 23394`

此时打开[nerfstudio viewer](https://viewer.nerf.studio/)，在Getting started中输入ws://localhost:7007，即可在viewer中查看

#### 更换服务器的端口
- 当服务器的7007被占用时：
    默认为7007，修改端口7007为6006 并训练
    `ns-train nerfacto --data data/nerfstudio/poster --viewer.websocket-port 6006`
- 此时在本地需运行
    `ssh -L 7007:localhost:6006 root@connect.beijinga.seetacloud.com -p 23394`

