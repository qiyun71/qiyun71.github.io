---
title: VideoSR
date: 2024-12-17 13:17:50
tags:
  - 
categories: Project
---

视频超分

<!-- more -->

## Ccrestoration

[TensoRaws/ccrestoration: an inference lib for image/video restoration with VapourSynth support](https://github.com/TensoRaws/ccrestoration)

Make sure you have Python >= 3.9 and PyTorch >= 1.13 installed

```shell
pip install ccrestoration
```

Install VapourSynth (optional)

为apt添加 repository

```bash
wget https://www.deb-multimedia.org/pool/main/d/deb-multimedia-keyring/deb-multimedia-keyring_2024.9.1_all.deb  
sudo dpkg -i deb-multimedia-keyring_2024.9.1_all.deb
sudo add-apt-repository 'deb https://www.deb-multimedia.org/ bookworm main non-free'
sudo apt update
sudo apt install vapoursynth
```

error:
```bash
vapoursynth : Depends: libpython3.11 (>= 3.11.0) but it is not installable
              Depends: python3 (< 3.12) but 3.12.3-0ubuntu2 is to be installed
              Depends: vapoursynth-ffms2 but it is not going to be installed
```


- 安装 python 3.11

```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.11

sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

sudo apt install python3.9
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.9 2

update-alternatives --list python

# 可以任意切换版本：
update-alternatives --config python
```

- 安装vapoursynth-ffms2

```bash
sudo apt install vapoursynth-ffms2
libavcodec60 : Depends: libcodec2-1.0 (>= 1.0.5) but it is not installable
               Depends: libvpx7 (>= 1.12.0) but it is not installable
vapoursynth : Depends: python3 (< 3.12) but 3.12.3-0ubuntu2 is to be installed

sudo apt install libcodec2-1.0
sudo apt install libvpx7
sudo apt install 
```


## VSET

[NangInShell/VSET: 基于Vapoursynth的图形化视频批量压制处理工具，超分辨率，补帧，vs滤镜一应俱全。](https://github.com/NangInShell/VSET)

简单点...


# Image Super-resolution

>  [zsyOAOA/InvSR: Arbitrary-steps Image Super-resolution via Diffusion Inversion](https://github.com/zsyOAOA/InvSR?tab=readme-ov-file)

![framework.png (8232×4536)|666](https://raw.githubusercontent.com/zsyOAOA/InvSR/refs/heads/master/assets/framework.png)