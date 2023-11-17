---
title: IPV6用法
date: 2023-11-16 21:16:09
tags:
  - Tools
categories: Other/Mine
---

校园网 IPV4 流量限量 120G，超额**0.6 元/G**
**开学季|校园网使用指南**URL: https://mp.weixin.qq.com/s/NsojrsEUVeGOfSEEkpY3kw

IPV6 免费使用

<!-- more -->

# PT 站

校园网 IPV6 免费流量

- 北洋园[北洋园PT :: 首页 - Powered by NexusPHP (tjupt.org)](https://tjupt.org/index.php)
- 北邮人 [BYRBT :: 首页 - Powered by NexusPHP](https://byr.pt/index.php)

# 镜像网站

[清华大学开源软件镜像站 | Tsinghua Open Source Mirror(https://mirrors6.tuna.tsinghua.edu.cn/)](https://mirrors6.tuna.tsinghua.edu.cn/) 只解析 IPv6

## Conda

Conda 配置 ipv6 的清华源

> 为 Conda 添加清华软件源 - 知乎 URL: https://zhuanlan.zhihu.com/p/47663391

```
# Anaconda官方库的镜像
conda config --add channels https://mirrors6.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors6.tuna.tsinghua.edu.cn/anaconda/pkgs/main/

# Anaconda第三方库 Conda Forge的镜像
conda config --add channels https://mirrors6.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/

# Pytorch的Anaconda第三方镜像
## win10 win-64
conda config --add channels https://mirrors6.tuna.tsinghua.edu.cn/anaconda/cloud/peterjc123/
## linux
conda config --add channels https://mirrors6.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/

conda config --set show_channel_urls yes
# 添加完成后可以使用conda info命令查看是否添加成功.
```

- Win10 的 conda 配置文件在 `C:\Users\用户名` 下.Condarc
- Ubuntu 在 `\home\用户名` 下 .condarc

```
channels:
  - https://mirrors6.tuna.tsinghua.edu.cn/anaconda/cloud/peterjc123/
  - https://mirrors6.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
  - https://mirrors6.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
  - https://mirrors6.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors6.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors6.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors6.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors6.tuna.tsinghua.edu.cn/anaconda/cloud
  msys2: https://mirrors6.tuna.tsinghua.edu.cn/anaconda/cloud
  bioconda: https://mirrors6.tuna.tsinghua.edu.cn/anaconda/cloud
  menpo: https://mirrors6.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors6.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch-lts: https://mirrors6.tuna.tsinghua.edu.cn/anaconda/cloud
  simpleitk: https://mirrors6.tuna.tsinghua.edu.cn/anaconda/cloud
```

## Pip

Pip 配置 pip 的包在/pypi/web/simple/目录下：

`pip config set global.index-url https://mirrors6.tuna.tsinghua.edu.cn/pypi/web/simple/`

- win10会在 `C:\Users\用户名\AppData\Roaming\pip` 下生成 pip.Ini 文件
- Ubuntu 在 `/home/用户名/.config/pip/` 下 pip.Conf

```
[global]
index-url = https://mirrors6.tuna.tsinghua.edu.cn/pypi/web/simple/
```



