---
title: 微PE装系统
date: 2022-06-21 14:31:45
tags:
    - Windows
categories: 技术力
---

学习使用微PE给电脑重装系统

<!-- more -->

根据主机信息确定型号：[惠普HP Pro 280 G3](https://detail.zol.com.cn/1211/1210556/param.shtml)


- 主板芯片组	Intel H110
- CPU型号	Intel 奔腾双核 G4400
- 内存容量 4GB DDR4 2400Hz
- 硬盘容量	500GB
- 预装系统为 Win10 X64


各种品牌电脑进bios
![20220621143527](https://raw.githubusercontent.com/yq010105/Blog_images/main/blogs/pictures/20220621143527.png)


# 0. 准备

- 一个容量大于8GB的U盘*作为启动盘*


# 1. 微PE安装到U盘

[微PE官网](https://www.wepe.com.cn/)，下载微PE，选择安装PE到U盘

![20220621144124](https://raw.githubusercontent.com/yq010105/Blog_images/main/blogs/pictures/20220621144124.png)

勾选包含DOS工具箱后，立即安装进U盘
![20220621144202](https://raw.githubusercontent.com/yq010105/Blog_images/main/blogs/pictures/20220621144202.png)

等待安装完成即可，安装完成后，会多出一个大概300MB的EFI盘，这就是微PE系统盘，不能随便乱动
**另一个名为微PE工具箱的空白盘即可以当做正常U盘来使用，并将下载的系统镜像拷贝到此盘中**

# 2. 下载系统镜像

1. [MSDN](https://msdn.itellyou.cn/)，使用ed2k协议下载，可以使用迅雷或者闪电下载
由于各种原因，2022/6/21下载没有网速，因此在CSDN上随便找了个大佬上传到百度网盘的WIN10系统镜像进行下载

2. [微软官方下载工具](https://www.microsoft.com/zh-cn/software-download/windows10) 立即下载工具，shift+F10在装系统界面打开终端
`diskpart` | `list disk` | `select disk 0` | `clean` | `convert gpt`或者`convert mbr` `exit`退出

# 3. 开始安装系统
打开电脑，狂按F12(惠普台式机)，进入Bios界面，选择启动项为从U盘启动，然后重启电脑

## 磁盘分区格式与启动引导的关系
UEFI启动的PE，分区表类型GUID/GPT（比较新）
Legacy启动的PE，分区表类型MBR

# 微PE还可以干的事
## 重置Windows忘记的密码
## 给电脑的硬盘分区
