---
title: Linux
date: 2024-03-27 20:25:20
tags: 
categories: Learn
---

Linux 万物皆可文件
> [A Complete Guide to Understanding Linux File System Tree | Cherry Servers](https://www.cherryservers.com/blog/a-complete-guide-to-understanding-linux-file-system-tree)

<!-- more -->

# 桌面

## gdm 

动作：关闭gdm `sudo systemctl disable gdm.service` | 更新附加驱动，使用Nvidia专有驱动重启

问题：重启后只有命令行界面

解决：开启gdm `sudo /etc/init.d/gdm start` or `sudo /etc/init.d/gdm3 start`

## V100联网

sudo curl "202.204.48.82/" -X POST -d "DDDDD=b20200267&upass=11241113&v6ip=&0MKKey=123456789" -v

sunyuanbo0287

## chmod 修改权限

chmod -a+r file

u User，即文件或目录的拥有者；
g Group，即文件或目录的所属群组；
o Other，除了文件或目录拥有者或所属群组之外，其他用户皆属于这个范围；
a All，即全部的用户，包含拥有者，所属群组以及其他用户；
r 读取权限，数字代号为“4”;
w 写入权限，数字代号为“2”；
x 执行或切换权限，数字代号为“1”；
- 不具任何权限，数字代号为“0”；
s 特殊功能说明：变更文件或目录的权限。

## ln

软链接
ln -s /usr/local/cuda /usr/local/cuda-11.3

-s, --symbolic              对源文件建立符号链接，而非硬链接；

## ls
$ ls       # 仅列出当前目录可见文件
$ ls -l    # 列出当前目录可见文件详细信息
$ ls -hl   # 列出详细信息并以可读大小显示文件大小，相当于 ll 命令
$ ls -al   # 列出所有文件（包括隐藏）的详细信息

## du

du
-h 或--human-readable 以K，M，G为单位，提高信息的可读性。
-a或-all 显示目录中个别文件的大小
-s或--summarize 仅显示总计
--max-depth=<目录层数> 超过指定层数的目录后，予以忽略

## 查看进程
软件?
sudo snap install htop
htop 

## 查看os

查看系统版本：
lsb_release -a

查看操作系统架构
$ dpkg --print-architecture
amd64

$ arch
x86_64