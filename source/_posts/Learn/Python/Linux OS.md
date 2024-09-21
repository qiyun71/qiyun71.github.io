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

## 环境变量

查看：`echo $LD_LIBRARY_PATH`

临时设置：
- `export LD_LIBRARY_PATH=/usr/local/lib64:$LD_LIBRARY_PATH`

永久设置：
- `echo 'export LD_LIBRARY_PATH=/usr/local/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc`
- `source ~/.bashrc` 使修改立即生效

为特定程序设置：
- `LD_LIBRARY_PATH=/usr/local/lib64:$LD_LIBRARY_PATH your_program`



## zip压缩

```bash
# 压缩。如果没有zip命令，安装命令：apt-get update && apt-get install -y zip
zip -r <自定义压缩包名称>.zip <待压缩目录的路径>

# 解压。如果没有zip命令，安装命令：apt-get update && apt-get install -y unzip
unzip  <待解压压缩包名称>.zip -d <解压到哪个路径>
```

## tar

```
# 压缩（具体是指打包，未压缩，非常推荐这种方式，因为压缩/解压都耗时，但是图片等都无法再压缩）
tar -cf <自定义压缩包名称>.tar <待压缩目录的路径>

# 解压
tar -xf <待解压压缩包名称>.tar -C <解压到哪个路径>
```


```
# 压缩
tar -czf <自定义压缩包名称>.tar <待压缩目录的路径>

# 解压
tar -xzf <待解压压缩包名称>.tar -C <解压到哪个路径>
```

## gdm 

动作：关闭gdm `sudo systemctl disable gdm.service` | 更新附加驱动，使用Nvidia专有驱动重启

问题：重启后只有命令行界面

解决：开启gdm `sudo /etc/init.d/gdm start` or `sudo /etc/init.d/gdm3 start`

## V100联网

sudo curl "202.204.48.82/" -X POST -d "DDDDD=b20200267&upass=11241113&v6ip=&0MKKey=123456789" -v

sunyuanbo0287

## chmod 修改权限

> [从今往后，谁再告诉你Linux上chmod -R 777解决权限，果断绝交 - 知乎](https://zhuanlan.zhihu.com/p/255000117)

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

`命令格式：ln -s [源文件或目录] [目标文件或目录]`
- -s --- 参数，软链接(符号链接)
- 源文件或目录 --- 指的是需要连接的源文件，若不在当前目录下需要指明路径
- 目标文件或目录 --- 指的是期望使用时系统或程序需要寻找的文件

软链接
ln -s /usr/local/cuda /usr/local/cuda-11.3

-s, --symbolic              对源文件建立符号链接，而非硬链接；

## ls
$ ls       # 仅列出当前目录可见文件
$ ls -l    # 列出当前目录可见文件详细信息
$ ls -hl   # 列出详细信息并以可读大小显示文件大小，相当于 ll 命令
$ ls -al   # 列出所有文件（包括隐藏）的详细信息

## du

> [linux命令-查看当前目录当前目录剩余空间以及目录文件大小和个数（pg清理大数据量表）_linux查看文件大小-CSDN博客](https://blog.csdn.net/inthat/article/details/108802061)

du
-h 或--human-readable 以K，M，G为单位，提高信息的可读性。
-a或-all 显示目录中个别文件的大小
-s或--summarize 仅显示总计
--max-depth=<目录层数> 超过指定层数的目录后，予以忽略

- 查看当前目录剩余空间：`df -h .`
- 只查看当前目录下文件大小: `du -sh * | sort -nr`
- 查看本目录下占用大小：`du -sh`

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