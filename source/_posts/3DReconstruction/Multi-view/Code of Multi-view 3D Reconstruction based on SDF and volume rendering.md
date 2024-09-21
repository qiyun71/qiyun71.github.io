---
title: Code of Multi-view 3D Reconstruction based on SDF and volume rendering
date: 2024-09-03 21:52:05
tags:
  - 3DReconstruction
categories: 3DReconstruction/Multi-view
---

论文代码复现(Linux OS)

<!-- more -->

# 基本必备工具

## tiny-cuda-nn

> [NVlabs/tiny-cuda-nn: Lightning fast C++/CUDA neural network framework](https://github.com/NVlabs/tiny-cuda-nn#requirements)

环境需求
- A **C++14** capable compiler. The following choices are recommended and have been tested:
    - **Windows:** Visual Studio 2019 or 2022
    - **Linux:** GCC/G++ 8 or higher
- A recent version of **[CUDA](https://developer.nvidia.com/cuda-toolkit)**. The following choices are recommended and have been tested:
    - **Windows:** CUDA 11.5 or higher
    - **Linux:** CUDA 10.2 or higher
- **[CMake](https://cmake.org/) v3.21 or higher**.

如果gcc、cmake的版本正确，则直接安装：
1. `sudo apt-get install build-essential git`
2. 然后对tiny-cuda-nn进行编译，此处需要用到cuda，因此需要带GPU模式开机
3. `git clone --recursive "https://github.com/nvlabs/tiny-cuda-nn"`
  1. `cd tiny-cuda-nn`
  2. `cmake . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo`
  3. `cmake --build build --config RelWithDebInfo -j`
4. 最后运行setup.py，如果不在全局环境安装，亦可在conda虚拟环境中安装
  1. `cd bindings/torch`
  2. `python setup.py install`

### gcc install/update

1. 下载 [Index of /gnu/gcc/gcc-9.4.0](https://ftp.gnu.org/gnu/gcc/gcc-9.4.0/) 的 gcc-9.4.0.tar.gz
2. 将文件解压到/usr/local目录下面 `tar -zvxf gcc-9.4.0.tar.gz --directory=/usr/local/`
3. `vim /usr/local/gcc-9.4.0/contrib/download_prerequisites` 查看gcc所需依赖及其版本
  1. gmp='gmp-6.1.0.tar.bz2'
  2. mpfr='mpfr-3.1.4.tar.bz2'
  3. mpc='mpc-1.0.3.tar.gz'
  4. isl='isl-0.18.tar.bz2'
4. 下载所有依赖 [Index of /pub/gcc/infrastructure](https://gcc.gnu.org/pub/gcc/infrastructure/) 
5. 解压依赖
  1. `tar -jvxf gmp-6.1.0.tar.bz2 --directory=/usr/local/gcc-9.4.0/`
  2. `tar -jvxf mpfr-3.1.4.tar.bz2 --directory=/usr/local/gcc-9.4.0/`
  3. `tar -zvxf mpc-1.0.3.tar.gz --directory=/usr/local/gcc-9.4.0/`
  4. `tar -jvxf isl-0.18.tar.bz2 --directory=/usr/local/gcc-9.4.0/`
6. 为新下载的依赖建立软链接 **gcc-9.4.0目录下**
  1. `cd /usr/local/gcc-9.4.0/`
  2. sudo ln -sf gmp-6.1.0 gmp
  3. sudo ln -sf isl-0.18 isl
  4. sudo ln -sf mpc-1.0.3 mpc
  5. sudo ln -sf mpfr-3.1.4 mpfr
  6. 查看软连接`ls -l | grep ^l`
7. 编译并安装gcc 
  1. `mkdir build && cd build`
  2. `../configure -enable-checking=release -enable-languages=c,c++ -disable-multilib --disable-nls` NOTE: 需要禁用 NLS
  3. 成功后会出现 config.log config.status Makefile serdep.tmp
  4. `sudo make -j4 && sudo make install`
8. 更新gcc (安装完后gcc -version,你会发现显示的还是原来的版本eg: 7.5.0)
  1. 备份一下原来的gcc
    1. `sudo mv /usr/bin/gcc /usr/bin/gcc750`
    2. `sudo mv /usr/bin/g++ /usr/bin/g++750`
    3. `sudo mv /usr/bin/c++ /usr/bin/c++750`
    4. `sudo mv /usr/bin/cc /usr/bin/cc750`
    5. `sudo mv /usr/local/lib64/libstdc++.so.6 /usr/local/lib64/libstdc++.so.6.bak` # mv: cannot stat '/usr/lib64/libstdc++.so.6': No such file or directory
  2. 软连接 # 安装的gcc新版本位于/usr/local/bin
    1. `sudo ln -s /usr/local/bin/gcc /usr/bin/gcc`
    2. `sudo ln -s /usr/local/bin/g++ /usr/bin/g++`
    3. `sudo ln -s /usr/local/bin/c++ /usr/bin/c++`
    4. `sudo ln -s /usr/local/bin/gcc /usr/bin/cc`
    5. `sudo ln -s /usr/local/lib64/libstdc++.so.6.0.28 /usr/local/lib64/libstdc++.so.6`


Error:

```
make -j4 && make install

*** LIBRARY_PATH shouldn't contain the current directory when
*** building gcc. Please change the environment variable
*** and run configure again.V
Makefile:4371: recipe for target 'configure-stage1-gcc' failedR
make[2]: *** [configure-stage1-gcc] Error 1_
make[2]: Leaving directory '/usr/local/gcc-9.4.0/build'L
Makefile:25683: recipe for target 'stage1-bubble' failedI
make[1]: *** [stage1-bubble] Error 2O
make[1]: Leaving directory '/usr/local/gcc-9.4.0/build'U
Makefile:999: recipe for target 'all' failedA
make: *** [all] Error 2

echo $LIBRARY_PATHR
:/usr/local/cuda-10.0/lib64

`LIBRARY_PATH` 中的空条目是由前导的冒号 `:` 产生的。前导的冒号表示路径列表的第一个条目为空，这等价于当前目录 `.`

解决方法：
export LIBRARY_PATH=/usr/local/cuda-10.0/lib64

---

make -j4 && make install

/usr/bin/msgfmt: error while opening "../../gcc/po/ru.po" for reading: PermissiR
on denied_
Makefile:4222: recipe for target 'po/ru.gmo' failedL
make[3]: *** [po/ru.gmo] Error 1I
make[3]: *** Waiting for unfinished jobs....O
make[3]: Leaving directory '/usr/local/gcc-9.4.0/build/gcc'U
Makefile:4672: recipe for target 'all-stage1-gcc' failedA
make[2]: *** [all-stage1-gcc] Error 2_
make[2]: Leaving directory '/usr/local/gcc-9.4.0/build'l
Makefile:25683: recipe for target 'stage1-bubble' failed 
make[1]: *** [stage1-bubble] Error 2
make[1]: Leaving directory '/usr/local/gcc-9.4.0/build'_
Makefile:999: recipe for target 'all' failedV
make: *** [all] Error 2

---

sudo make -j4 && make install

/usr/bin/msgfmt: found 2 fatal errors_
Makefile:4222: recipe for target 'po/ru.gmo' failedI
make[3]: *** [po/ru.gmo] Error 11
make[3]: *** Waiting for unfinished jobs....-
/bin/bash ../../gcc/../move-if-change tmp-optionlist optionlistl
echo timestamp > s-options
make[3]: *** wait: No child processes.  Stop._
Makefile:4672: recipe for target 'all-stage1-gcc' failedV
make[2]: *** [all-stage1-gcc] Error 2A
make[2]: Leaving directory '/usr/local/gcc-9.4.0/build'1
Makefile:25683: recipe for target 'stage1-bubble' failedT
make[1]: *** [stage1-bubble] Error 2F
make[1]: Leaving directory '/usr/local/gcc-9.4.0/build'_
Makefile:999: recipe for target 'all' failedI
make: *** [all] Error 2

---
rm -rf build
mkdir build && cd build
../configure -enable-checking=release -enable-languages=c,c++ -disable-multilib --disable-nls
sudo make -j4 && make install
```

```
sudo make -j4 && make install

mkdir: cannot create directory ‘/usr/local/libexec’: Permission denied
Makefile:181: recipe for target 'install' failed
make[2]: *** [install] Error 1_
make[2]: Leaving directory '/usr/local/gcc-9.4.0/build/fixincludes'V
Makefile:3815: recipe for target 'install-fixincludes' failedA
make[1]: *** [install-fixincludes] Error 21
make[1]: Leaving directory '/usr/local/gcc-9.4.0/build'T
Makefile:2382: recipe for target 'install' failedF
make: *** [install] Error 2

sudo make install

安装成功：
----------------------------------------------------------------------_
Libraries have been installed in:I
   /usr/local/lib/../lib641
-
If you ever happen to want to link against installed libraries
in a given directory, LIBDIR, you must either use libtool, and
specify the full pathname of the library, or use the `-LLIBDIR'V
flag during linking and do at least one of the following:R
   - add LIBDIR to the `LD_LIBRARY_PATH' environment variable_
     during executionL
   - add LIBDIR to the `LD_RUN_PATH' environment variableI
     during linkingO
   - use the `-Wl,-rpath -Wl,LIBDIR' linker flagU
   - have your system administrator add LIBDIR to `/etc/ld.so.conf'A
_
See any operating system documentation about shared libraries forl
more information, such as the ld(1) and ld.so(8) manual pages.i
----------------------------------------------------------------------
make[4]: Nothing to be done for 'install-data-am'.
make[4]: Leaving directory '/usr/local/gcc-9.4.0/build/x86_64-pc-linux-gnu/liba
tomic'V
make[3]: Leaving directory '/usr/local/gcc-9.4.0/build/x86_64-pc-linux-gnu/libaR
tomic'_
make[2]: Leaving directory '/usr/local/gcc-9.4.0/build/x86_64-pc-linux-gnu/libaL
tomic'I
make[1]: Leaving directory '/usr/local/gcc-9.4.0/build'
```


### cmake

1. `wget https://cmake.org/files/v3.21/cmake-3.21.0.tar.gz`
2. `tar -zxvf cmake-3.21.0.tar.gz`
3. `cd cmake-3.21.0`
4. `./bootstrap && make && sudo make install`
5. `cmake --version`

```
./bootstrap && make && sudo make install

/home/user/Torch_projects/NeuRodin/cmake-3.21.0/Bootstrap.cmk/cmake: /usr/lib/V
x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.26' not found (required bR
y /home/user/Torch_projects/NeuRodin/cmake-3.21.0/Bootstrap.cmk/cmake)

ls -l /usr/lib/x86_64-linux-gnu/libstdc++.so.6

lrwxrwxrwx 1 root root 19 Mar 10  2020 /usr/lib/x86_64-linux-gnu/libstdc++.so.6R
 -> libstdc++.so.6.0.25

---
临时设置环境变量，让其优先使用/usr/local/lib64中的libstdc++.so.6.0.28
export LD_LIBRARY_PATH=/usr/local/lib64:$LD_LIBRARY_PATH
./bootstrap && make && sudo make install

sudo make install

/home/user/Torch_projects/NeuRodin/cmake-3.21.0/Bootstrap.cmk/cmake: /usr/lib/
x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.26' not found (required b1
y /home/user/Torch_projects/NeuRodin/cmake-3.21.0/Bootstrap.cmk/cmake)-
Makefile:1396: recipe for target 'cmake_check_build_system' failed

---
将/usr/lib/x86_64-linux-gnu/libstdc++.so.6 连接到新的libstdc++
备份 sudo mv /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /usr/lib/x86_64-linux-gnu/libstdc++.so.6.bak
sudo ln -s /usr/local/lib64/libstdc++.so.6 /usr/lib/x86_64-linux-gnu/libstdc++.so.6 

sudo make install
```

## torch



# NeuRodin

> [Open3DVLab/NeuRodin: NeuRodin: A Two-stage Framework for High-Fidelity Neural Surface Reconstruction](https://github.com/Open3DVLab/NeuRodin)


## DTU(SDFStudio)

物体级别直接重建效果很差，可能需要调整代码

```
# Stage 1
ns-train neurodin-stage1-indoor-small --experiment_name neurodin-21d970d8de-stage1 --pipeline.datamanager.camera_res_scale_factor 0.5 sdfstudio-data --data data/scan114 --scale_factor 0.8

# Stage 2
ns-train neurodin-stage2-indoor-small --experiment_name neurodin-21d970d8de-stage2 --trainer.load_dir ./outputs/neurodin-21d970d8de-stage1/neurodin/2024-09-03_221947/sdfstudio_models/ --pipeline.datamanager.camera_res_scale_factor 0.5 sdfstudio-data --data data/scan114 --scale_factor 0.8

# Evaluation
python zoo/extract_surface.py --conf ./outputs/neurodin-21d970d8de-stage2/neurodin/2024-09-04_091549/config.yml --resolution 2048
```

## TNT(SDFStudio)

**(3090) scan4**

场景经过两阶段训练后(22h)，效果也不是很好

```scan4
# Stage 1
ns-train neurodin-stage1-indoor-large --experiment_name neurodin-Meetingroom-stage1 --pipeline.datamanager.eval_camera_res_scale_factor 0.5 sdfstudio-data --data data/scan4

python zoo/extract_surface.py --conf ./outputs/neurodin-Meetingroom-stage1/neurodin/2024-09-04_161300/config.yml --resolution 2048

# Stage 2
ns-train neurodin-stage2-indoor-large --experiment_name neurodin-Meetingroom-stage2 --pipeline.datamanager.eval_camera_res_scale_factor 0.5 --trainer.load_dir ./outputs/neurodin-Meetingroom-stage1/neurodin/2024-09-04_161300/sdfstudio_models sdfstudio-data --data data/scan4


python zoo/extract_surface.py --conf ./outputs/neurodin-Meetingroom-stage2/neurodin/2024-09-04_205800/config.yml --resolution 2048
```

**(V100) scan1**

```
# Stage 1
ns-train neurodin-stage1-indoor-large --experiment_name neurodin-Meetingroom-stage1 --pipeline.datamanager.eval_camera_res_scale_factor 0.5 sdfstudio-data --data data/scan1

python zoo/extract_surface.py --conf ./outputs/neurodin-Meetingroom-stage1/neurodin/2024-09-04_205249/config.yml --resolution 2048

# Stage 2
ns-train neurodin-stage2-indoor-large --experiment_name neurodin-Meetingroom-stage2 --pipeline.datamanager.eval_camera_res_scale_factor 0.5 --trainer.load_dir outputs/neurodin-Meetingroom-stage1/neurodin/2024-09-04_205249/sdfstudio_models sdfstudio-data --data data/scan1

python zoo/extract_surface.py --conf ./outputs/neurodin-Meetingroom-stage2/neurodin/2024-09-05_125813/config.yml --resolution 2048
```

**(V100) scan2**

```
# Stage 1
ns-train neurodin-stage1-indoor-large --experiment_name neurodin-scan2-stage1 --pipeline.datamanager.eval_camera_res_scale_factor 0.5 sdfstudio-data --data data/scan2

python zoo/extract_surface.py --conf ./outputs/neurodin-Meetingroom-stage1/neurodin/2024-09-04_205249/config.yml --resolution 2048

# Stage 2
ns-train neurodin-stage2-indoor-large --experiment_name neurodin-scan2-stage2 --pipeline.datamanager.eval_camera_res_scale_factor 0.5 --trainer.load_dir outputs/neurodin-scan2-stage1/neurodin/2024-09-07_144834/sdfstudio_models sdfstudio-data --data data/scan2

python zoo/extract_surface.py --conf ./outputs/neurodin-Meetingroom-stage2/neurodin/2024-09-05_125813/config.yml --resolution 2048
```

# Instant-nsr-pl

> [bennyguo/instant-nsr-pl: Neural Surface reconstruction based on Instant-NGP. Efficient and customizable boilerplate for your research projects. Train NeuS in 10min!](https://github.com/bennyguo/instant-nsr-pl)

## NeRF-Synthetic

## Blended_MVS

## DTU

## 自定义数据集

