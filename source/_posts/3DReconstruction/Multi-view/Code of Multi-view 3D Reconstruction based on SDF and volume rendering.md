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
  3. `cmake --build build --config RelWithDebInfo -j` If compilation fails inexplicably or takes longer than an hour, you might be running **out of memory**. Try running the above command without `-j` in that case.
4. 最后运行setup.py，如果不在全局环境安装，亦可在conda虚拟环境中安装
  1. `cd bindings/torch`
  2. `python setup.py install`

in wsl2 with Ubuntu:
> [CUDA Toolkit 11.3 Update 1 Downloads | NVIDIA Developer](https://developer.nvidia.com/cuda-11-3-1-download-archive?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_network)
> [CUDA Toolkit 12.6 Update 1 Downloads | NVIDIA Developer](https://developer.nvidia.com/cuda-12-6-1-download-archive?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_network)
> [Notice: CUDA Linux Repository Key Rotation - Accelerated Computing / Announcements - NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/notice-cuda-linux-repository-key-rotation/212772) key失效了


```bash
# cuda 11.3
sudo apt install gcc
sudo apt install cmake
sudo apt install nvidia-cuda-toolkit # unknown source

# cuda 官网
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600

# 这个key 无法使用了 
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/7fa2af80.pub 
# 换成这个key：
sudo apt-key del 7fa2af80
wget https://developer.download.nvidia.com/compute/cuda/repos/$distro/$arch/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb

# $distro/$arch：
debian10/x86_64
debian11/x86_64
ubuntu1604/x86_64
ubuntu1804/cross-linux-sbsa
ubuntu1804/ppc64el
ubuntu1804/sbsa
ubuntu1804/x86_64
ubuntu2004/cross-linux-sbsa
ubuntu2004/sbsa
ubuntu2004/x86_64
ubuntu2204/sbsa
ubuntu2204/x86_64
wsl-ubuntu/x86_64

sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/ /"
sudo apt-get update
sudo apt-get -y install cuda # 默认安装最高版本
sudo apt-get -y install cuda-toolkit-11-3

# 更高版本 12.6

wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-6
```

```
# cudann 对应cuda版本 11.3
tar -zxvf cudnn-自己补全版本号.tgz
sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda-11.3/lib64/
sudo cp cuda/include/cudnn.h /usr/local/cuda-11.3/include/
 
#为更改读取权限：
sudo chmod a+r /usr/local/cuda-11.3/include/cudnn.h
sudo chmod a+r /usr/local/cuda-11.3/lib64/libcudnn*

## test
cd /usr/local/cuda/samples/4_Finance/BlackScholes
sudo make
./BlackScholes
```

[通过修改软链接升高 gcc 版本、降低 gcc 版本_gcc软连接-CSDN博客](https://blog.csdn.net/wohu1104/article/details/107371779)
[CUDA incompatible with my gcc version - Stack Overflow](https://stackoverflow.com/questions/6622454/cuda-incompatible-with-my-gcc-version) CUDA对应gcc版本

- `sudo apt install gcc-$MAX_GCC_VERSION g++-$MAX_GCC_VERSION`
- `sudo rm gcc g++` in /usr/bin/
- `sudo ln -s gcc-$MAX_GCC_VERSION gcc`
  - `sudo ln -s /usr/bin/gcc-$MAX_GCC_VERSION /usr/local/cuda/bin/gcc `
  - `sudo ln -s /usr/bin/g++-$MAX_GCC_VERSION /usr/local/cuda/bin/g++`


### gcc install/update

简单方法：[linux - 【Ubuntu20.04】安装gcc11 g++11, Ubuntu18.04 - 个人文章 - SegmentFault 思否](https://segmentfault.com/a/1190000044587299)

原始方法：
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

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## pytorch3d

使用pip安装torch与使用conda安装torch不同
>  [全面总结 pip install 与 conda install 的使用区别_pytorch安装pip和conda的区别-CSDN博客](https://blog.csdn.net/whc18858/article/details/127135973)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
conda install pytorch=2.4.1 torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y
conda install -c iopath iopath
conda install pytorch3d -c pytorch3d # 会报错
# 直接从github安装 😊
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d
pip install -e .
```


> [pytorch3d/INSTALL.md at main · facebookresearch/pytorch3d](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)

```bash
# 行不通😵
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install pytorch3d=0.7.5 -c pytorch3d # 可能torch版本太高了

# torch 2.4.1 使用预编译的pytorch3d 行不通😵
pip install torch==2.4.1
pip install --extra-index-url https://miropsota.github.io/torch_packages_builder pytorch3d==0.7.8+pt2.4.1cu121
```

> [ImportError: libtorch_cuda_cu.so: cannot open shared object file · Issue #438 · open-mmlab/mmdetection3d](https://github.com/open-mmlab/mmdetection3d/issues/438) 不是pytorch3d的问题
> [Prebuilt wheels provided via 3rd party repository · facebookresearch/pytorch3d · Discussion #1752](https://github.com/facebookresearch/pytorch3d/discussions/1752) 用大佬编译好的pytorch3d


## visual in Jupyter

### pyvista

`pip install ipywidgets widgetsnbextension pandas-profiling`

> [PyVista + Trame = Jupyter 3D Visualization - Announcements - VTK](https://discourse.vtk.org/t/pyvista-trame-jupyter-3d-visualization/10610)

`pip install 'pyvista[jupyter]>=0.38.1'`

>[apt - Issues installing libgl1-mesa-glx - Ask Ubuntu](https://askubuntu.com/questions/1517352/issues-installing-libgl1-mesa-glx)

this solve my problem in Ubuntu 24.04LTS libgl1-mesa-glx_23.0.4-0ubuntu1.22.04.1_amd64.deb DOWNLOAD https://github.com/PetrusNoleto/Error-in-install-cisco-packet-tracer-in-ubuntu-23.10-unmet-dependencies/releases/tag/CiscoPacketTracerFixUnmetDependenciesUbuntu23.10

exec :

`sudo dpkg -i libgl1-mesa-glx_23.0.4-0ubuntu1.22.04.1_amd64.deb`


> [MESA and glx errors when running glxinfo Ubuntu 24.04 - Ask Ubuntu](https://askubuntu.com/questions/1516040/mesa-and-glx-errors-when-running-glxinfo-ubuntu-24-04)

MESA: error: ZINK: failed to choose pdev glx: failed to create drisw screen

Kisak-mesa PPA 提供了 Mesa 的最新小版本。您可以通过在终端中逐个输入以下命令来使用它：

```bash
sudo add-apt-repository ppa:kisak/kisak-mesa
sudo apt update
sudo apt upgrade
```

## ipywidgets

```bash
ImportError: IProgress not found. Please update jupyter and ipywidgets. See [https://ipywidgets.readthedocs.io/en/stable/user_install.html](https://ipywidgets.readthedocs.io/en/stable/user_install.html)
```

[pandas - ImportError: IProgress not found. Please update jupyter and ipywidgets although it is installed - Stack Overflow](https://stackoverflow.com/questions/67998191/importerror-iprogress-not-found-please-update-jupyter-and-ipywidgets-although)


```bash
pip install jupyter
pip install ipywidgets widgetsnbextension pandas-profiling
```


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
# Geo-Neus (AutoDL long time)

> [GhiXu/Geo-Neus: Geo-Neus: Geometry-Consistent Neural Implicit Surfaces Learning for Multi-view Reconstruction (NeurIPS 2022)](https://github.com/GhiXu/Geo-Neus)

```bash
git clone https://github.com/GhiXu/Geo-Neus.git
conda create -n geoneus python=3.7  
conda activate geoneus  
conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.3 -c pytorch  
conda install fvcore iopath  
conda install -c bottler nvidiacub  
conda install pytorch3d -c pytorch3d  
pip install -r requirements.txt
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install tensorboard tqdm pytorch3d pickle5

git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d
pip install -e .

unzip DTU.zip -d ./data/
```

```python
# 需要修改 womask.conf 中的 data_dir = ./data/DTU/CASE_NAME
import os

#106 #105 #97 #83 #69 #65 #63 #55 #40 #37 #24
dtu_scenes = ['scan24', 'scan37', 'scan40', 'scan55', 'scan63', 'scan65', 'scan69', 'scan83', 'scan97', 'scan105', 'scan106', 'scan110', 'scan114', 'scan118', 'scan122']

for scene in dtu_scenes:
    os.system(f"python exp_runner.py --mode train --conf ./confs/womask.conf --case {scene}")

刚开始跑scan69，中期来不及了，先停掉
```

# MonoSDF

Given Mesh!!! Nice Author!!!

# NeuS2 (wsl)

- [x] Given weight!!! Test --> no background (with mask)

## 环境配置

```bash
git clone --recursive https://github.com/19reborn/NeuS2
cd NeuS2

git submodule update --init --recursive
cmake . -D TCNN_CUDA_ARCHITECTURES=86 -D CMAKE_CUDA_COMPILER=$(which nvcc) -B build
cmake --build build --config RelWithDebInfo -j

conda create -n neus2 python=3.9
conda activate neus2
pip install -r requirements.txt

conda install -c conda-forge gcc=12.1.0
pip install commentjson imageio scipy trimesh termcolor 

git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d
pip install -e . 'OR' python3 setup.py install
```

```bash error
Compiling the CUDA compiler identification source file
"CMakeCUDACompilerId.cu" failed.

Compiler: /usr/bin/nvcc

Build flags:

Id flags: --keep;--keep-dir;tmp -v

解决方法：cmake . -D TCNN_CUDA_ARCHITECTURES=86 -D CMAKE_CUDA_COMPILER=$(which nvcc) -B build
```


Pytorch3d一直安装不上：

```bash 
#pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
#pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124

pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d
pip install -e . 'OR' python3 setup.py install
# 不知道为啥不行不行着就装上了😵    command = ['ninja', '-v']改了一下这个，又该回去
# command = ['ninja', '--version']

# conda install -c iopath iopath
# pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu121_pyt210/download.html
```

```bash error
ImportError: /home/qi/miniconda3/envs/neus2/lib/python3.9/site-packages/numpy/_core/../../../../libstdc++.so.6: version `GLIBCXX_3.4.30' not found (required by /home/qi/Project/NeuS2/build/pyngp.cpython-39-x86_64-linux-gnu.so)

pip install --upgrade numpy

ImportError: /home/qi/miniconda3/envs/neus2/bin/../lib/libstdc++.so.6: version `GLIBCXX_3.4.30' not found (required by /home/qi/Project/NeuS2/build/pyngp.cpython-39-x86_64-linux-gnu.so)

# 真正解法：替换为高版本的 libstdc++.so.6.0.33
rm /home/qi/miniconda3/envs/neus2/bin/../lib/libstdc++.so.6.0.33
cp /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.33 /home/qi/miniconda3/envs/neus2/bin/../lib
ln -s /home/qi/miniconda3/envs/neus2/bin/../lib/libstdc++.so.6.0.33 /home/qi/miniconda3/envs/neus2/bin/../lib/libstdc++.so.6
strings /home/qi/miniconda3/envs/neus2/bin/../lib/libstdc++.so.6 | grep GLIBCXX_3.4.29
```

## 运行

```
```bash
python scripts/run.py --scene ${data_path}/transform.json --name ${your_experiment_name} --network ${config_name} --n_steps ${training_steps}

python scripts/run.py --test --save_mesh --load_snapshot ./dtu_neus2_release/checkpoints/scan24.msgpack --network dtu.json --name scan24
```

# COLMAP (Known camera pose)

## 环境配置

> [Installation — COLMAP 3.11.0.dev0 documentation](https://colmap.github.io/install.html#linux)

```bash
sudo apt-get install \
    git \
    cmake \
    ninja-build \
    build-essential \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libgmock-dev \
    libsqlite3-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libceres-dev


sudo apt-get install -y \
    nvidia-cuda-toolkit \
    nvidia-cuda-toolkit-gcc


git clone https://github.com/colmap/colmap.git
cd colmap
mkdir build
cd build
cmake .. -GNinja # -DCMAKE_CUDA_ARCHITECTURES=native 代表只在本机上运行
ninja
sudo ninja install

# cmake .. -GNinja时 wsl报错"CMakeCUDACompilerId.cu" failed. 解决方法：
cmake .. -GNinja -D TCNN_CUDA_ARCHITECTURES=86 -D CMAKE_CUDA_COMPILER=$(which nvcc) -DCMAKE_CUDA_ARCHITECTURES=native

# Ubuntu 22.04 还需要：
sudo apt-get install gcc-10 g++-10
export CC=/usr/bin/gcc-10
export CXX=/usr/bin/g++-10
export CUDAHOSTCXX=/usr/bin/g++-10
```

## 已知相机参数的重建

[COLMAP已知相机内外参数重建稀疏/稠密模型 - thronsbird - 博客园](https://www.cnblogs.com/li-minghao/p/11865794.html)

稀疏重建：
- feature_extractor
- 手动或者代码生成 cameras.txt images.txt和points3D.txt文件，并将其导入db文件中
- exhaustive_matcher
- point_triangulator
- bundle_adjuster

稠密重建：
- image_undistorter 指定目录为dense时：
  - 在 **dense/sparse/** 目录下的.bin文件的内容与之前建立的.txt文件内容相同。在 **dense/stereo/** 目录下的 **patch-match.cfg** 规定了源图像进行块匹配的参考图像，默认的`__auto__, 20`代表自动最优的20张，必须要进行了point_triangulator才可用。或者：`__all__`指定所有图像为参考图像，`image001.jpg, image003.jpg, image004.jpg, image007.jpg`手动指定参考图像
- patch_match_stereo
  - 如果同样没有稀疏点云，则需要根据场景手动指定最小和最大深度：--PatchMatchStereo.depth_min 0.0 --PatchMatchStereo.depth_max 20.0
- stereo_fusion
- poisson_mesher

***如果相机位姿导入的时候不正确，则稀疏重建point_triangulator的点云数量很少，结果很差***

![image.png|444](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20241117040617.png)

DTU数据集中给的矩阵为pixel2world矩阵：

```python
2607.429996 -3.844898 1498.178098 -533936.661373
-192.076910 2862.552532 681.798177 23434.686572
-0.241605 -0.030951 0.969881 22.540121
```

```python
camera_poses = []
for i in range(64):
  poses = []
  pose_file = os.path.join(pose_dir, f"pos_{i+1:03d}.txt")
  # print(pose_file)
  with open(pose_file, 'r') as f:
      pose = f.readlines()
  for l in pose:
      l = l.strip().split()
      poses.append([float(x) for x in l])
  poses_np = np.array(poses)
  P = poses_np
  # pose is the c2w to COLMAP
  intrinsics, pose = load_K_Rt_from_P(P)
  camera_poses.append(pose)
camera_poses = np.array(camera_poses)
np.savez("xxx.npz", camera_poses=camera_poses)
```


# Mine 

## Accuracy

多类几何先验混合监督的方法： Depth | Normal | SFM points

### Data Generation

Colmap 生成点云的时候可以使用 GT camera pose（cameras.npz） 进行监督？

>Method1:[DTU camera Poses · Issue #5 · hbb1/2d-gaussian-splatting](https://github.com/hbb1/2d-gaussian-splatting/issues/5)  https://github.com/NVlabs/neuralangelo/blob/main/projects/neuralangelo/scripts/convert_tnt_to_json.py
>Method2: [How to run COLMAP with ground truth camera poses on DTU? · Issue #20 · dunbar12138/DSNeRF](https://github.com/dunbar12138/DSNeRF/issues/20) [Frequently Asked Questions — COLMAP 3.11.0.dev0 documentation](https://colmap.github.io/faq.html#reconstruct-sparse-dense-model-from-known-camera-poses)

### RUN

Geo-Neus的数据集，相机位姿和world坐标系下的点云与DTU数据集中的一致，可以评价CD指标：

```bash
# no cue
python run.py --conf confs/neus-dtu_geo.yaml --train dataset.root_dir="scene_dir"
# with depth
python run.py --conf confs/neus-dtu_geo.yaml --train dataset.root_dir="scene_dir" tag='depth' dataset.apply_depth=True
# with normal
python run.py --conf confs/neus-dtu_geo.yaml --train dataset.root_dir="scene_dir" tag='normal' dataset.apply_normal=True
# with sfm points
python run.py --conf confs/neus-dtu_geo.yaml --train dataset.root_dir="scene_dir" tag='sfm' dataset.apply_sfm=True
```

dtu_like自定义数据集，colmap生成的点云坐标系可能与DTU的GT点云坐标系不同(**相机位姿也不同**)，因此无法评价CD指标：

```bash
# no cue
python run.py --conf confs/neus-dtu.yaml --train dataset.root_dir="scene_dir" dataset.name='dtu_like'
# with depth
python run.py --conf confs/neus-dtu.yaml --train dataset.root_dir="scene_dir" dataset.name='dtu_like' tag='depth' dataset.apply_depth=True
# with normal
python run.py --conf confs/neus-dtu.yaml --train dataset.root_dir="scene_dir" dataset.name='dtu_like' tag='normal' dataset.apply_normal=True
# with sfm points
python run.py --conf confs/neus-dtu.yaml --train dataset.root_dir="scene_dir" dataset.name='dtu_like' tag='sfm' dataset.apply_sfm=True
```

### Comparison

对比启用不同先验：
- no cue
- with depth
- with normal
- with sfm points
- with all

不同深度先验：
- [DepthAnything/Depth-Anything-V2: \[NeurIPS 2024\] Depth Anything V2. A More Capable Foundation Model for Monocular Depth Estimation](https://github.com/DepthAnything/Depth-Anything-V2)
- [LiheYoung/Depth-Anything: \[CVPR 2024\] Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data. Foundation Model for Monocular Depth Estimation](https://github.com/LiheYoung/Depth-Anything)
- [apple/ml-depth-pro: Depth Pro: Sharp Monocular Metric Depth in Less Than a Second.](https://github.com/apple/ml-depth-pro)
- [VisualComputingInstitute/diffusion-e2e-ft: Fine-Tuning Image-Conditional Diffusion Models is Easier than You Think](https://github.com/VisualComputingInstitute/diffusion-e2e-ft)
- [noahzn/Lite-Mono: \[CVPR2023\] Lite-Mono: A Lightweight CNN and Transformer Architecture for Self-Supervised Monocular Depth Estimation](https://github.com/noahzn/Lite-Mono)
- [shariqfarooq123/AdaBins: Official implementation of Adabins: Depth Estimation using adaptive bins](https://github.com/shariqfarooq123/AdaBins)
- [SysCV/P3Depth](https://github.com/SysCV/P3Depth)

不同法向量先验：
- [Stable-X/StableNormal: \[SIGGRAPH Asia 2024 (Journal Track)\] StableNormal: Reducing Diffusion Variance for Stable and Sharp Normal](https://github.com/Stable-X/StableNormal)
- [VisualComputingInstitute/diffusion-e2e-ft: Fine-Tuning Image-Conditional Diffusion Models is Easier than You Think](https://github.com/VisualComputingInstitute/diffusion-e2e-ft)
- [EPFL-VILAB/omnidata: A Scalable Pipeline for Making Steerable Multi-Task Mid-Level Vision Datasets from 3D Scans \[ICCV 2021\]](https://github.com/EPFL-VILAB/omnidata) 图片尺寸限制
- [baegwangbin/DSINE: \[CVPR 2024 Oral\] Rethinking Inductive Biases for Surface Normal Estimation](https://github.com/baegwangbin/DSINE)
- [YvanYin/Metric3D: The repo for "Metric3D: Towards Zero-shot Metric 3D Prediction from A Single Image" and "Metric3Dv2: A Versatile Monocular Geometric Foundation Model..."](https://github.com/yvanyin/metric3d)


## Efficiency

```bash
# lmc
python run.py --conf confs/neus-dtu_geo.yaml --train dataset.root_dir="scene_dir" dataset.sampling_type='lmc'
# uniform
python run.py --conf confs/neus-dtu_geo.yaml --train dataset.root_dir="scene_dir" dataset.sampling_type='uniform'
```

## Uncertainty

