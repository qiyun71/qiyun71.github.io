```
conda create -n name python=3.xx
conda remove -n name --all
conda rename -n name new_name

conda activate name
conda deactivate name
```

# 本地conda环境(主机win11)

## 三维重建

Diffusion-TS: 时序信号生成 https://github.com/Y-debug-sys/Diffusion-TS  
SuperRes：
sccs: SC-CS 变形高斯，环境torch.func有问题
sugar: SuGaR | CUDA 11.8
gs: gaussian-splatting | python 3.7.13, torch 1.12.1, CUDA 11.6
instantangelo: Instant-angelo-main | python 3.8.18, torch 2.1.1+cu121, CUDA 12.3
mine: NeRF-Mine |  python 3.8.18, torch 1.12.1+cu113, CUDA 11.3
sdfstudio: SDFStudio | python 3.8.18, torch 1.12.1+cu113, CUDA 11.3
voxurf/voxurf113: voxurf，环境一直不对
clone mine:
  innsr: old_version\instantNGP成功了，instant-nsr-pl 编译通过不了



dtu_eval: 评估CD的程序

depth_anything: Depth Anything项目

## 模型修正
satellite卫星
massspring质量弹簧系统

## other
npyviewer: Other_Proj\NPYViewer 查看npy文件
wopas: web operate scripts 保留autodl实例
plot: ...
dmodel: https://github.com/NVlabs/nvdiffrec
dm: diffusion model learn
latexocr: 识别latex公式
wechatAI: 微信接入GPT模型
gpt:学术gpt项目


# conda 清华源

vi  ~/.condarc
修改为以下内容
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch-lts: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  simpleitk: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud

or :

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
  
## 目前(win11: C:\\Users\\Qiyun\\.condarc)
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

ssl_verify: false

proxy_servers:
  http: http://127.0.0.1:10809
  https: http://127.0.0.1:10809
