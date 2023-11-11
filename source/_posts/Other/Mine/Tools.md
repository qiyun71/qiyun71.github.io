# WSL2

移动到其他盘[WSL2安装Ubuntu20.04 - 王谷雨 - 博客园 (cnblogs.com)](https://www.cnblogs.com/konghuanxi/p/14731846.html)

```powershell
# 查看当前wsl安装的虚拟机
wsl -l -v

# 关闭所有正在运行的虚拟机
wsl --shutdown

# 对需要迁移的分发或虚拟机导出
wsl --export 虚拟机名称Ubuntu 文件导出路径D:\Ubuntu.tar

# 卸载虚拟机（删除C盘的虚拟机数据）
wsl --unregister 虚拟机名称Ubuntu

# 导入新的虚拟机
wsl --import 虚拟机名称Ubuntu 目标路径D:\WSL\Ubuntu 虚拟机文件路径D:\Ubuntu.tar --version 2
# 目标路径就是想要将虚拟机迁移到的具体位置
```

[WSL文件存储位置迁移 - 掘金 (juejin.cn)](https://juejin.cn/post/7284962800668475455)
迁移后还需要切换默认用户`虚拟机名称Ubuntu config --default-user <username>`，Ubuntu为D盘中

## 硬件配置

修改内存和swap大小`C:\Users\Qiyun`下新建`/.wslconfig`文件
[Advanced settings configuration in WSL | Microsoft Learn](https://learn.microsoft.com/en-us/windows/wsl/wsl-config#configuration-setting-for-wslconfig)

```
[wsl2]
memory=16GB
swap=8GB
```

查看linux内存`free -h`

## 系统指令

查看 Ubuntu 版本`lsb_release -a`

### shell脚本编写

[Linux shell 脚本中， $@ 和$# 分别是什么意思？-CSDN博客](https://blog.csdn.net/qiezikuaichuan/article/details/80014298)

## Python&Conda

安装pycuda时，gcc一直找不到lcuda：添加软连接

`sudo ln -s /usr/local/cuda/lib64/libcuda.so /usr/lib/libcuda.so`
`sudo ln -s /usr/local/cuda/lib64/libcuda.so /home/yq/miniconda3/envs/retr/lib/libcuda.so`


安装带cuda版本的Pytorch：[Previous PyTorch Versions | PyTorch](https://pytorch.org/get-started/previous-versions/)
> [高版本CUDA能否安装低版本PYTORCH？ - 知乎 (zhihu.com)](https://www.zhihu.com/question/450523810)


# Google搜索技巧

- 限定关键词`"关键词"`，只返回特定的搜索结果（关键词完整出现）
- 标题`intitle:关键词`
  - 多个关键词`allintitle:关键词1 关键词2 ...`
- 文章内容`intext:关键词`
- 特定网站`inurl:关键词`，`site:网站域名` （网站内置搜索很差时）
- 搜索图片`imagesize:2560x1440`
- 文件格式`filetype:pdf等格式`

# Github

## 笔记

思维导图，免费开源
[wanglin2/mind-map: 一个还算强大的Web思维导图。A relatively powerful web mind map. (github.com)](https://github.com/wanglin2/mind-map)

使用 ChatGPT 自动将 Markdown 文件批量翻译为多语言
[linyuxuanlin/Auto-i18n: 使用 ChatGPT 自动将 Markdown 文件批量翻译为多语言 | Auto translate Markdown files to multi languages using ChatGPT (github.com)](https://github.com/linyuxuanlin/Auto-i18n)

### 书籍/经验 tips

快速回忆 python 知识
[gto76/python-cheatsheet: Comprehensive Python Cheatsheet (github.com)](https://github.com/gto76/python-cheatsheet)

[d2l-ai/d2l-zh: 《动手学深度学习》：面向中文读者、能运行、可讨论。中英文版被70多个国家的500多所大学用于教学。 (github.com)](https://github.com/d2l-ai/d2l-zh)

[scutan90/DeepLearning-500-questions: 深度学习500问，以问答形式对常用的概率知识、线性代数、机器学习、深度学习、计算机视觉等热点问题进行阐述](https://github.com/scutan90/DeepLearning-500-questions)

[chenyuntc/pytorch-book: PyTorch tutorials and fun projects including neural talk, neural style, poem writing, anime generation (《深度学习框架PyTorch：入门与实战》) (github.com)](https://github.com/chenyuntc/pytorch-book)

[amusi/AI-Job-Notes: AI算法岗求职攻略（涵盖准备攻略、刷题指南、内推和AI公司清单等资料） (github.com)](https://github.com/amusi/AI-Job-Notes)

[krahets/hello-algo: 《Hello 算法》：动画图解、一键运行的数据结构与算法教程，支持 Java, C++, Python, Go, JS, TS, C#, Swift, Rust, Dart, Zig 等语言。 (github.com)](https://github.com/krahets/hello-algo)

[PKUFlyingPig/cs-self-learning: 计算机自学指南 (github.com)](https://github.com/PKUFlyingPig/cs-self-learning)

[QianMo/Real-Time-Rendering-3rd-CN-Summary-Ebook: :blue_book: 电子书 -《Real-Time Rendering 3rd》提炼总结 | 全书共9万7千余字](https://github.com/QianMo/Real-Time-Rendering-3rd-CN-Summary-Ebook)

[pengsida/learning_research: 本人的科研经验 (github.com)](https://github.com/pengsida/learning_research)

[jbhuang0604/awesome-tips (github.com)](https://github.com/jbhuang0604/awesome-tips)

快速了解一些网络结构的部署 (代码+描述)
[labmlai/annotated_deep_learning_paper_implementations](https://github.com/labmlai/annotated_deep_learning_paper_implementations)

李沐精读论文
[mli/paper-reading: 深度学习经典、新论文逐段精读 (github.com)](https://github.com/mli/paper-reading)

模型调参
[google-research/tuning_playbook: A playbook for systematically maximizing the performance of deep learning models. (github.com)](https://github.com/google-research/tuning_playbook)

嵌入式
[zhengnianli/EmbedSummary: 精品嵌入式资源汇总 (github.com)](https://github.com/zhengnianli/EmbedSummary)

### 字体

[TrionesType/zhuque: 朱雀仿宋/朱雀宋朝/Zhuque Fangsong: An open-source Fansong typeface project (github.com)](https://github.com/TrionesType/zhuque)

### 绘图

NeuralNet 图片绘制
[HarisIqbal88/PlotNeuralNet: Latex code for making neural networks diagrams (github.com)](https://github.com/HarisIqbal88/PlotNeuralNet)

### 表格

[ivankokan/Excel2LaTeX: The Excel add-in for creating LaTeX tables (github.com)](https://github.com/ivankokan/Excel2LaTeX)

## 图像处理

图片背景移除
[danielgatis/rembg: Rembg is a tool to remove images background (github.com)](https://github.com/danielgatis/rembg)

图片物体分割
[facebookresearch/segment-anything: The repository provides code for running inference with the SegmentAnything Model (SAM)](https://github.com/facebookresearch/segment-anything)

ChatGPT + 图像处理
[microsoft/TaskMatrix (github.com)](https://github.com/microsoft/TaskMatrix)

DragGAN
[XingangPan/DragGAN: Official Code for DragGAN (SIGGRAPH 2023) (github.com)](https://github.com/XingangPan/DragGAN)

## 可视化

相机外参可视化
[demul/extrinsic2pyramid: Visualize Camera's Pose Using Extrinsic Parameter by Plotting Pyramid Model on 3D Space (github.com)](https://github.com/demul/extrinsic2pyramid)

3D Gaussian Splatting Viewer (WebGL)
[antimatter15/splat: WebGL 3D Gaussian Splat Viewer (github.com)](https://github.com/antimatter15/splat)

InstantNGP 所用的 GUI
[ocornut/imgui: Dear ImGui: Bloat-free Graphical User interface for C++ with minimal dependencies (github.com)](https://github.com/ocornut/imgui)

WSL GUI
[microsoft/wslg: Enabling the Windows Subsystem for Linux to include support for Wayland and X server related scenarios (github.com)](https://github.com/microsoft/wslg)

Paper2gui，一款 windows 应用，包含很多 AI 工具（with GUI）
[Baiyuetribe/paper2gui: Convert AI papers to GUI，Make it easy and convenient for everyone to use artificial intelligence technology。让每个人都简单方便的使用前沿人工智能技术 (github.com)](https://github.com/Baiyuetribe/paper2gui)

# 软件

卸载 Windows 默认的应用程序
[PyDebloatX](https://pydebloatx.com/)

三维重建开源免费软件 Meshroom
[alicevision/Meshroom: 3D Reconstruction Software (github.com)](https://github.com/alicevision/Meshroom)

阅读器
[troyeguo/koodo-reader: A modern ebook manager and reader with sync and backup capacities for Windows, macOS, Linux and Web (github.com)](https://github.com/troyeguo/koodo-reader)

音乐播放器
[HyPlayer/HyPlayer: 仅供学习交流使用 | 第三方网易云音乐播放器 | A Netease Cloud Music Player (github.com)](https://github.com/HyPlayer/HyPlayer)

浏览器
[Mzying2001/CefFlashBrowser: Flash浏览器 / Flash Browser (github.com)](https://github.com/Mzying2001/CefFlashBrowser)

碧蓝航线脚本
[LmeSzinc/AzurLaneAutoScript: Azur Lane bot (CN/EN/JP/TW) 碧蓝航线脚本 | 无缝委托科研，全自动大世界 (github.com)](https://github.com/LmeSzinc/AzurLaneAutoScript)

E-Viewer
[OpportunityLiu/E-Viewer: An UWP Client for https://e-hentai.org. (github.com)](https://github.com/OpportunityLiu/E-Viewer)

PowerToys
[microsoft/PowerToys: Windows system utilities to maximize productivity (github.com)](https://github.com/microsoft/PowerToys)

Scrcpy 连接安卓和电脑
[Genymobile/scrcpy: Display and control your Android device (github.com)](https://github.com/Genymobile/scrcpy)

手机应用去广告
[gkd-kit/gkd: 基于 无障碍 + 高级选择器 + 订阅规则 的自定义屏幕点击 Android APP (github.com)](https://github.com/gkd-kit/gkd)

一款开源的跨平台文件传送软件，不需要互联网连接，依靠共享 Wifi 分享文件
[LocalSend](https://localsend.org/#/)

# Blog

[huangshiyu13/webtemplate: 收集各种网站前端模板 (github.com)](https://github.com/huangshiyu13/webtemplate)

# 装机 2023.11.11

## 配置

| 配件 | 型号                           | 价格    |
| ---- | ------------------------------ | ------- |
| CPU  | i5-13490f                      | 2414.17 |
| 主板 | B760M 天选 wifi d5             | ↑       |
| 内存 | 威刚D500 DDR5 32G(16x2) 6400   | 778     |
| 显卡 | 铭瑄4060Ti 16g 小瑷珈          | 3199    |
| 电源 | 安钛克NE750白金牌              | 528.5   |
| 硬盘 | 致态 2TB TiPlus7100 Gen4       | 796.5   |
| 散热 | 利民PA 120SE 白6x6MM双塔青春版 | 127.98  |
| 机箱 | 冰立方ap201                    | 376.16  |
| 风扇 | 玩嘉棱镜4 正3 反2              | 147.6   |
| 总价 |                                | 8367.91        |

## 硬件

[硬件茶谈装机教程](https://www.bilibili.com/video/BV1BG4y137mG)

![349255d6b6dc1979c044b895916608c.jpg|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/349255d6b6dc1979c044b895916608c.jpg)

## 系统

[微PE辅助安装_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1DJ411D79y?p=2&vd_source=1dba7493016a36a32b27a14ed2891088)

## 其他软件

### V2rayN

V2rayN 用户自定义 PAC 设置 Version3.2.9

```v2ray
||github.com,
||google.com,
||chat.openai.com,
||huggingface.co,
||github.io,
||ip138.com,
||paperswithcode.com,
||kaggle.com,
||arxiv.org,
||motorica.ai,
||www.nerfacc.com,
||*.docs.*,
||alicevision.org,
||plotly.com,
||wallhaven.cc,
||ai.meta.com,
||dl.fbaipublicfiles.com,
||awesome-selfhosted.net,
||aka.ms,
||plotly.com,
||trimesh.org.||ten24.info,
||cn.noteai.com,
||ipinfo.io,
||web.stanford.edu,
||awesomeopensource.com,
||*.thecvf.com,
||fnzhan.com,
||skybox.blockadelabs.com,
||www.lumalabs.ai,
||www.ahhhhfs.com,
||www.dropbox.com,
||groups.csail.mit.edu,
||suezjiang.github.io,
||www.arxivdaily.com,
||zero123.cs.columbia.edu,
||happy.5ge.net,
||erikwernquist.com,
||brilliant.org,
||alexyu.net/plenoctrees,
||chenhsuanlin.bitbucket.io,
||discord.com,
||pyflo.net,
||cin.cx
```

V2rayN Version6.29

> [路由规则设定方法 · V2Ray 配置指南|V2Ray 白话文教程 (toutyrater.github.io)](https://toutyrater.github.io/routing/configurate_rules.html)
>软件：[2dust/v2rayN: A GUI client for Windows, support Xray core and v2fly core and others (github.com)](https://github.com/2dust/v2rayN)

路由设置(分流)：
- 全局代理：所有网站都走代理
- GFWList 黑名单模式：黑名单下的国外网站代理，其他直连
  - 坏处：不在黑名单的国外网站也直连，可能无法访问
- ChinaList 白名单模式：白名单下的国内网站直连，其他代理
  - 坏处：不在白名单的国内网站也走代理，可能访问很慢
  - 解决：在系统代理设置中添加国内网站

### PT站

校园网IPV6免费流量

- 北洋园
- 北邮人