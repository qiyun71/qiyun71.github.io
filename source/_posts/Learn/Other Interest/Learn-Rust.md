---
title: Learn-Rust
date: 2024-10-25 20:08:27
tags:
  - 
categories: Learn/Other Interest
---

Start from:
- [安装 Rust 环境 - Rust语言圣经(Rust Course)](https://course.rs/first-try/installation.html)
- [安装 Rust - Rust 程序设计语言](https://www.rust-lang.org/zh-CN/tools/install)

<!-- more -->

# windows

**1. `x86_64-pc-windows-msvc`（官方推荐）**

先安装 [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/zh-hans/visual-cpp-build-tools/)，勾选安装 C++ 环境即可。安装时可自行修改缓存路径与安装路径，避免占用过多 C 盘空间。安装完成后，Rust 所需的 msvc 命令行程序需要手动添加到环境变量中，否则安装 Rust 时 `rustup-init` 会提示未安装 Microsoft C++ Build Tools，其位于：`%Visual Studio 安装位置%\VC\Tools\MSVC\%version%\bin\Hostx64\x64`（请自行替换其中的 %Visual Studio 安装位置%、%version% 字段）下。

如果你不想这么做，可以选择安装 Microsoft C++ Build Tools 新增的“定制”终端 `Developer Command Prompt for %Visual Studio version%` 或 `Developer PowerShell for %Visual Studio version%`，在其中运行 `rustup-init.exe`。

准备好 C++ 环境后开始安装 Rust：

在 [RUSTUP-INIT](https://www.rust-lang.org/learn/get-started) 下载系统相对应的 Rust 安装程序，一路默认即可。

`PS C:\Users\Hehongyuan> rustup-init.exe ...... Current installation options:     default host triple: x86_64-pc-windows-msvc      default toolchain: stable (default)                profile: default   modify PATH variable: yes  1) Proceed with installation (default) 2) Customize installation 3) Cancel installation`

## 常用命令

```bash
# udpate
rustup update
```

# Proj

## 3DGS

>[ArthurBrussee/brush: 3D Reconstruction for all](https://github.com/ArthurBrussee/brush?tab=readme-ov-file)Brush is a 3D reconstruction engine, using [Gaussian splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/). It aims to be highly portable, flexible and fast.

`git clone https://github.com/ArthurBrussee/brush.git`

Install rust 1.81+ and run `cargo run` or `cargo run --release`. You can run tests with `cargo test --all`. Brush uses the wonderful [rerun](https://github.com/ArthurBrussee/brush/blob/main/rerun.io) for additional visualizations while training. It currently requires rerun 0.19 however, which isn't released yet.

![image.png|444](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20241028144213.png)

## GUI

[emilk/egui: egui: an easy-to-use immediate mode GUI in Rust that runs on both web and native](https://github.com/emilk/egui)