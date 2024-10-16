---
title: (WX) Dynamics experiment
date: 2024-09-25 19:29:24
tags:
  - 
categories: Blog&Book&Paper/Read/Papers
---

[王兴 | 中山大学航空航天学院](https://saa.sysu.edu.cn/teacher/347)

**研究领域**

- 航天器在轨组装的地面动力学试验理论与技术
- 高精高稳航天器平台微振动分析方法及抑制技术
- 飞行器结构动力学试验方法与有限元模型修正理论
- 非线性结构模态试验理论与技术

- 每年招收博士研究生1~2人（需有较好的结构动力学基础）

<!-- more -->

# 动力学试验方向

## MsIFFT

[A Multi-step Interpolated-FFT procedure for full-field nonlinear modal testing of turbomachinery components - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0888327021010852)

Model updating for lightweight structures featuring geometrical nonlinearities, its low mass-to-area ratio, complex spatial deformation shapes, and geometrically nonlinear behaviours.

现有非线性结构的 full-field 测量方法主要针对flat、small-scale 和学术结构(beams or plates)
本文基于Three-Dimensional Scanning Laser Doppler Vibrometry (**3D SLDV**)，在结构共振时测量其full-field multi-harmonic operating deflection shapes：
- a **super-short sampling interval** is used for each scan point to achieve a significant reduction in measurement duration. 使用超短的采样区间让采样的数据更有意义
- A novel Multi-step Interpolated-Fast Fourier Transform (**Multi-step Interpolated-FFT**) procedure is proposed to refine the coarse frequency resolution and suppress the severe spectral leakage of the signal spectra. 解决频谱泄露问题
  - the instantaneous瞬时 driving frequency is first interpolated using the force signal and then used to perform a fixed-frequency interpolation for each harmonic of the response signals. 可以准确估计测量信号中的谐波频率、振幅和相位

结构动力学方程：$\boldsymbol{M\ddot{u}}+\boldsymbol{C\dot{u}}+\boldsymbol{Ku}+\boldsymbol{f}_{nl}(\boldsymbol{u},\boldsymbol{\dot{u}}){=\boldsymbol{p}(t)}$
- the nonlinear restoring force $\boldsymbol{f}_{nl}(\boldsymbol{u},\boldsymbol{\dot{u}})$
- the external excitation $\boldsymbol{p}(t)$ applied to the structure during testing 
  - nonlinear phase resonance testing is via a MISO vibration controller, leading to a single point, mono-harmonic force $p(t)=A_p\cos(2\pi f_pt+\varphi_p)$

the vibration of a scan point can be expressed as: 
$u_{(dir)}=\sum_{m=1}^{N_\kappa}A_m^{(dir)}\cos\left(2\pi mf_pt+\varphi_m^{(dir)}\right)$
- 三个方向$dir\triangleq x,y\mathrm{~and~}z$
- 谐波成分的数量 $N_\kappa$

### Full-field Measurement

MISO vibration controller 使用 force appropriation algorithm(使用单点测量的数据进行反馈调节) 来维持结构振动在其某个resonances(共振态)，来保证3D SLDV的测量。使用3D SLDV来测量各个scan point的XYZ三方向加速度，对每个点测量很小的sample interval 来减少总体测试时间

two estimates of driving force $\boldsymbol{p}(t)$ 来处理空间测量不一致的问题：
- average estimates: vibration controller 提供的估计值，large interval
- Instantaneous estimates：3D SLDV测量得到的值，small interval
- 这两个估计值之间的差异越小，则说明数据集越准确

Scan Point Measurement:
Sampling rate of 3D SLDV $f^{\mathrm{SLDV~}}=1/\Delta t$ ， 每个scan point 采样时长为$T=N\Delta t$，为了抗混叠应远高于响应信号最高谐波Nyquist rate：$f^{\mathrm{SLDV}}>2.56\cdot N_{\kappa}f_{p},$

时间信号：
- External excitation Force: $p(k\Delta t){=A_p\cos(2\pi f_p\Delta t+\varphi_p),}\quad k=0,1\cdots,N-1\mathrm{~,}$
- Response Signals of 3 directions: $u_{(dir)}(k\Delta t){=\sum_{m=1}^{N_\kappa}A_m^{(dir)}\cos\left(2\pi mf_p\Delta t+\varphi_m^{(dir)}\right)},\quad k=0,1\cdots,N-1 ,$

DFT:
- External excitation Force: $P(n)=\frac{A_p}2\Big[e^{\mathrm{j}\varphi_p}D(n-f_p/\Delta f) + e^{-\mathrm{j}\varphi_p}D(n+f_p/\Delta f)\Big],$
- Response Signals of 3 directions: $U(n)=\sum_{m=1}^{N_\kappa}\frac{A_m}2\Big[e^{\mathrm{j}\varphi_m}D(n-mf_p/\Delta f) + e^{-\mathrm{j}\varphi_m}D(n+mf_p/\Delta f)\Big]$
  - Dirichlet kernel $D(\theta){=e^{{-\mathrm{j}\pi\theta[(N-1)/N]}}\frac{\sin(\pi\theta)}{N\sin(\pi\theta/N)}.}$






