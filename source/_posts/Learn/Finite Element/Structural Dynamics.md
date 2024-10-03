---
title: Structural Dynamics
date: 2024-09-24 21:32:23
tags: 
categories: Blog&Book&Paper/Read/Blog
---

Structural Dynamics Basic Knowledge.

> [mdof](https://chrystalchern.github.io/mdof/) provides fast and friendly system identification for structures.

<!-- more -->

Book ToDo：
- [ ] Dynamics of Structures by Anil K. Chopra
- [ ] Mechanical Vibrations Theory and Application to Structural Dynamics (Michel Geradin, Daniel J. Rixen) 🤓
- [ ] 机械振动 张义民

Nice Video
- [x] [Understanding Vibration and Resonance - YouTube](https://www.youtube.com/watch?v=vLaFAKnaRJU) 三自由度弹簧系统中每个质量块都有自己的位移$x_{1}(t),x_{2}(t),x_{3}(t)$，因此该系统会有三个固有频率和模态振型


> [「结构动力学入门」引入 - 知乎](https://zhuanlan.zhihu.com/p/377733382)

实际工程中，力是变化的$P(t)$，且要考虑惯性，因此在动力平衡方程中要添加惯性力:
惯性力：$\boldsymbol{F}\left(t\right)=-\boldsymbol{Ma}\left(t\right)=-\boldsymbol{M \ddot{u}}\left(t\right)$
动力平衡方程：$\boldsymbol{M\ddot{u}}\left(t\right)+\boldsymbol{Ku}\left(t\right)=\boldsymbol{P}\left(t\right)$ 

> [一维质量弹簧系统阻尼简谐运动 - 知乎](https://zhuanlan.zhihu.com/p/594067837)

无外力: $P(t)=0$
- $u(t)=A\cos(\omega_0t+\varphi)$
  - $\omega_{0}=\sqrt{ \frac{k}{m} }$ 系统的固有频率只与刚度质量有关
  - $\varphi=\arctan(-\frac{\dot{x}_0}{\omega_0x_0})-\omega_0t_0$ 
  - $A=\sqrt{x_0^2+\frac{\dot{x}_0^2}{\omega_0^2}}$ 

有外力：$P(t)=F_{0}\sin(\omega_{f}t)$ $\omega_{f}$是外加力的频率
- $u(t)=Ae^{-\xi\omega t}\sin(\omega_0 t+\phi) + \frac{F_0/k \sin(w_ft-\theta)}{\sqrt{ (1-r^{2})^{2}+(2r \xi)^{2} }}$

