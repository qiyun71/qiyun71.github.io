---
title: Mass Spring System
date: 2025-08-13 12:50:10
tags: 
categories: ModelUpdating/Cases
Year: 
Journal:
---

## Mass Spring System

The absolute value of the first component of the first eigenvector reflects some vibration information. The introduction of structural vibration modes as output responses will increase the difficulty of IMU.

**Numerical case studies: a mass-spring system**


![massSpring.png|555](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/massSpring.png)


| 结构参数         | Well-Separated | Close      |
| ------------ | -------------- | ---------- |
| $m_{1}$ (kg)  | 1              | 1          |
| $m_{2}$ (kg)  | 1              | 4          |
| $m_{3}$ (kg)  | 1              | 1          |
| $k_{1}$ (N/m) | [0.8, 1.2]     | 0          |
| $k_{2}$ (N/m) | [0.8, 1.2]     | [7.5, 8.5] |
| $k_{3}$ (N/m) | 1              | 0          |
| $k_{4}$ (N/m) | 1              | [1.8, 2.2] |
| $k_{5}$ (N/m) | [0.8, 1.2]     | [1.8, 2.2] |
| $k_{6}$ (N/m) | 3              | 1          |

| 动力学响应                 |
| --------------------- |
| $\omega_1^2(rad/s)^2$ |
| $\omega_2^2(rad/s)^2$ |
| $\omega_3^2(rad/s)^2$ |
| $\|\varphi(1,1)\|$    |

### Equation

M-K matrix (FE): $M\ddot{X} + KX = 0$

$M = \left( \begin{matrix}  m_{1} & 0 & 0\\ 0 & m_{2} & 0 \\ 0 & 0 & m_{3} \end{matrix} \right)$

$K = \left( \begin{matrix}  k_{1}+k_{4}+k_{6} & -k_{4} & -k_{6}\\ -k_{4} & k_{2}+k_{4}+k_{5} & -k_{5} \\ -k_{6} & -k_{5} & k_{3}+k_{5}+k_{6} \end{matrix} \right)$


$(-M \omega^{2}+K)A = 0$
$|-M \omega^{2}+K| = 0$

$\omega^{2} = M^{-1}K = Q\Sigma Q^{\top}$

$\omega_{1}^{2} = \Sigma(1,1)$
$\omega_{2}^{2} = \Sigma(2,2)$
$\omega_{3}^{2} = \Sigma(2,2)$
$|\varphi(1,1)| = Q(1,1)$

```python
M = np.array([[m1, 0, 0], 
            [0, m2, 0], 
            [0, 0, m3]])
K = np.array([[k1 + k4 + k6, -k4,           -k6], 
            [-k4,            k2 + k4 + k5,  -k5], 
            [-k6,            -k5,            k3 + k5 + k6]])

lambda_ , vector_ = np.linalg.eig(np.linalg.inv(M) @ K)
lambda_ , vector_ = lambda_.real, vector_.real
vector_ = vector_[:,np.argsort(lambda_)]
lambda_ = np.sort(lambda_)

# print(f'Matrix K: {K} \nMatrix M: {M} \nEigenvalues: {lambda_} \nEigenvectors: {vector_}')
# print('K\n',K)
# print(f'K(=M @ v @ lambda @ v.T) \n{M @ vector_ @ np.array([[lambda_[0], 0, 0], [0, lambda_[1], 0], [0, 0, lambda_[2]]]) @ vector_.T}')
# exit()

lambda_1 = lambda_[0]
lambda_2 = lambda_[1]
lambda_3 = lambda_[2]
phi1_1 = np.abs(vector_[0,0])

# the length of the first eigenvector
# phi1_1 = np.sqrt(np.sum(vector_[:,0]**2))
```

Torch 版本：
```python
k1, k2, k3, k4, k5, k6 = k_params
m1, m2, m3 = m_params
device = k_params.device
dtype = k_params.dtype

M = torch.diag(torch.stack([m1, m2, m3]))
K = torch.zeros((3, 3), device=device, dtype=dtype)
K[0, 0] = k1 + k4 + k6
K[0, 1] = -k4
K[0, 2] = -k6
K[1, 0] = -k4
K[1, 1] = k2 + k4 + k5
K[1, 2] = -k5
K[2, 0] = -k6
K[2, 1] = -k5
K[2, 2] = k3 + k5 + k6

# torch.linalg.inv 和 torch.linalg.eig 都是可微的
M_inv = torch.linalg.inv(M)
A = M_inv @ K

# torch.linalg.eig 返回可能为复数的特征值和特征向量
eigenvalues, eigenvectors = torch.linalg.eig(A)

# 对于这类物理问题，特征值应为实数。我们取其实部。
# .real 操作在 PyTorch 中是可微的
eigenvalues = eigenvalues.real
eigenvectors = eigenvectors.real

# torch.sort 是可微的。
sorted_lambda, sort_indices = torch.sort(eigenvalues)

sorted_vectors = eigenvectors[:, sort_indices]

lambda_1 = sorted_lambda[0]
lambda_2 = sorted_lambda[1]
lambda_3 = sorted_lambda[2]

# torch.abs() 是可微的
phi1_1 = torch.abs(sorted_vectors[0, 0])
```

### Applications

#### Interval Uncertainty Propagation

将结构参数的不确定性区间(认知不确定性)传播到动力学响应的不确定性区间，常用的有 MC 法和区间摄动法：

***Interval perturbation***

> 参考：[Interval parameter sensitivity analysis based on interval perturbation propagation and interval similarity operator](https://hal.science/hal-04273667v1/document)

$\overline{\widehat{\boldsymbol{f}}}=F(\boldsymbol{\theta}^c)+\sum_{j=1}^N\frac{\boldsymbol{F}\left(\theta_j^c+\delta\theta_j\right)-\boldsymbol{F}\left(\theta_j^c\right)}{\delta\theta_j}\Delta\theta_j$
$\underline{\widehat{\boldsymbol{f}}}=F(\boldsymbol\theta^c)-\sum_{j=1}^N\frac{\boldsymbol{F}\left(\theta_j^c+\delta\theta_j\right)-\boldsymbol{F}\left(\theta_j^c\right)}{\delta\theta_j}\Delta\theta_j$

***Monte Carlo***

随机采样结构参数 $\{x_{i}^{g}\}_{g=1}^{N_{s}}$，然后经过前向仿真计算得到对应的动力学响应 $\{y_{i}^{g}\}_{g=1}^{N_{s}}$

$\begin{gathered}\boldsymbol{Y}^{\mathbf{C}}=\frac{1}{2}\left(\max_{1\leq g\leq N_{s}}y_{j}^{g}+\min_{1\leq g\leq N_{s}}y_{j}^{g}\right)\\\boldsymbol{\Delta Y}^\mathbf{I}=\frac{1}{2}\left(\max_{1\leq g\leq N_s}y_j^g-\min_{1\leq g\leq N_s}y_j^g\right)\end{gathered}$

区别：(常用)MC 法虽然效率低，但是在充分样本量的情况下，可以保证复杂模型的区间不确定性传播精度。摄动法的效率虽然高，但只能传播简单模型的区间不确定性，对于复杂模型传播精度差。


（使用 MC 需要大量采样时）为了提升有限元计算的效率，多采用 FE surrogate model 来代替 FE 仿真计算，常用的有 Response surface model (RSM)、polynomial chaos expansion（PCE）、Gaussian process model/Kriging、Radial Basis Function（RBF） network、Multi-Layer Perceptron (MLP)等

Response Surface Model (RSM)
- Well-separated modes
$$
\begin{aligned}\omega_1^2&=0.2840+0.3416k_1+0.4122k_2+0.0078k_5+0.0745k_1k_2+0.0011k_1k_5\\&-0.0014k_2k_5-0.0423k_1^2-0.0753k_2^2-0.0020k_5^2, \\
\omega_2^2&=1.6117+0.1249k_1+0.5882k_2+1.7402k_5-0.0735k_1k_2+0.1243k_1k_5\\&-0.0015k_2k_5-0.0021k_1^2+0.0748k_2^2-0.1871k_5^2, \\
\omega_3^2&=7.1036+0.5331k_1+0.0001k_2+0.2531k_5-0.0014k_1k_2-0.1247k_1k_5\\&+0.0025k_2k_5+0.0444k_1^2+0.0007k_2^2+0.1885k_5^2, \\
|\varphi(1,1)|&=0.5642-0.0894k_1+0.1060k_2+0.0171k_5+0.0082k_1k_2+0.0059k_1k_5\\&-0.0194k_2k_5+0.0009k_1^2-0.0150k_2^2-0.0012k_5^2.\end{aligned}
$$

- Close modes
$$
\begin{aligned}\omega_1^2&=-0.0002+0.0830k_2+0.0839k_4+0.0842k_5+0.0186k_2k_4+0.0185k_2k_5\\&-0.0094k_4k_5-0.0046k_2^2-0.0325k_4^2-0.0325k_5^2,\\
\omega_2^2&=1.6103+0.0104k_2+1.0455k_4-0.0937k_5-0.0097k_2k_4+0.0055k_2k_5\\&+0.0094k_5+0.0042k_2^2+0.0396k_4^2+0.0005k_5^2, \\
\omega_3^2&=1.1103+0.0273k_2+0.0162k_4+1.1572k_5-0.0003k_2k_4-0.0165k_2k_5\\&+0.0104k_4k_5+0.0065k_2^2-0.0034k_4^2+0.0372k_5^2,\\
|\varphi(1,1)|&=0.6658+0.0125k_2-0.0988k_4+0.0496k_5-0.0062k_2k_4+0.0072k_2k_5\\&+0.0020k_4k_5-0.0005k_2^2+0.0190k_4^2-0.0170k_5^2.\end{aligned}
$$


| 区间传播方法                      | Interval Perturbation(First-order) | Monte Carlo(MC) |
| --------------------------- | ---------------------------------- | --------------- |
| M-K matrix (FE)             | 😊                                 |                 |
| Response Surface Model(RSM) |                                    |                 |

结果对比

***well-separated modes***
- 使用 M&K(FE)或者 RSM，与 Monte Carlo 方法得到的响应区间相比，区间摄动法得到的$|\varphi(1,1)|$ 区间误差很大，$\Delta|\varphi(1,1)|$ 计算的偏小

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240603104442.png)

***close modes***
- (绿色)通过质量刚度矩阵(M&K)和蒙特卡洛法得到的响应区间在 ws 模式时可以很准确，但是在 cl 模式下，使用 M&K 与 RSM 相比有一定误差，主要是对$\omega _{2}^{2}$和$\omega _{3}^{2}$预测的不好
- (紫色)RSM 和区间摄动法得到$|\varphi(1,1)|$的响应区间相较于(黑色)RSM 和 MC 方法的误差还是很大

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240603104718.png)

#### NN-based interval model updating (ISRERM 会议)

>  [qiyun71/MU_MassSpring](https://github.com/qiyun71/MU_MassSpring) 基于 MLP 的 Interval Model Updating(EI 会议)

***Well-separated modes***

数据集生成：
- 在区间$[0,2]$内均匀生成 10000 组$k_1,k_2,k_5$
- 根据 [Interval Identification of Structural Parameters Using Interval Deviation Degree and Monte Carlo Simulation](Interval%20Identification%20of%20Structural%20Parameters%20Using%20Interval%20Deviation%20Degree%20and%20Monte%20Carlo%20Simulation.md)，关于$\begin{aligned}\omega_1^2,\omega_2^2,\omega_3^2,\text{ and }|\varphi(1,1)|\end{aligned}$ 四个参数的二阶 RSM(根据 CCD(central composite design)生成 15 个 samples)，得到 10000 组$\begin{aligned}\omega_1^2,\omega_2^2,\omega_3^2,\text{ and }|\varphi(1,1)|\end{aligned}$

目标：网络可以根据一组输入的$\begin{aligned}\omega_1^2,\omega_2^2,\omega_3^2,\text{ and }|\varphi(1,1)|\end{aligned}$得到对应的一组$k_1,k_2,k_5$

| 实验             | 初始区间(N/m)        | KP 的*错误率*      | IRSM           | PF&RBF-NN | IOR&MC         | IDD&MC         | **本文方法**       |
| -------------- | ---------------- | ------------- | -------------- | --------- | -------------- | -------------- | -------------- |
| $[0.80, 1.20]$ | $k_1=[0.5, 1.5]$ | $[0.4,  0.0]$ | $[0.81, 1.20]$ | NAN       | $[0.79, 1.21]$ | $[0.80, 1.20]$ | $[0.80, 1.20]$ |
| $[0.80, 1.20]$ | $k_2=[0.5, 1.5]$ | $[0.8, 1.7]$  | $[0.80, 1.21]$ | NAN       | $[0.80, 1.20]$ | $[0.80, 1.19]$ | $[0.80, 1.20]$ |
| $[0.80, 1.20]$ | $k_5=[0.5, 1.5]$ | $[0.8, 1.7]$  | $[0.80, 1.20]$ | NAN       | $[0.80, 1.20]$ | $[0.80, 1.20]$ | $[0.80, 1.20]$ |
| ER             |                  |               |                | NAN       |                |                |                |
|                | $[37.5, 25]$     | $[0.4,  0.0]$ | $[1.3, 0]$     | NAN       | $[1.3, 0.8]$   | $[0, 0]$       | $[0, 0]$       |
|                | $[37.5, 25]$     | $[0.8, 1.7]$  | $[0, 0.8]$     | NAN       | $[0, 0]$       | $[0, 0.8]$     | $[0, 0]$       |
|                | $[37.5, 25]$     | $[0.8, 1.7]$  | $[0, 0]$       | NAN       | $[0, 0]$       | $[0,0]$        | $[0, 0]$       |
| **mean**       | $[37.5, 25]$     | $[0.7, 1.1]$  | $[0.4, 0.3]$   | NAN       | $[0.4, 0.3]$   | $[0, 0.3]$     | $[0, 0]$       |

***Close modes***

数据集生成：10000 组$k_2,k_4,k_5$-->10000 组$\begin{aligned}\omega_1^2,\omega_2^2,\omega_3^2,\text{ and }|\varphi(1,1)|\end{aligned}$

目标：网络可以根据一组输入的$\begin{aligned}\omega_1^2,\omega_2^2,\omega_3^2,\text{ and }|\varphi(1,1)|\end{aligned}$得到对应的一组$k_2,k_4,k_5$

| 实验           | 初始区间(N/m)        | KP 的*错误率*     | IRSM           | PF&RBF-NN | IOR&MC         | IDD&MC         | **本文方法**       |
| ------------ | ---------------- | ------------ | -------------- | --------- | -------------- | -------------- | -------------- |
| $[7.5, 8.5]$ | $k_2=[6.5, 9.5]$ | $[0.6, 0.7]$ | $[7.55, 8.54]$ | NAN       | $[7.48, 8.50]$ | $[7.46, 8.52]$ | $[7.50, 8.50]$ |
| $[1.8, 2.2]$ | $k_4=[1.6, 2.4]$ | $[0.8, 1.0]$ | $[1.80, 2.19]$ | NAN       | $[1.80, 2.21]$ | $[1.80, 2.20]$ | $[1.80, 2.20]$ |
| $[1.8, 2.2]$ | $k_5=[1.5, 2.4]$ | $[0.4, 0.5]$ | $[1.80, 2.20]$ | NAN       | $[1.80, 2.21]$ | $[1.81, 2.20]$ | $[1.80, 2.20]$ |
| ER           |                  |              |                | NAN       |                |                |                |
|              | $[13.3, 11.8]$   | $[0.6, 0.7]$ | $[0.7, 0.5]$   | NAN       | $[0.3, 0]$     | $[0.5, 0.2]$   | $[0, 0]$       |
|              | $[11.1, 9.1]$    | $[0.8, 1.0]$ | $[0, 0.5]$     | NAN       | $[0, 0.5]$     | $[0, 0]$       | $[0, 0]$       |
|              | $[11.1, 9.1]$    | $[0.4, 0.5]$ | $[0, 0]$       | NAN       | $[0, 0.5]$     | $[0.6, 0]$     | $[0, 0]$       |
| **mean**     | $[11.8, 10.0]$   | $[0.6, 0.7]$ | $[0.2, 0.3]$   | NAN       | $[0.1, 0.3]$   | $[0.4, 0.1]$   | $[0, 0]$       |
|              |                  |              |                |           |                |                |                |

#### Response-consistent MLP for interval model calibration


#### NN-based stochastic model calibration

***Well-separated modes***

> [Interval Identification of Structural Parameters Using Interval Deviation Degree and Monte Carlo Simulation](Interval%20Identification%20of%20Structural%20Parameters%20Using%20Interval%20Deviation%20Degree%20and%20Monte%20Carlo%20Simulation.md)

| 结构参数    | 待修正参数            | 区间范围(均匀生成 1000) |
| ------- | ---------------- | -------------- |
| $k_{1}$ | $\mu_{k_{1}}$    | $[0,2]$        |
|         | $\sigma_{k_{1}}$ | $[0,2]$        |
| $k_{2}$ | $\mu_{k_{2}}$    | $[0,2]$        |
|         | $\sigma_{k_{2}}$ | $[0.1,0.2]$    |
| $k_{5}$ | $\mu_{k_{5}}$    | $[0.1,0.2]$    |
|         | $\sigma_{k_{5}}$ | $[0.1,0.2]$    |

- 在区间$[0,2]$ 和 $[0.1,0.2]$内均匀生成 1000 组的均值与方差：$\mu_{k_{1}}$, $\sigma_{k_{1}}$, $\mu_{k_{2}}$, $\sigma_{k_{2}}$, $\mu_{k_{5}}$, $\sigma_{k_{5}}$
- 每一组均值与方差生成 60 组$k_{1}$, $k_{2}$, $k_{5}$
- 根据 RSM 计算得到 60 组$\begin{aligned}\omega_1^2,\omega_2^2,\omega_3^2,\text{ and }|\varphi(1,1)|\end{aligned}$

共需要计算 60x1000 次有限元模型(代理模型)

目标：网络可以根据一组 60 个输入$\begin{aligned}\omega_1^2,\omega_2^2,\omega_3^2,\text{ and }|\varphi(1,1)|\end{aligned}$得到对应的$k_1,k_2,k_5$三个参数的分布(用 PDF 曲线表示)
- 输入：60x4
- 输出：

***Close modes***
