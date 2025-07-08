---
title: Sensitivity Analysis
date: 2025-06-17 13:17:17
tags:
  - 
categories: ModelUpdating/Method
---

Sensitivity Analysis

<!-- more -->

# GSA

全局敏感性分析（GSA）是一种通过使用计算机辅助完成整体计算分析的强大技术，用于分析输入不确定性对模型输出结果变化的影响。

```BibTex
@misc{sadeghiReviewGlobalSensitivity2024,
	title = {A {Review} of {Global} {Sensitivity} {Analysis} {Methods} and a comparative case study on {Digit} {Classification}},
	url = {http://arxiv.org/abs/2406.16975},
	doi = {10.48550/arXiv.2406.16975},
	abstract = {Global sensitivity analysis (GSA) aims to detect influential input factors that lead a model to arrive at a certain decision and is a significant approach for mitigating the computational burden of processing high dimensional data. In this paper, we provide a comprehensive review and a comparison on global sensitivity analysis methods. Additionally, we propose a methodology for evaluating the efficacy of these methods by conducting a case study on MNIST digit dataset. Our study goes through the underlying mechanism of widely used GSA methods and highlights their efficacy through a comprehensive methodology.},
	language = {en},
	urldate = {2025-06-17},
	publisher = {arXiv},
	author = {Sadeghi, Zahra and Matwin, Stan},
	month = jun,
	year = {2024},
	note = {arXiv:2406.16975 [cs]},
	keywords = {/reading, Computer Science - Artificial Intelligence, Computer Science - Machine Learning},
	file = {PDF:D\:\\Download\\Zotero_data\\storage\\CHJHEB36\\Sadeghi和Matwin - 2024 - A Review of Global Sensitivity Analysis Methods and a comparative case study on Digit Classification.pdf:application/pdf},
}
```

GSA 分类：
- Variance based methods
  - Sobol *常用方法*
  - FAST
  - RBD and FAST RBD
-  Derivative based methods
  - Morris
  - DGSM
- Distribution based methods
  - DELTA

## Sobol方法

输入$\mathbf{x}=\left({x}_{1},{x}_{2},\cdots ,{x}_{n}\right)$，系统输出$y=f(x)$，在假设输入参数相互独立的前提下，输出响应$y$的总方差为：

$$\mathrm{V}\mathrm{a}\mathrm{r}\left(y\right)=\sum _{i=1}^{n}{\mathrm{V}\mathrm{a}\mathrm{r}}_{i}\left(y\right)+\sum _{1\le i<j\le n}{\mathrm{V}\mathrm{a}\mathrm{r}}_{ij}\left(y\right)+\cdots +{\mathrm{V}\mathrm{a}\mathrm{r}}_{12\cdots n}\left(y\right)$$
- ${\mathrm{V}\mathrm{a}\mathrm{r}}_{i}\left(y\right)$ 表示仅由参数$x_{i}$引起的输出方差
- ${\mathrm{V}\mathrm{a}\mathrm{r}}_{ij}\left(y\right)$ 表示$x_{i}$与$x_j$交互作用引起的输出方差
- ${\mathrm{V}\mathrm{a}\mathrm{r}}_{12\cdots n}\left(y\right)$ 表示所有参数联合作用产生的高阶交互影响

一阶Sobol指标 ${S}_{i}=\frac{{E}_{-i}\left[{\mathrm{V}\mathrm{a}\mathrm{r}}_{i}\left(y\right)\right]}{\mathrm{V}\mathrm{a}\mathrm{r}\left(y\right)}$， 用于衡量单个参数$x_{i}$对输出方差的贡献

总阶Sobol指标  ${S}_{{T}_{i}}=1-\frac{{E}_{i}\left[Va{r}_{-i}\left(y\right)\right]}{Var\left(y\right)}$ 用于衡量包含参数$x_{i}$的所有阶次（包括其自身及与其他参数的交互作用）对输出方差的贡献

### WIKI source

> [Variance-based sensitivity analysis - Wikipedia](https://en.wikipedia.org/wiki/Variance-based_sensitivity_analysis)
> [Sobol敏感性分析详解-CSDN博客](https://blog.csdn.net/xiaosebi1111/article/details/46517409)

从黑箱的角度来看，任何模型可以被表示为一个函数：$Y=f(X)$，输入X为一个d维向量，Y为所感兴趣的标量(例如某个模态频率值)。其中假设输入在单位“超立方体”内独立均匀分布，即$X_i\in[0,1]$

函数$f(X)$可以被分解为：$Y=f_0+\sum_{i=1}^df_i(X_i)+\sum_{i<j}^df_{ij}(X_i,X_j)+\cdots+f_{1,2,\ldots,d}(X_1,X_2,\ldots,X_d)$
- $f_{0}$为常数
- $f_{i}$为$X_{i}$的函数
- $f_{ij}$为$X_{i},X_{j}$的函数

并且所有的函数满足正交的特性，即：$\int_0^1f_{i_1i_2\ldots i_s}(X_{i_1},X_{i_2},\ldots,X_{i_s})dX_k=0,\mathrm{~for~}k=i_1,\ldots,i_s$，则可以从期望的角度来描述

Procedure：N为样本数量，d为自变量个数
- 生成`Nx2d` 样本矩阵，前d列为$A$，后d列为$B$
- 用矩阵$B$中的第i列替换矩阵$A$的第i列，生成$AB^{i}$ 矩阵(`Nxd`)
- 得到d+2个矩阵（$A,B,AB^{i}(i=1,2,\dots d)$），每个矩阵根据d个自变量获得输出y，总共有$N*(d+2)$个样本点($x_{1},x_{2}\dots x_{n} \rightarrow y$)
- $\mathrm{Var}(Y)=var(cat(A,B))$，对AB两矩阵得到的$N*2$个样本点求方差
- $\mathrm{Var}_{X_i}\left(E_{\mathbf{X}_{\sim i}}\left(Y|X_i\right)\right)\approx\frac{1}{N}\sum_{j=1}^Nf(\mathbf{B})_j\left(f{\left(\mathbf{A}_B^i\right)}_j-f{\left(\mathbf{A}\right)}_j\right)$
- 一阶Sobol指标 $S_{i}=\frac{Var_{X_{i}}(E_{X\sim i}(Y|X_{i}))}{Var(Y)}$
- $E_{\mathbf{X}_{\sim i}}\left(\mathrm{Var}_{X_i}\left(Y\mid\mathbf{X}_{\sim i}\right)\right)\approx\frac{1}{2N}\sum_{j=1}^N\left(f(\mathbf{A})_j-f{\left(\mathbf{A}_B^i\right)_j}\right)^2$
- 总阶Sobol指标 $S_{Ti}=\frac{E_{X\sim i}(Var_{X_{i}}(Y|X_{\sim i}))}{Var(Y)}$
- 不常用的二阶Sobol指标：
  - $V_{ij}=\frac{1}{N}\sum_{q=1}^{N}f(\mathbf{B})_{q}\left(f(\mathbf{A}_{B}^{ij})_{q}-f(\mathbf{A}_{B}^{i})_{q}-f(\mathbf{A}_{B}^{j})_{q}+f(\mathbf{A})_{q}\right)$
  - $S_{ij}=\frac{V_{ij}}{\mathrm{Var}(Y)}$

