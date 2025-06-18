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

> [Variance-based sensitivity analysis - Wikipedia](https://en.wikipedia.org/wiki/Variance-based_sensitivity_analysis)

输入$\mathbf{x}=\left({x}_{1},{x}_{2},\cdots ,{x}_{n}\right)$，系统输出$y=f(x)$，在假设输入参数相互独立的前提下，输出响应$y$的总方差为：

$$\mathrm{V}\mathrm{a}\mathrm{r}\left(y\right)=\sum _{i=1}^{n}{\mathrm{V}\mathrm{a}\mathrm{r}}_{i}\left(y\right)+\sum _{1\le i<j\le n}{\mathrm{V}\mathrm{a}\mathrm{r}}_{ij}\left(y\right)+\cdots +{\mathrm{V}\mathrm{a}\mathrm{r}}_{12\cdots n}\left(y\right)$$
- ${\mathrm{V}\mathrm{a}\mathrm{r}}_{i}\left(y\right)$ 表示仅由参数$x_{i}$引起的输出方差
- ${\mathrm{V}\mathrm{a}\mathrm{r}}_{ij}\left(y\right)$ 表示$x_{i}$与$x_j$交互作用引起的输出方差
- ${\mathrm{V}\mathrm{a}\mathrm{r}}_{12\cdots n}\left(y\right)$ 表示所有参数联合作用产生的高阶交互影响

一阶Sobol指标 ${S}_{i}=\frac{{E}_{-i}\left[{\mathrm{V}\mathrm{a}\mathrm{r}}_{i}\left(y\right)\right]}{\mathrm{V}\mathrm{a}\mathrm{r}\left(y\right)}$， 用于衡量单个参数$x_{i}$对输出方差的贡献

总阶Sobol指标  ${S}_{{T}_{i}}=1-\frac{{E}_{i}\left[Va{r}_{-i}\left(y\right)\right]}{Var\left(y\right)}$ 用于衡量包含参数$x_{i}$的所有阶次（包括其自身及与其他参数的交互作用）对输出方差的贡献

