---
zotero-key: XBBGA9BJ
zt-attachments:
  - "4648"
title: Global sensitivity analysis using polynomial chaos expansions
created: 2025-07-31 10:51:26
modified: 2025-08-18 04:27:35
tags:
  - Global
  - sensitivity
  - analysis
  - Analysis
  - of
  - variance
  - Generalized
  - chaos
  - Polynomial
  - chaos
  - Regression
  - Sobol’
  - indices
  - Stochastic
  - finite
  - elements
collections: Sensitivity Analysis  NN
year: "2008"
publication: Reliability Engineering & System Safety
citekey: sudretGlobalSensitivityAnalysis2008
---

| Title        | "Global sensitivity analysis using polynomial chaos expansions"                                                                                    |
| ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| Author       | [Bruno Sudret]                                                                                                                                     |
| Organization | Electricite ́ de France, R&D Division, Site des Renardieres, F 77818 Moret-sur-Loing Cedex, **France**                                             |
| Paper        | [Zotero pdf](zotero://select/library/items/XBBGA9BJ) [attachment](file:///D:/Download/Zotero_data/storage/TRWBH3LF/S0951832007001329.html)<br><br> |
| Project      |                                                                                                                                                    |

<!-- more -->

## Background

## Innovation

## Outlook

## Cases

## Equation

另外，粗略看下这个文章，为什么“The advantage of PCE for UQ and SA purpose is that the calculation of Sobol sensitivity indices can be directly performed as the postprocessing step”，是否可以用NN直接代理相关过程，求敏感性指标

Polynomial chaos expansion (PCE)
“any second-order random variable” ([Sudret, 2008, p. 967](zotero://select/library/items/XBBGA9BJ)) ([pdf](zotero://open-pdf/library/items/XI4SEA5A?page=4&annotation=KNC3HYU5)) 可以表示为： $Z=\sum_{j=0}^\infty Z_j\Psi_j(\{\xi_n\}_{n=1}^\infty).$
- $\{\xi_n\}_{n=1}^\infty$ ：independent standard normal random variables
- $\Psi_j$ ：the multivariate Hermite polynomials (is orthogonal with respect to the Gaussian measure)
- 目标是找到合适的系数$Z_{j}$，使得PCE多项式可以很好地描述FE model simulation

$Y=f(X)\approx f_{\mathrm{PC}}(X)=\sum_{j=0}^{P-1}f_{j}\Psi_{j}(X),\quad X{\sim}\mathscr{U}([-1,1]^{n}).$

Due to the orthogonality of the basis, it is easy to show that the mean and variance of the response respectively read:

(GoogleAI: $\operatorname{E}[\Psi_j(X)\Psi_k(X)]=0\quad\text{当 }j\neq k$ 。并且在构造基函数时，第一个基函数是常数1，其他$j\geq 1$的基函数期望为0)

$\begin{aligned}&\bar{Y}=\mathrm{E}[f(X)]=f_{0},\\&D_{\mathrm{PC}}=\mathrm{Var}\left[\sum_{j=0}^{P-1}f_{j}\Psi_{j}(X)\right]=\sum_{j=1}^{P-1}f_{j}^{2}\mathrm{E}[\Psi_{j}^{2}(X)].\end{aligned}$


$\begin{aligned}f_{\mathrm{PC}}(x)&=f_0+\sum_{i=1}^n\sum_{\alpha\in\mathscr{I}_i}f_\alpha\Psi_\alpha(x_i)+\sum_{1\leqslant i_1<i_2\leqslant n}\sum_{\alpha\in\mathscr{I}_{i_1,i_2}}f_\alpha\Psi_\alpha(x_{i_1},x_{i_2})\\&+\cdots+\sum_{1\leqslant i_{1}<\cdots<i_{s}\leqslant n}\sum_{\alpha\in\mathscr{I}_{i_{1},...,i_{s}}}f_{\alpha}\Psi_{\alpha}(x_{i_{1}},\ldots,x_{i_{s}})+\cdots+\sum_{\alpha\in\mathscr{I}_{1,2,\ldots,n}}f_{\alpha}\Psi_{\alpha}(x_{1},\ldots,x_{n}).\end{aligned}$

