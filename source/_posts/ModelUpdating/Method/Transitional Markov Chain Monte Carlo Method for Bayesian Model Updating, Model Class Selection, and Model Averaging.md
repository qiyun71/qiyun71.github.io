---
title: Transitional Markov Chain Monte Carlo Method for Bayesian Model Updating, Model Class Selection, and Model Averaging
date: 2024-03-15 11:23:15
tags:
  - 
categories: ModelUpdating/Method
---

| Title     | Transitional Markov Chain Monte Carlo Method for Bayesian Model Updating, Model Class Selection, and Model Averaging                                                                                                                                           |
| --------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Author    | J. Ching; Yi-Chu Chen, *NationalTaiwan Univ. of Science and Technolog*                                                                                                                                                                                         |
| Conf/Jour | Journal of Engineering Mechanics                                                                                                                                                                                                                               |
| Year      | 2007                                                                                                                                                                                                                                                           |
| Project   | [Transitional Markov Chain Monte Carlo Method for Bayesian Model Updating, Model Class Selection, and Model Averaging \| Journal of Engineering Mechanics \| Vol 133, No 7](https://ascelibrary.org/doi/10.1061/%28ASCE%290733-9399%282007%29133%3A7%28816%29) |
| Paper     | https://ascelibrary.org/doi/epdf/10.1061/%28ASCE%290733-9399%282007%29133%3A7%28816%29                                                                                                                                                                         |

TMCMC

<!-- more -->

# 贝叶斯MU

两个关键：
- 似然函数的定义
- 求解后验分布的MCMC采样算法

## 近似贝叶斯

贝叶斯：$\mathrm{P}(\xi|\Psi_{obs})=\frac{\mathrm{P}_{L}(\Psi_{obs}|\xi)\mathrm{P}(\xi)}{\mathrm{P}(\Psi_{obs})}$

**近似贝叶斯**，就是使用UQ指标(如IOR区间重合度)来代替其中的似然函数，保证当sim和obs之间的差异变小时，似然函数变大，使得后验概率更大
近似贝叶斯**似然函数**：$\mathbb{P}_L(\boldsymbol{\Psi}_{obs}|\boldsymbol{\xi})\propto exp\left\{-\frac{\left(IOR\left(\boldsymbol{\Psi}_{\boldsymbol{N}_{sim}}^{\boldsymbol{I}-\boldsymbol{sub}}\left|\boldsymbol{\Psi}_{\boldsymbol{N}_{obs}}^{\boldsymbol{I}-\boldsymbol{sub}}\right)\right)^2\right)}{\varepsilon^2}\right\}$

## TMCMC(Transitional Markov Chain Monte Carlo Method)

(马尔可夫+蒙特卡洛)MCMC本质是采样过程，被用来求解复杂的后验分布，即当先验与后验分布差异较大时，也可很好地求解。

基于MCMC的改进有Metropolis-Hasting、Gibbs sampling等等，TMCMC可以简单理解为分布MCMC，解决了从复杂PDFs中采样的问题，变为从一系列中间PDFs中采样，可以从复杂的PDFs(如multimodel PDFs，very peaked PDFs，PDFs with flat manifold)中采样

> [走进贝叶斯统计（五）—— Metropolis-Hasting 算法 - 知乎](https://zhuanlan.zhihu.com/p/411689417)
> [走进贝叶斯统计（六）—— 吉布斯抽样 （Gibbs Sampling） - 知乎](https://zhuanlan.zhihu.com/p/416670115)


程序(基于M-H的TMCMC)
- 根据参数的先验分布$f_{0}(\theta){=}f(\theta|M).$，抽取N组参数$\theta_{0,k}$(这里的参数是刚度or其他的均值和方差，也就是采样了N组多参数的均值和方差)
- 针对N组参数，每一组均值和方差生成$N_{sim}$个样本，然后与实验的$N_{exp}$使用UQ进行对比，最终得到N组似然函数的指数部分：$-\frac{UQ^{2}}{\varepsilon^{2}}$ (每一组与实验的UQ值，大小为Nx1)
- 初始$p_{j}= 0$，根据$p_{j}$和似然指数部分$-\frac{UQ^{2}}{\varepsilon^{2}}$计算$p_{j+1}$
  - **计算方法**：$y = exp^{-\frac{|x| \cdot UQ^{2}}{\varepsilon^{2}}}$,y的大小也是Nx1，找到一个x，使得$\frac{std(y)}{mean(y)} = threshold$，则$p_{j+1} = min(1,p_{j} + x)$
  - 解释：保证数据集$P\left(D|\theta_{i}^{(j)}\right)^{\Delta\beta_{j}}$的变异系数尽可能接近100%——$COV=\frac{\sigma\left\{P(D|\theta_i)^{\Delta\beta_j}\right\}}{\mu\left\{P(D|\theta_i)^{\Delta\beta_j}\right\}}=100\%$，即求$f\big(\Delta\beta_j\big)=\sigma\big\{\exp\big(P(D|\theta_i)\cdot\Delta\beta_j\big)\big\}-\mu\big\{\exp\big(P(D|\theta_i)\cdot\Delta\beta_j\big)\big\}$的根。变异系数$\beta$在代码中为$p_{j}$
- 计算权重$w(\theta_{j,k})=\frac{f(\theta_{j,k}|M)f(D|M,\theta_{j,k})^{p_{j+1}}}{f(\theta_{j,k}|M)f(D|M,\theta_{j,k})^{p_{j}}}=f(D|M,\theta_{j,k})^{p_{j+1}-p_{j}}$
  - 在近似贝叶斯中：$w = exp^{ -\frac{UQ^{2}}{\varepsilon^{2}} \times (p_{j+1} - p_j)}$
- 计算$S_j=\sum_{k=1}^{N_j}w(\theta_{j,k})\left/N_j\right.$（权重均值）
- ResamplingN个样本$\theta^*\sim\begin{array}{c}q(\theta^*|\theta^{(j)})=N(\theta^{(j)},\Sigma)\end{array}$，以前一步的样本为均值，协方差通过权重进行计算，得到$f_{j+1}(\theta)$的近似分布：$\theta_{j+1,k}=\theta_{j,l}\quad\text{w.p.}\quad w(\theta_{j,l})\Bigg/\sum\limits_{l=1}^{N_j}w(\theta_{j,l})\quad k=1,\ldots,N_{j+1}$ 
- MH采样：
  - 协方差：$\begin{gathered}\sum_{j} \left.=\mathbf{\beta}^{2}\sum_{k=1}^{N_{j}}w(\theta_{j,k})\left\{\theta_{j,k}-\left[\sum_{j=1}^{N_{j}}w(\theta_{j,l})\theta_{j,l}\right/\sum_{j=1}^{N_{j}}w(\theta_{j,l})\right]\right\} \\\times\left\{\left.\theta_{j,k}-\left[\sum_{j=1}^{N_{j}}w(\theta_{j,l})\theta_{j,l}\right/\sum_{j=1}^{N_{j}}w(\theta_{j,l})\right]\right\}^{T}\end{gathered}$ ，$\beta$为prescribed scaling factor,这里的$\beta$并不是变异系数. 根据$\Sigma=\gamma^2\sum_{i=1}^{N_{sam}}\widehat{w}(\theta_i)\bullet\left[\{\theta_i-\overline{\theta}\}\times\{\theta_i-\overline{\theta}\}^T\right]$
  - 根据采样的N组多参数$\theta_{0,k}$、logpdf、proppdf、proprnd、thin、burnin，进行MH采样，得到下一步的N组参数$\theta_{1,k}$
    - $logpdf(t) = log(fT(t)) + p_{j+1} \cdot (-\frac{UQ^{2}}{\varepsilon^{2}}(t))$  通过将PDF取log(以e为底)，得到后验分布的logPDF=log先验+log似然，其中fT为priorPDF，根据输入的矩阵NxM，得到一列先验的Nx1(每个样本点的概率)
    - `proppdf(x,y) = mvnpdf(x,y,covmat).*fT(x)` $covmat = \sum_{j}$
      - `fT(x)`: `x_pdf = (1.0^-3)*(0.1^-3) * ones(size(x,1), 1);` or `x_pdf = mu_post.pdf(x(:, 1:3)) * (0.1^-3) * ones(size(x,1), 1);`
    - `proprnd(x) = mvnrnd(x,covmat,1) while true keep doing until if fT(x) then break;`
    - `[thetaj1(i,:), acceptance_rate] = mhsample(thetaj(idx(i), :), 1, 'logpdf',  log_fj1, 'proppdf', proppdf, 'proprnd', proprnd, 'thin', 3,  'burnin',  burnin);` burnin = 10
  - 公式解释：
    - 使用提议分布生成新样本作为候选样本$\theta^*{\sim}q(\theta^*|\theta^{(j)})=N(\theta^{(j)},\Sigma)$，提议分布(以之前样本为均值，以$\sum_{j}$为协方差的联合高斯分布)
    - 得到候选样本后，$\alpha=\min\left[\frac{P(\mathbf{Y}_{exp}|\theta^*)}{P(\mathbf{Y}_{exp}|\theta^{(j)})},1\right]$，然后使用均匀分布采样$u{\sim}Uniform(0,1)$，接受or拒绝：$\theta^{(j+1)}=\begin{cases}\theta^*,\alpha\geq u\\\theta^{(j)},\alpha<u\end{cases}$

不断循环上述过程，直到$p_{j} \geq 1$，得到posterior多参数的N个样本


> [Metropolis-Hastings sample - MATLAB mhsample - MathWorks 中国](https://ww2.mathworks.cn/help/stats/mhsample.html)
> 
> w.p.：with probability


>  [mukeshramancha/transitional-mcmc: This repo contains the code of Transitional Markov chain Monte Carlo algorithm](https://github.com/mukeshramancha/transitional-mcmc)
>  [civiltechnocrats.wordpress.com/wp-content/uploads/2013/11/bayesian-methods-for-structural-dynamics-and-civil-engineering.pdf#page=53.49](https://civiltechnocrats.wordpress.com/wp-content/uploads/2013/11/bayesian-methods-for-structural-dynamics-and-civil-engineering.pdf#page=53.49)


example 2DOF system: 
- `main_2DOF.py/log_likelihood`
  - input: 每个参数值s, sig1 $\sigma_{1}$,sig2 $\sigma_{2}$
  - 根据参数s计算对应的响应$\lambda_{1}, \phi_{1}^{2}$
  - output: $LL=log\_likelihood = \log((2\pi \sigma_{1} \sigma_{2})^{-5}) + -\frac{1}{2}\sigma_{1}^{-2}\sum(\lambda_{1}-\lambda_{1\exp}) +$
- `tmcmc.py/compute_beta_update_evidence`
  - $w = exp^{ (LL -LL_{max}) \times (p_{j+1} - p_j)}$
  - $w_{n} = \frac{w}{\sum w}$
  - $ESS = \frac{1}{\sum w_{n}^{2}}$

```python
LL = (np.log((2*np.pi*sig1*sig2)**-5) +
      (-0.5*(sig1**(-2))*sum((lambda1_s-data1)**2)) +
      (-0.5*(sig2**-2)*sum((phi12_s-data3)**2)))
```

NASA subA:
- $LL = log\_likelihood = -\frac{UQ^{2}}{\varepsilon^{2}}$
- `tmcmc.py/compute_beta_update_evidence` $\mathrm{LogSumExp}(x_1\ldots x_n)=\log\left(\sum_{i=1}^ne^{x_i}\right)$
  - $\log w= (p_{j+1} - p_j) \times LL$
  - $\log w_{n}= \log w-\log\left( \sum e^{\log w} \right)$
  - $ESS = e^{-\log \sum e^{2\log w_{n}}}$

