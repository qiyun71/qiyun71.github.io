---
title: Math
date: 2024-03-29 22:24:11
tags: 
categories: Learn
---
概率/微积分/矩阵/拓扑/泛函/复变...

MATH $e^{i\pi}+1=0$

<!-- more -->

# 泛函

> [(72 封私信 / 80 条消息) 「泛函」究竟是什么意思？ - 知乎](https://www.zhihu.com/question/21938224)
> [Calculus of Variations：变分计算 - 分享我的学习心得](https://www.pynumerical.com/archives/48/) 变分：自变量x不变，函数$y(\cdot)$改变

研究不同的函数，对输出的影响。
例如MLP要拟合一个函数，让预测的输出与标签非常相近。又如有限元求解偏微分方程，是要根据**最小作用量原理**，求得一个满足边界条件的、相对准确的近似解

## 范数

> [L0,L1,L2范数（双竖线，有下标）_数学公式两对竖线右下角加个2-CSDN博客](https://blog.csdn.net/u013066730/article/details/83013885)


# 复变函数

## 拉普拉斯变换

(传递函数必用)

[拉普拉斯变换 - 维基百科，自由的百科全书](https://zh.wikipedia.org/wiki/%E6%8B%89%E6%99%AE%E6%8B%89%E6%96%AF%E5%8F%98%E6%8D%A2)

$F(s)=\int_0^\infty e^{-st}f(t)\mathrm{d}t$

## 柯西积分公式

[柯西积分公式 - 维基百科，自由的百科全书](https://zh.wikipedia.org/zh-cn/%E6%9F%AF%E8%A5%BF%E7%A9%8D%E5%88%86%E5%85%AC%E5%BC%8F)

$f(a)=\frac{1}{2\pi i}\oint_\gamma\frac{f(z)}{z-a}dz.$

## 留数

[留数 - 维基百科，自由的百科全书](https://zh.wikipedia.org/wiki/%E7%95%99%E6%95%B0)


[zh.wikipedia.org/wiki/留数定理](https://zh.wikipedia.org/wiki/%E7%95%99%E6%95%B0%E5%AE%9A%E7%90%86)
计算解析函数沿着闭曲线的路径积分的一个有力的工具，也可以用来计算实函数的积分

[从零开始的留数定理 - Monsoon的文章 - 知乎](https://zhuanlan.zhihu.com/p/708571624)





# 数学基础

## 帕斯瓦定理

信号的能量在时域和频域是一样的
$\int_{-\infty}^{+\infty}|x(t)|^2dt=\frac{1}{2\pi}\int_{-\infty}^{+\infty}|F_x(\omega)|^2d\omega=\int_{-\infty}^{+\infty}|F_x(2\pi f)|^2df$

## 偏微分方程

拉普拉斯算子 $\Delta = \frac{\partial^{2}}{\partial x^{2}} + \frac{\partial^{2}}{\partial y^{2}}$

Laplace eqn: $\Delta u = u_{xx}+u_{yy}=\nabla \cdot \nabla u = \nabla ^{2} u = 0$
Possion eqn: $\Delta u = F(x,y)$

二维平面D，边界为$\partial D$：$u(x,y)$
边界条件：$(x,y) \in \partial D$
- Dirichlet：$u(x,y)=g(x,y)$
- Neumann：$\partial n u(x,y)=g(x,y)$ $where \partial n = \hat n \cdot \nabla，\hat{n}$为表面法向量
- Robin：$u(x,y)+\alpha(x,y)\partial n u(x,y)=g(x,y)$

## 卷积

> [但什么是卷积呢？ - YouTube](https://www.youtube.com/watch?v=KuXjwB4LzSA)

$f\left(t\right)*g\left(t\right)=\int_{0}^{t}f\left(\tau\right)g\left(t-\tau\right)d\tau$ 时间信号的时间一般是从0开始的

Example1：已知$f(x) = a_{0}+a_{1}x+\dots+a_{n}x^{n}$ 和 $g(x)=b_{0}+b_{1}x+\dots b_{n}x^{n}$，求$h(x)=f(x) \cdot g(x)$
- $h(x)$的系数c是ab两系数的卷积结果：直接计算的话时间复杂度为$\mathcal{O}(n^{2})$

$$\left.\mathbf{a}*\mathbf{b}=\left[\begin{array}{c}a_0b_0,\\a_0b_1+a_1b_0,\\a_0b_2+a_1b_1+a_2b_0,\\\vdots\\a_{n-1}b_{m-1}\end{array}\right.\right]$$

- 另一种思路是先将$f(x)$于$g(x)$进行FFT$f(\omega),g(\omega)$，频域的系数$\hat{\mathbf{a}}=[\hat{a}_0,\hat{a}_1,\hat{a}_2,\ldots,\hat{a}_{m+n-1}]$ 和$\hat{\mathbf{b}}=[\hat{b}_0,\hat{b}_1,\hat{b}_2,\ldots,\hat{b}_{m+n-1}]$，两者直接相乘得到$\hat{\mathbf{a}}\cdot\hat{\mathbf{b}}=[\hat{a}_0\hat{b}_0,\hat{a}_1\hat{b}_1,\hat{a}_2\hat{b}_2,\ldots,]$，然后进行逆FFT，得到的$h(x)$系数即想要的结果，时间复杂度为$\mathcal{O}(n\log n)$

>[【官方双语】卷积的两种可视化|概率论中的X+Y既美妙又复杂_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1Yk4y1K7Az/?spm_id_from=333.999.0.0&vd_source=1dba7493016a36a32b27a14ed2891088) 最好的动画⭐


### 自相关

[Modal Testing](../Finite%20Element/Modal%20Testing.md)

## 摄动法

> [轻微的扰动——摄动法简介(1) - 科学空间|Scientific Spaces](https://kexue.fm/archives/1878) 

## 泰勒

> [dezeming.top/wp-content/uploads/2021/06/多元函数（及向量函数）的泰勒展开.pdf](https://dezeming.top/wp-content/uploads/2021/06/%E5%A4%9A%E5%85%83%E5%87%BD%E6%95%B0%EF%BC%88%E5%8F%8A%E5%90%91%E9%87%8F%E5%87%BD%E6%95%B0%EF%BC%89%E7%9A%84%E6%B3%B0%E5%8B%92%E5%B1%95%E5%BC%80.pdf)

泰勒展开本质上是求近似

## Domain Transform

> [The intuition behind Fourier and Laplace transforms I was never taught in school - YouTube](https://www.youtube.com/watch?v=3gjJDuCAEQQ)

### Fourier Transform

>[频域处理Frequency domain processing：傅里叶变换 | YangWC's Blog](https://yangwc.com/2019/10/24/FFT/)


$Fourier: F(\omega)=\int_{-\infty}^{\infty}f(t)e^{-i\omega t}dt$
$F(\omega)=\int_{-\infty}^{\infty}f(t)\cos{(\omega t)}dt-i\int_{-\infty}^{\infty}f(t)\sin{(\omega t)}dt$ 

傅里叶变换可以看作使用不同频率$\omega$的sin和cos对原始时域信号进行积分，sin和cos积分得到的值越大，则虚部和实部的值越大，直观的模/振幅 magnitude 也越大，该频率$\omega$成分下的值也越大

对于信号$cos\pi t$ 进行FT，其虚部$\cos \pi t \cdot \sin \omega t$一直为0，实部$\sin \pi t \cdot \sin \omega t$在其他地方为0，在$\omega =\pi$时达到正无穷

#### DFT

>[Understanding The Discrete Fourier Transform « The blog at the bottom of the sea](https://blog.demofox.org/2016/08/11/understanding-the-discrete-fourier-transform/)

$X_{k}=\sum_{n=0}^{N-1}x_{n}\cdot e^{-2\pi ikn/N}$



#### FFT

#### STFT

>[短时傅里叶变换和小波变换有何不同？ - Mr.看海的回答 - 知乎](https://www.zhihu.com/question/26673889/answer/3292923354)

可见当信号的频率成分随时间显著变化时，即所谓的非平稳信号，传统频谱分析就不太合适了。
举几个例子：
- 研究信号的局部特性：如爆炸声、机器的故障噪声等，这些信号的特性在短时间内变化很大。
- 语音处理：在自动语音识别和语音合成中，语音的特征如音高和音量随着时间而变化。
- 雷达和无线电信号分析：雷达信号的特性依赖于时间和观测的角度，需分析这些信号随时间的变化。
- 生物医学信号分析：比如心电图（ECG）和脑电图（EEG）这样的生理信号，它们的频率特性随时间改变。

![image.png|333](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20241001123745.png)


在短时傅里叶变换（STFT）中，窗口函数及其大小选择是分析的关键。窗口函数决定了在任何给定时间点，信号的哪一部分被用于分析。窗口大小的选择直接影响了分析结果的时间分辨率和频率分辨率，这是进行有效STFT分析的最重要的权衡。

窗口大小的时间分辨率影响：时间分辨率与窗口的宽度密切相关。一个窄窗口提供较高的时间分辨率，因为它捕捉了信号在很短时间内的变化。这对于分析包含快速变化的瞬态事件，如敲击声或爆炸声，是非常有用的。然而，较小的窗口将限制频率分辨率，因为频率分析需要足够的周期来准确估计。

窗口大小的频率分辨率影响：频率分辨率与窗口的宽度呈反比。一个宽窗口覆盖了信号的较长时间段，提供了较高的频率分辨率。这是因为更多的周期可以在窗口内被分析，从而更准确地确定低频成分。但是，这会牺牲时间分辨率，因为窗口中的信号被假定在这段时间内是平稳的。

**那有没有一种可能，窗口大小是可调的呢？**

#### Wavelet Transform

可以发现其特点：高频部分具有较高的时间分辨率和较低的频率分辨率，而低频部分具有较高的频率分辨率和较低的时间分辨率，这就恰好解决了STFT的痛点


### Laplace Transform

> [What does the Laplace Transform really tell us? A visual explanation (plus applications) - YouTube](https://www.youtube.com/watch?v=n2y7n6jw5d0)


$X(s)=\int_0^\infty x(t)e^{-st}dt$

$Laplace:F(s)=\int_0^\infty f(t)e^{-st}dt$

$s=\alpha+i\omega$

$F(s)=\int_0^\infty f(t)e^{-i\omega t}e^{-\alpha t}dt$
- $e^{-i\omega t}$ scans for sinusoids
- $e^{-\alpha t}$ scans for exponentials

时域的微积分在s域中可以很好地计算

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240925100202.png)


## 复数 Complex Number

> [傅里叶变换(Fourier Transform) | Long Luo's Life Notes](https://www.longluo.me/blog/2021/12/29/fourier-transform/)

在复平面上，1 有 n 个不同的 n 次方根，它们位于复平面的单位圆上，构成**正多边形的顶点**，但最多只可有两个顶点同时标在实数线上

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240921212004.png)


## KL 散度

> [KL 散度（Kullback-Leibler Divergence）：图示+公式+代码](https://www.vectorexplore.com/tech/loss-functions/kl-divergence/)
> [机器学习中的散度 - 知乎](https://zhuanlan.zhihu.com/p/45131536)

概率分布P的概率密度函数为 $p(x_i)$ 
信息理论的主要目标是量化数据中的信息量。信息理论中最重要的度量标准称为熵（Entropy），通常表示为 H其信息熵：$H(X)=-\sum_{i=1}^np(x_i)\log p(x_i)$ , ***可以将-log(p)看成权重，概率p越大，权重越小*** (如果有人告诉我们一个相当不可能的事件发生了，我们收到的信息要多于我们被告知某个很可能发生的事件发生时收到的信息。如果我们知道某件事情一定会发生，那么我们就不会接收到信息。)
- 如果我们在计算中使用 $log_{2}$​ 我们可以将熵解释为“编码我们的信息所需的最小比特数”。
  - 【例1】A、B、C、D 四个字母分别占 1/2（4096个），1/4（2048个），1/8（1024个），1/8（1024个）。那么最有效的一种编码方式为 A(0)，B（10），C（110），D(111)。整个语料库的长度 4096 x 1 + 2048 x 2 + 1024 x 3 x 2 = 14336，平均长度为 14336/8192 = 1.75。和下面代码中的【结果1】一致。*如果使用另一种编码方式(例2)即  A（110），B（111），C（0），D（10），则熵为(4096 * 3 + 2048 * 3 + 1024 * 1 + 1024 * 2 )/8192 = 2.625*
    - 其中如果p(x)=0.5，则只需要$-log_{2}(0.5)=1$个bit的编码；如果p(x)=0.25，则需要2 bit的编码，以此类推
    - 依照这种规律的编码，可以保证信息中的熵最小
  - 【例2】ABCD 四个字母占比变成了1/8（1024 个），1/8（1024 个），1/2（4096 个），1/4（2048 个），这样最有效的一种编码方式为 A（110），B（111），C（0），D（10），计算平均长度为1.75，和代码中的【结果3】一致。 *如果使用另一种编码方式(例1)即  A(0)，B（10），C（110），D(111)，则熵为(1024 * 1 + 1024 * 2 + 4096 * 3 + 2048 * 3 )/8192 = 2.625*
  - 相对熵(KL散度)：
    - 例1 相对于 例2 的相对熵为 $p(x)\log_{2}\left( \frac{p(x)}{q(x)} \right)$=(4096 * -(1-3) + 2048 * -(2-3) + 1024 * -(3-1) + 1024 * -(3-2) )/8192 = 0.875 = 2.625 - 1.75。 这说明针对例子中的这一分布，使用法1编码相对于法2来说，平均可以省下0.875bit
    - 例2 相对于 例1 的相对熵为 $q(x)\log_{2}\left( \frac{q(x)}{p(x)} \right)$=(1024 * -(3-1) + 1024 * -(3-2) + 4096 * -(1-3) + 2048 * -(2-3) )/8192 =0.875 = 2.625 - 1.75
    - 两个例子尽管编码方式不同，但字母的频率分布是一致的。因此，在计算KL散度时，由于 p(x) 和 q(x) 实际上是相同的，导致 KL散度的两个方向相等。**为什么是0.875而不是0？** 这说明即使两种编码方案都能描述相同的概率分布，**但由于它们在信息表示方式上的差异**，导致了每种编码方式在另一种分布下都有额外的冗余。KL散度反映了这种冗余：即使分布相同，编码方式的差异会导致你用一种方式来编码另一种信息时需要额外的0.875 bits。**KL散度真正衡量的是概率分布之间的差异，而不是编码方案之间的差异。**

KL散度的值越大，表示用一个分布近似另一个分布时引入的信息损失或误差越大：
- 非对称性：KL散度是非对称的，即从 P 分布到 Q 分布的 KL 散度与从 Q 分布到 P 分布的 KL 散度可能不同。
  - 虽然KL散度通常是非对称的，但在特定条件下**KL散度从 P 到 Q** 与 **从 Q 到 P** 的值可以相等：
    - **在某些特定的离散分布情况下**，当 P(x)和 Q(x)的概率质量函数具有某种对称结构时，它们的KL散度可能相等（但这仍是概率分布的特殊情况，而不是普遍现象）。
    - 。当且仅当两个概率分布完全相同时，KL散度的值为零
- 非负性：KL散度的值始终为非负数。KL散度值越大，表示两个概率分布越不相似。
- 非度量性：KL散度并不满足度量空间的性质，特别是三角不等式。由于非对称性和非度量性，KL 散度不能用于计算两个分布之间的“距离”或“相似度”。

>[KL 散度（相对熵） - 小时百科 (wuli.wiki)](https://wuli.wiki/online/KLD.html)

**KL 散度**（Kullback–Leibler divergence，缩写 KLD）是一种统计学度量，**表示的是一个概率分布相对于另一个概率分布的差异程度**，在信息论中又称为**相对熵**（Relative entropy）。对于随机变量Q的概率分布，相对于随机变量P的概率分布的KL散度定义为： $D_{KL}(P||Q)=H(P,Q)-H(P)$
$$\begin{equation}
D_{KL}(P||Q)=\sum_{x\in X}P(x)ln(\frac{P(x)}{Q(x)})=\sum_{x\in X}P(x)(ln(P(x))-ln(Q(x)))~.
\end{equation}$$

对于连续型随机变量，设概率空间 X 上有两个概率分布 P 和 Q，其概率密度分别为 p 和 q，那么，P 相对于 Q 的 KL 散度定义如下：
$$\begin{equation}
D_{KL}(P||Q)=\int_{-\infty}^{+\infty}p(x)ln(\frac{p(x)}{q(x)})dx~.
\end{equation}$$

 显然，当 P=Q 时，$D_{KL}=0$

两个一维高斯分布的 KL 散度公式：
> [KL散度(Kullback-Leibler Divergence)介绍及详细公式推导 | HsinJhao's Blogs](https://hsinjhao.github.io/2019/05/22/KL-DivergenceIntroduction/)

$$\begin{aligned}
KL(p,q)& =\int[\left.p(x)\log(p(x))-p(x)\log(q(x))\right]dx  \\
&=-\frac12\left[1+\log(2\pi\sigma_1^2)\right]-\left[-\frac12\log(2\pi\sigma_2^2)-\frac{\sigma_1^2+(\mu_1-\mu_2)^2}{2\sigma_2^2}\right] \\
&=\log\frac{\sigma_2}{\sigma_1}+\frac{\sigma_1^2+(\mu_1-\mu_2)^2}{2\sigma_2^2}-\frac12
\end{aligned}$$

## 凸函数与Jensen不等式
凸函数是一个定义在某个向量空间的凸子集 C（区间）上的实值函数 f，如果在其定义域 C 上的任意两点$x_{1},x_{2}$，$0\leq t\leq 1$，有：
满足$tf(x_1)+(1-t)f(x_2)\geq f\left(tx_1+(1-t)x_2\right)$
也就是说凸函数任意两点的割线位于函数图形上方， **这也是Jensen不等式的两点形式**


若对于任意点集 $\{x_{i}\}$，若 $\lambda_i≥0$且 $∑_{i}\lambda_{i}=1$ ，使用**数学归纳法**，可以证明凸函数 f (x) 满足：
$f(\sum_{i=1}^M\lambda_ix_i)\leq\sum_{i=1}^M\lambda_if(x_i)$ 即为Jesen不等式

**在概率论中**，如果把 $\lambda_i$ 看成取值为 $x_{i}$的离散变量 x 的概率分布$p_{i}$，那么公式(2)就可以写成
$f(E[x])\leq E[f(x)]$ , $E[\cdot]$代表期望

对于连续变量，Jensen不等式给出了积分的凸函数值和凸函数的积分值间的关系：
$f(\int xp(x)dx)\leq\int f(x)p(x)dx$


# 概率论

## Basic

### 均值方差

为什么样本估计方差要除以n-1 [【浅谈】样本方差的分母“n”为什么要改为“n-1” - 知乎](https://zhuanlan.zhihu.com/p/550427703)

其实就是自由度，当你算标准差的时候，已经知道均值了，那么就只有n-1个数字是自由的，第n个数值可以由前面的n-1个数字和均值算出来了，所以他实际上不包含任何关于数据波动的信息

无偏估计的方差：
$\left\{\begin{array}{c}M_n=\frac{X_1+X_2+\cdots+X_n}n\\\hat{S}_n^2=\frac{\sum_{i=1}^n(X_i-M_n)^2}{n-1}\end{array}\right.$

[【AP统计】期望E(X)与方差Var(X) - 知乎](https://zhuanlan.zhihu.com/p/64859161)


### 概率/似然

> [通俗理解“极大似然估计” - 知乎](https://zhuanlan.zhihu.com/p/334890990)

概率函数：由因到果，已知参数(概率)，根据真实参数（或已经发生的观测结果）去推测未来的观测结果
似然函数：由果到因，根据已经发生的观测结果去猜想真实参数，这个过程叫做**估计**；估计正确的可能性叫做**似然性**。估计参数的似然性，其目的是帮助我们根据已观测的结果，推测出最符合观测结果、最合理的参数。

假设一个参数p，则在这一参数基础上出现结果的概率即为似然
$$\begin{aligned}
&L(p)=L(p|x_1,\ldots,x_n) \\
&=P(X_1=x_1|p)•\ldots•P(X_n=x_n|p) \\
&=\prod_{i=1}^nP(X_i=x_i|p)
\end{aligned}$$

极大似然估计（maximum likelihood estimation，缩写为MLE），也称[最大似然估计](https://zhida.zhihu.com/search?content_id=162550522&content_type=Article&match_order=2&q=%E6%9C%80%E5%A4%A7%E4%BC%BC%E7%84%B6%E4%BC%B0%E8%AE%A1&zhida_source=entity)。
$\arg\max_pL(p)=\arg\max_p\prod_{i=1}^nP(X_i=x_i|p)$
将$L(p)$看作p的函数，对其求导，导数为0，即可得到最大的似然值对应的参数p

求导前为什么要取对数？：**（1）避免下溢出** **（2）便于计算** 将累积乘法转换成累加


## Bayes

![GZsfHmbaAAE1ZFD (1264×1128)|555](https://pbs.twimg.com/media/GZsfHmbaAAE1ZFD?format=jpg&name=large)

Follow:

> [KERO - 知乎](https://www.zhihu.com/people/chenxran0916/posts)

- 最大似然

最大似然估计(Maximum Likelihood Estimation, MLE)，就是利用已知的样本结果信息，反推最具有可能（最大概率）导致这些样本结果出现的模型参数值！样本从某一个客观存在的模型中抽样得来，然后根据样本来计算该模型的数学参数，即：模型已定，参数未知！

$$\begin{aligned}
\widehat{\theta}_{\mathrm{MLE}}& =\arg\max P(X;\theta)  \\
&=\arg\max P(x_1;\theta)P(x_2;\theta)\cdot\cdots P(x_n;\theta) \\
&=\arg\max\log\prod_{i=1}^nP(x_i;\theta) \\
&=\arg\max\sum_{i=1}^n\log P(x_i;\theta) \\
&=\arg\min-\sum_{i=1}^n\log P(x_i;\theta)
\end{aligned}$$

- 最大后验

$$\begin{aligned}
\hat{\theta}_{\mathrm{MAP}}& =\arg\max P(\theta|X)  \\
&=\arg\min-\log P(\theta|X) \\
&=\arg\min-\log P(X|\theta)-\log P(\theta)+\log P(X) \\
&=\arg\min-\log P(X|\theta)-\log P(\theta)
\end{aligned}$$



> [贝叶斯定理，改变信念的几何学 - YouTube](https://www.youtube.com/watch?v=HZGCoVF3YvM) ——数形结合
> [走进贝叶斯统计（一）—— 先验分布与后验分布 - 知乎](https://zhuanlan.zhihu.com/p/401258319)
> [超详细讲解贝叶斯网络(Bayesian network) - USTC丶ZCC - 博客园](https://www.cnblogs.com/USTC-ZCC/p/12786860.html)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20240316151511.png)


全概率公式：$\mathbf{P(H)}=\mathbf{P(H|A)P(A)}+\mathbf{P(H|B)P(B)}$，结果H发生的概率

贝叶斯公式：$\mathbf{P}(\mathbf{A}|\mathbf{H})=\frac{P(A)P(H|A)}{P(H)}$，H结果发生时，是由A导致的概率
- 连续$p(y_0|x)=\frac{p(x|y_0)p(y_0)}{\int_{-\infty}^{+\infty}p(x|y)p(y)dy}$
- 离散$p(y_j|x)=\frac{p(x|y_j)p(y_j)}{\sum_{i=0}^np(x|y_i)p(y_i)}$

在使用数据估计参数$\theta$之前，我们需要给这个参数设定一个分布，即先验分布$p(\theta)$（根据经验得到）

$p(\theta|X)=\frac{p(\theta,X)}{p(X)}=\frac{p(X|\theta)p(\theta)}{\int_{-\infty}^{+\infty}p(X|\theta)p(\theta)d\theta}.$
- $p(\theta|X)$是$\theta$的后验分布
- $p(X|\theta)$是在给定$\theta$下关于数据样本的似然函数
- $\int_{-\infty}^{+\infty}p(X|\theta)p(\theta)d\theta$ 为常数c，可以写为$p(\theta|X)\propto p(X|\theta)p(\theta).$

## Distribution

### 核密度估计

> [核密度估计 (KDE, Kernel Density Estimation) | Blog](https://viruspc.github.io/blog/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/2021/09/27/kde.html)



### Gamma 分布

> [Gamma distribution - Wikipedia](https://en.wikipedia.org/wiki/Gamma_distribution)

Gamma Function：$\Gamma(z)=\int_0^\infty t^{z-1}e^{-t}\mathrm{d}t,\quad\Re(z)>0.$ or $\Gamma(n)=(n-1)!.$

$f(x)=\frac{\beta^{\alpha}}{\Gamma(\alpha)}x^{\alpha-1}e^{-\beta x}$

图中k对应$\alpha$，$\theta$对应$\beta$

![image.png|333](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240328193106.png)

### Beta 分布

> [Beta distribution - Wikipedia](https://en.wikipedia.org/wiki/Beta_distribution)

通常是概率分布的分布

$\mathbf{B}(\alpha,\beta)={\frac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha+\beta)}}$

![image.png|333](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240328193446.png)

### 高斯分布
$X\sim N(\mu,\sigma^2)$

PDF：$f(x)=\frac1{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$


### 多元高斯分布

> [多元/多维高斯/正态分布概率密度函数推导 (Derivation of the Multivariate/Multidimensional Normal/Gaussian Density) - 凯鲁嘎吉 - 博客园](https://www.cnblogs.com/kailugaji/p/15542845.html)
> [多元高斯分布（The Multivariate normal distribution） - bingjianing - 博客园](https://www.cnblogs.com/bingjianing/p/9117330.html)

概率密度函数
$p(x)=p(x_{1},x_{2},\ldots,x_{D})=\frac{1}{(2\pi)^{D/2}}\exp\left(-\frac{1}{2}(x-\mu)^{T}\Sigma^{-1}(x-\mu)\right)$


### 混合高斯分布

GMM

### 联合高斯分布

如何构造两个协方差标准分布

> [生成一定相关性的二元正态分布_怎么产生二元正态分布的随机数-CSDN博客](https://blog.csdn.net/kdazhe/article/details/104599229)

AB为两个独立的标准正态分布：

$\begin{aligned}\mathrm{X}&=\alpha\mathrm{A}+\beta\mathrm{B},\\\\\mathrm{Y}&=\gamma\mathrm{A}+\delta\mathrm{B}\end{aligned}$

$\alpha,\beta,\gamma,\delta$是四个待确定的参数，希望找到这四个参数，使得X和Y也服从标准正态分布$N(0,1)$，并且相关系数$Cor(X,Y)=\rho$。

由标准正态分布性质可得：$\mathrm{X}\sim\mathrm{N}(0,\alpha^2+\beta^2),\mathrm{Y}\sim\mathrm{N}(0,\gamma^2+\delta^2)。$
要想：$\bar{\alpha}^{2}+\beta^{2}=1,\gamma^{2}+\delta^{2}=1$
则可以假设：$\alpha=\cos\theta,\beta=\sin\theta;\gamma=\sin\theta,\delta=\cos\theta,$

由相关系数：$\mathrm{Cor(X,~Y)}=\frac{\mathrm{Cov(X,~Y)}}{\sqrt{\mathrm{Var(X)Var(Y)}}}$，$\operatorname{Var}(\mathrm{X})=1$，$\operatorname{Var}(\mathrm{Y})=1$
因此使得：
$$\begin{aligned}
\mathrm{Cov}(\mathrm{X,Y})& =\mathrm{Cov}\Big((\cos\theta)\mathrm{A}+(\sin\theta)\mathrm{B},(\sin\theta)\mathrm{A}+(\cos\theta)\mathrm{B}\Big)  \\
&=\cos\theta\sin\theta+\sin\theta\cos\theta  \\
&=\sin2\theta 
\end{aligned}$$
- （$Cov(A,A)=1,Cov(A,B)=0$）
- $\text{Cov}(x_1 + x_2, y_1 + y_2) = \text{Cov}(x_1, y_1) + \text{Cov}(x_2, y_1) + \text{Cov}(x_1, y_2) + \text{Cov}(x_2, y_2)$
- $\mathrm{Cov}(aX,Y)=a\mathrm{Cov}(X,Y)$

则有：$\theta=\frac{\arcsin\rho}2$，然后令$\begin{cases}\mathrm X=(\cos\theta)\mathrm A+(\sin\theta)\mathrm B\\\mathrm Y=(\sin\theta)\mathrm A+(\cos\theta)\mathrm B\end{cases}$，则计算出来的X和Y就是相关性为$\rho$的标准正态分布

```python
import numpy as np
import matplotlib.pyplot as plt

class bivariateNormal:
    
    def __init__(self, rho: 'float', m: int):
        """
        Suppose we want to generate a pair of 
        random variables X, Y, with X ~ N(0, 1), 
        Y ~ N(0, 1), and Cor(X, Y) = rho. m is 
        the number of data pairs we want to generate.
        """
        self.rho = rho
        self.m = m
    
    def generateBivariate(self) -> 'tuple(np.array, np.array)':
        """
        Generate two random variables X, Y, with X ~ N(0, 1), 
        Y ~ N(0, 1), and Cor(X, Y) = rho. 
        self.m is the number of sample points we generated.
        We return a tuple (X, Y). 
        """
        theta = np.arcsin(self.rho) / 2
        A = np.random.normal(0, 1, self.m)
        B = np.random.normal(0, 1, self.m)
        X = np.cos(theta) * A + np.sin(theta) * B
        Y = np.sin(theta) * A + np.cos(theta) * B
        return X, Y

m = 10 ** 3
rho = -0.4
a = bivariateNormal(rho, m)
X, Y = a.generateBivariate()
np.corrcoef(X, Y)
```


### P-box

## Markov Chains

> [Markov Chains explained visually](https://setosa.io/ev/markov-chains/) 入门





## Sampling

如何在不知道目标概率密度函数的情况下，抽取所需数量的样本，使得这些样本符合目标概率密度函数。这个问题简称为抽样

> [Finite Element Model Updating in Bridge Structures Using Kriging Model and Latin Hypercube Sampling Method - Wu - 2018 - Advances in Civil Engineering - Wiley Online Library](https://onlinelibrary.wiley.com/doi/full/10.1155/2018/8980756) 不同采样方法的讨论(simple random sampling (SRS), stratified sampling method, cluster sampling method, and systematic sampling) **Latin Hypercube Sampling 属于 =分层采样**
> [It's all about Sampling - 子淳的博客 | Just Me](https://huangc.top/2019/03/24/sampling-2019/)

### Monte Carlo

>[一文看懂蒙特卡洛采样方法 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/338103692)
>[简明例析蒙特卡洛（Monte Carlo）抽样方法 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/39628670)
>[走进贝叶斯统计（四）—— 蒙特卡洛方法 - 知乎](https://zhuanlan.zhihu.com/p/406256344)
>[逆变换采样和拒绝采样 - barwe - 博客园 (cnblogs.com)](https://www.cnblogs.com/barwe/p/14140681.html)
>[Monte Carlo method - Wikipedia](https://en.wikipedia.org/wiki/Monte_Carlo_method)
>[【数之道 22】巧妙使用"接受-拒绝"方法，玩转复杂分布抽样 - YouTube](https://www.youtube.com/watch?v=c2WrJY8tnGE)

MC Sampling
![v2-eb0945aa2185df958f4568e58300e77a_1440w.gif|222](https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/v2-eb0945aa2185df958f4568e58300e77a_1440w.gif)

对某一种概率分布p(x)进行蒙特卡洛采样的方法主要分为直接采样、拒绝采样与重要性采样三种:
- Naive Method
    - 根据概率分布进行采样。对一个已知概率密度函数与累积概率密度函数的概率分布，我们可以直接从累积分布函数（cdf）进行采样（类似逆变换采样）
- Acceptance-Rejection Method 
    - 逆变换采样虽然简单有效，但是当累积分布函数或者反函数难求时却难以实施，可使用MC的接受拒绝采样
    - 对于累积分布函数未知的分布，我们可以采用接受-拒绝采样。如下图所示，p(z)是我们希望采样的分布，q(z)是我们提议的分布(proposal distribution)，令kq(z)>p(z)，我们首先在kq(z)中按照直接采样的方法采样粒子，接下来判断这个粒子落在途中什么区域，对于落在灰色区域的粒子予以拒绝，落在红线下的粒子接受，最终得到符合p(z)的N个粒子
    - ![image.png|333](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230801135729.png)
    - 数学推导：
        - ![image.png|500](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230801135800.png)
        1. 从 $f_r(x)$ 进行一次采样 $x_i$
        2. 计算 $x_i$ 的 **接受概率** $\alpha$（Acceptance Probability）:$\alpha=\frac{f\left(x_i\right)}{f_r\left(x_i\right)}$
        3. 从 (0,1) 均匀分布中进行一次采样 u
        4. 如果 $\alpha$≥u，接受 $x_i$ 作为一个来自 f(x) 的采样；否则，重复第1步

```python
N=1000 #number of samples needed
i = 1
X = np.array([])
while i < N:
    u = np.random.rand()
    x = (np.random.rand()-0.5)*8
    res = u < eval(x)/ref(x)
    if res:
        X = np.hstack((X,x[res])) #accept
        ++i
```

- **Importance Sampling**
    - 接受拒绝采样完美的解决了累积分布函数不可求时的采样问题。但是接受拒绝采样非常依赖于提议分布(proposal distribution)的选择，如果提议分布选择的不好，可能采样时间很长却获得很少满足分布的粒子。
    - $E_{p(x)}[f(x)]=\int_a^bf(x)\frac{p(x)}{q(x)}q(x)dx=E_{q(x)}[f(x)\frac{p(x)}{q(x)}]$
    - 我们从提议分布q(x)中采样大量粒子$x_1,x_2,...,x_n$，每个粒子的权重是 $\frac{p(x_i)}{q(x_i)}$，通过加权平均的方式可以计算出期望:
    - $E_{p(x)}[f(x)]=\frac{1}{N}\sum f(x_i)\frac{p(x_i)}{q(x_i)}$
        - q提议的分布，p希望的采样分布

```python
N=100000
M=5000
x = (np.random.rand(N)-0.5)*16
w_x = eval(x)/ref(x)
w_x = w_x/sum(w_x)
w_xc = np.cumsum(w_x) #accumulate

X=np.array([])
for i in range(M):
    u = np.random.rand()
    X = np.hstack((X,x[w_xc>u][0])) # 其中，w_xc是对归一化后的权重计算的累计分布概率。每次取最终样本时，都会先随机一个(0,1)之间的随机数，并使用这个累计分布概率做选择。样本的权重越大，被选中的概率就越高。
    
```

### 分层采样

#### 分层抖动采样

> [Physically Based Rendering：采样和重建（二） | YangWC's Blog](https://yangwc.com/2020/04/11/Sampling2/)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240722141841.png)

- 随机采样
- 分层均匀采样
- 分层抖动采样。理想的分层抖动采样很容易陷入**维数灾难**，因此有人提出了高维转低维采样+随机串联(or随机配对)的方法

#### Latin Hypercube Sampling

1. 将每个维度的区间划分为m个不重叠的区间，每个区间概率相等（取均匀分布，区间大小应相等）
2. 从均匀分布中随机采样，每个维度的每个间隔中的一个点
3. 将每个维度的点随机配对（相同可能的组合）

相较于Simple Random Sampling，LHS的方法更加分散，且不存在聚类效应


### MCMC

(Markov Chain Monte Carlo)

Blog:
> [马尔可夫链蒙特卡罗算法（MCMC） - 知乎](https://zhuanlan.zhihu.com/p/37121528)
> 动画 [The Markov-chain Monte Carlo Interactive Gallery](https://chi-feng.github.io/mcmc-demo/) | 代码 [Javascript demos](https://github.com/chi-feng/mcmc-demo)
> [MCMC](https://prappleizer.github.io/Tutorials/MCMC/MCMC_Tutorial.html) MCMC: A (very) Beginnner’s Guide

Paper:
> [An effective introduction to the Markov Chain Monte Carlo method](https://arxiv.org/pdf/2204.10145) **For physics**
> [A Conceptual Introduction to Markov Chain Monte Carlo Methods](https://arxiv.org/pdf/1909.12313)
> [Markov Chain Monte Carlo in Practice | W.R. Gilks, S. Richardson, Davi](https://www.taylorfrancis.com/books/mono/10.1201/b14835/markov-chain-monte-carlo-practice-david-spiegelhalter-gilks-richardson) MCMC需要小心地初始化，早期阶段需要warm-up time



#### M-H采样

> [走进贝叶斯统计（五）—— Metropolis-Hasting 算法 - 知乎](https://zhuanlan.zhihu.com/p/411689417)

#### Gibbs采样

> [走进贝叶斯统计（六）—— 吉布斯抽样 （Gibbs Sampling） - 知乎](https://zhuanlan.zhihu.com/p/416670115)

#### TMCMC

> [Transitional Markov Chain Monte Carlo Method for Bayesian Model Updating, Model Class Selection, and Model Averaging](Transitional%20Markov%20Chain%20Monte%20Carlo%20Method%20for%20Bayesian%20Model%20Updating,%20Model%20Class%20Selection,%20and%20Model%20Averaging.md)

#### 拉丁超立方采样(Latin hypercube sampling, LHS)

> [拉丁超立方采样(Latin hypercube sampling, LHS)及蒙特卡洛模拟简介 - 知乎](https://zhuanlan.zhihu.com/p/385966408)

拉丁超立方采样先把样本空间分层，在此问题下要分为5层，于是便有了 [1,20],[21,40],[41,60],[61,80],[81,100] 共5个样本空间，在各样本空间内进行随机抽样，然后再打乱顺序，得到结果。这样就结束了~

可以看出拉丁超立方采样分为了三步——**分层、采样、乱序**。

#### Langevin Monte Carlo(LMC)

>[什么是diffusion model? 它为什么好用？ - Zephyr的回答 - 知乎](https://www.zhihu.com/question/613579202/answer/3310826408)


#### Hamiltonian Monte Carlo(HMC)



## 置信区间

[68–95–99.7法则 - 维基百科，自由的百科全书](https://zh.wikipedia.org/wiki/68%E2%80%9395%E2%80%9399.7%E6%B3%95%E5%89%87)

![image.png|444](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240401085715.png)


## 图像展示

### 箱型图

> [如何深刻理解箱线图（boxplot） - 知乎](https://zhuanlan.zhihu.com/p/347067055)

### t-SNE 数据降维

> [降维方法之t-SNE - 知乎](https://zhuanlan.zhihu.com/p/426068503)
> [Everything About t-SNE. t-SNE means t-distribution Stochastic… | by Ram Thiagu | The Startup | Medium](https://medium.com/swlh/everything-about-t-sne-dde964f0a8c1)

PCA是一种常用的降维方式，但是不方便展示结果。例如对两类数据进行降维(二维)，PCA和t-SNE两种方法的展示结果为：

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240921142810.png)



# 线性代数

## 特殊矩阵定义

**正交矩阵** [正交矩阵 - 维基百科，自由的百科全书](https://zh.wikipedia.org/wiki/%E6%AD%A3%E4%BA%A4%E7%9F%A9%E9%98%B5)
- $Q^T=Q^{-1}\Leftrightarrow Q^TQ=QQ^T=I.$ 其中$I$为单位矩阵
- 正交矩阵的行列式值必定为+1或−1。
- 行列式值为+1的正交矩阵，称为**特殊正交矩阵**，它是一个旋转矩阵。
- 行列式值为-1的正交矩阵，称为瑕旋转矩阵。瑕旋转是旋转加上镜射。镜射也是一种瑕旋转。


**正定矩阵**[【线性代数】详解正定矩阵、实对称矩阵、矩阵特征值分解、矩阵 SVD 分解 - 知乎](https://zhuanlan.zhihu.com/p/234967628)
- 任意非零向量$\mathbf{x}$，若$\mathbf{x}^{T}\mathbf{A}\mathbf{x}>0$恒成立，则$\mathbf{A}$为正定矩阵。若$\mathbf{x}^{T}\mathbf{A}\mathbf{x}\geq0$恒成立，则$\mathbf{A}$为半正定矩阵
- 

## 特征分解

[特征分解 - 维基百科，自由的百科全书](https://zh.wikipedia.org/wiki/%E7%89%B9%E5%BE%81%E5%88%86%E8%A7%A3)

[奇异值分解（SVD） - 知乎](https://zhuanlan.zhihu.com/p/29846048)



## 变换

### 2D

仿射变换

### 3D

像素Pixel | 相机Camera | 世界World

内参矩阵 = c2p
外参矩阵 = w2c
根据世界坐标计算像素坐标 = `c2p * w2c * world_position`


### Homography

>[单应性Homography估计：从传统算法到深度学习 - 你再好好想想的文章 - 知乎](https://zhuanlan.zhihu.com/p/74597564)

单应性不严谨的定义：用 **[无镜头畸变]** 的相机从不同位置拍摄 **[同一平面物体]** 的图像之间存在单应性，可以用 **[透视变换]** 表示 。

刚体变换：平移+旋转，只改变物体位置，不改变物体形状。
$$\begin{pmatrix}x'\\y'\\1\end{pmatrix}=\begin{bmatrix}\mathrm{cos}\theta&-\mathrm{sin}\theta&t_x\\\mathrm{sin}\theta&\mathrm{cos}\theta&t_y\\0&0&1\end{bmatrix}\begin{pmatrix}x\\y\\1\end{pmatrix}=\begin{bmatrix}R_{2\times2}&T_{2\times1}\\0^T&1\end{bmatrix}\begin{pmatrix}x\\y\\1\end{pmatrix}$$

仿射变换：改变物体位置和形状，但是原来平行的边依然平行。

$$\begin{pmatrix}x'\\y'\\1\end{pmatrix}=\begin{bmatrix}A_{2\times2}&T_{2\times1}\\0^T&1\end{bmatrix}\begin{pmatrix}x\\y\\1\end{pmatrix}$$

透视变换（也称投影变换）：彻底改变物体位置和形状

$$\begin{pmatrix}x'\\y'\\1\end{pmatrix}=\begin{bmatrix}A_{2\times2}&T_{2\times1}\\V^T&s\end{bmatrix}\begin{pmatrix}x\\y\\1\end{pmatrix}=H_{3\times3}\begin{pmatrix}x\\y\\1\end{pmatrix}$$

# Computer Graphics 

## SDF计算与求导

空间中的子集$\partial\Omega$，SDF值定义为：$\left.f(x)=\left\{\begin{array}{ll}d(x,\partial\Omega)&\mathrm{~if~}x\in\Omega\\-d(x,\partial\Omega)&\mathrm{~if~}x\in\Omega^c\end{array}\right.\right.$
其中$d(x,\partial\Omega):=\inf_{y\in\partial\Omega}d(x,y)$表示x到表面子集上一点的距离，inf表示infimum最大下界



# Computer Vision

## 卷积图像大小计算公式

图像卷积后的大小计算公式： $N=\left\lfloor\frac{W-F+2P}{Step}\right\rfloor+1$
- 输入图片大小 $W \times W$
- Filter（卷积核）大小 $F \times F$
- 步长 Step
- padding（填充）的像素数 $P$
- 输出图片的大小为$N \times N$

## linearColor 2 sRGB

(Why)为什么要将线性RGB转换成sRGB

> [小tip: 了解LinearRGB和sRGB以及使用JS相互转换 « 张鑫旭-鑫空间-鑫生活 (zhangxinxu.com)](https://www.zhangxinxu.com/wordpress/2017/12/linear-rgb-srgb-js-convert/)

**人这种动物，对于真实世界的颜色感受，并不是线性的，而是曲线的**

![image.png|444](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230810162118.png)

(How)线性RGB与sRGB相互转化

> [RGB与Lab转换_rgb转lab-CSDN博客](https://blog.csdn.net/bby1987/article/details/109522126)

```python
def linear_to_srgb(linear):
    if isinstance(linear, torch.Tensor):
        """Assumes `linear` is in [0, 1], see https://en.wikipedia.org/wiki/SRGB."""
        eps = torch.finfo(torch.float32).eps
        srgb0 = 323 / 25 * linear
        srgb1 = (211 * torch.clamp(linear, min=eps)**(5 / 12) - 11) / 200
        return torch.where(linear <= 0.0031308, srgb0, srgb1)
    elif isinstance(linear, np.ndarray):
        eps = np.finfo(np.float32).eps
        srgb0 = 323 / 25 * linear
        srgb1 = (211 * np.maximum(eps, linear) ** (5 / 12) - 11) / 200
        return np.where(linear <= 0.0031308, srgb0, srgb1)
    else:
        raise NotImplementedError

def srgb_to_linear(srgb):
    if isinstance(srgb, torch.Tensor):
        """Assumes `srgb` is in [0, 1], see https://en.wikipedia.org/wiki/SRGB."""
        eps = torch.finfo(torch.float32).eps
        linear0 = 25 / 323 * srgb
        linear1 = torch.clamp(((200 * srgb + 11) / (211)), min=eps)**(12 / 5)
        return torch.where(srgb <= 0.04045, linear0, linear1)
    elif isinstance(srgb, np.ndarray):
        """Assumes `srgb` is in [0, 1], see https://en.wikipedia.org/wiki/SRGB."""
        eps = np.finfo(np.float32).eps
        linear0 = 25 / 323 * srgb
        linear1 = np.maximum(((200 * srgb + 11) / (211)), eps)**(12 / 5)
        return np.where(srgb <= 0.04045, linear0, linear1)
    else:
        raise NotImplementedError
```


# Other

[DeepLearning](DeepLearning.md)

## 数据处理方法

> [SOM(自组织映射神经网络)——理论篇 - 知乎](https://zhuanlan.zhihu.com/p/73534694)
> [自组织映射（SOM）理论基础与Python NumPy实现 | 美美智能博客站](https://nice-ai.top/posts/SelfOrganizingMaps-SOM.html)
> 针对数据量大的情况

自组织映射(Self-organizing map, SOM)通过学习输入空间中的数据，生成一个低维、离散的映射(Map)，从某种程度上也可看成一种降维算法。（*降维或者升为可以由输入和输出尺寸决定*，SOM输入为1E6，输出为70x70，本质上数据量变少，因此为降维算法）

## 希腊字母

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240503135053.png)
