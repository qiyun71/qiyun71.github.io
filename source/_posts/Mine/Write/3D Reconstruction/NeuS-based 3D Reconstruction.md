---
title: NeuS-based 3D Reconstruction
date: 2024-06-17 17:11:22
tags:
  - 
categories: Other/Mine/Write
---
| Title     | NeuS-based 3D Reconstruction |
| --------- | ---------------------------- |
| Author    |                              |
| Conf/Jour |                              |
| Year      |                              |
| Project   |                              |
| Paper     |                              |

- **Accuracy** 重建的模型精度不好，影响因素：
  - 数据集质量：照片拍摄质量(设备)、相机位姿估计精度(COLMAP)
    - 照片质量问题：混叠、模糊、滚动快门 (RS) 效应、HDR/LDR、运动模糊、低光照
  - NeuS方法的问题：Loss函数约束、体渲染的过度简化、缺少监督(*深度or法向量*)
  - 网格提取方法(Marching Cube)
- **Efficiency** 训练/渲染的速度太慢，影响因素：
  - MLP计算次数多
  - MLP层数多计算慢

<!-- more -->

# IDEA

- 监督深度图信息


# Basic of 3D Reconstruction based on NeuS

> [NeuS: Learning Neural Implicit Surfaces by Volume Rendering for Multi-view Reconstruction](https://arxiv.org/pdf/2106.10689)

SDF内部-1，外部1，表面0
NeuS没有与NeRF一样直接使用MLP输出的不透明度$\sigma$作为$\rho$，而是使用预测的sdf进行相应计算得到$\rho$，以及权重
- $C(\mathbf{o},\mathbf{v})=\int_{0}^{+\infty}w(t)c(\mathbf{p}(t),\mathbf{v})\mathrm{d}t$ 
- $\rho(t)=\max\left(\frac{-\frac{\mathrm{d}\Phi_s}{\mathrm{d}t}(f(\mathbf{p}(t)))}{\Phi_s(f(\mathbf{p}(t)))},0\right)$ , MLP预测的sdf即$f(\mathbf{p}(t))$
- $\phi_s(x) =\frac{se^{-sx}}{(1+e^{-sx})^{2}}$, $\Phi_s(x)=(1+e^{-sx})^{-1},\text{i.e.,}\phi_s(x)=\Phi_s'(x)$

除了$\mathcal{L}1$和$\mathcal{L}_{mask}$损失之外还使用了$\mathcal{L}_{r e g}=\frac{1}{n m}\sum_{k,i}(\|\nabla f(\hat{\mathbf{p}}_{k,i})\|_{2}-1)^{2}.$ (Eikonal term)

# Accuracy Improvement

## Camera Pose Estimation(Accuracy)

## 3D Point Sampling(Accuracy)

### 混合采样方式

NerfAcc：占据+逆变换采样
先使用占据网格确定哪些区域需要采样，再通过粗采样得到的权重使用逆变换采样进行精采样得到采样点

## Loss Function

### [S3IM: Stochastic Structural SIMilarity and Its Unreasonable Effectiveness for Neural Fields](https://arxiv.org/pdf/2308.07032)

MSE loss 是point-wise的，没有考虑到一组pixel的结构特征，而SSIM通过一个KxK的核和stride扫过整个图像，最后求平均MSSIM，可以考虑图像的结构信息。
- $\mathrm{SSIM}(a,b)=l(\boldsymbol{a},\boldsymbol{b})c(\boldsymbol{a},\boldsymbol{b})s(\boldsymbol{a},\boldsymbol{b}).$  一个核覆盖的图像，$C_{1},C_{2},C_{3}$ 是小的常数
  - luminance亮度：$l(\boldsymbol{a},\boldsymbol{b})=\frac{2\mu_a\mu_b+C_1}{\mu_a^2+\mu_b^2+C_1},$
  - contrast对比度：$c(\boldsymbol{a},\boldsymbol{b})=\frac{2\sigma_a\sigma_b+C_2}{\sigma_a^2+\sigma_b^2+C_2},$
  - structure结构：$s(\boldsymbol{a},\boldsymbol{b})=\frac{\sigma_{ab}+C_3}{\sigma_a\sigma_b+C_3}.$
*但是在NeRF训练过程中pixel在一个batch是随机的，丢失了局部patch的像素中位置相关的信息*，本文提出的S3IM，是SSIM的随机变体。每个minibatch有B个像素，核大小KxK，步长s=K(因为在minibatch中的随机patch是独立的，而且不需要重叠的情况)
- 将B个像素/光线构成一个rendered patch $\mathcal{P}(\hat{\mathcal{C}})$，同时有一个gt image patch $\mathcal{P}(\mathcal{C})$
- 计算rendered和gt patch之间的$SSIM(\mathcal{P}(\hat{\mathcal{C}}),\mathcal{P}(\mathcal{C}))$ with kernel size KxK and stride size s =K
- 由于patch是随机的，重复M次上述两步，并计算M次SSIM的平均值

$\mathrm{S3IM}(\hat{\mathcal{R}},\mathcal{R})=\frac{1}{M}\sum_{m=1}^{M}\mathrm{SSIM}(\mathcal{P}^{(m)}(\hat{\mathcal{C}}),\mathcal{P}^{(m)}(\mathcal{C}))$
然后$L_{\mathrm{S3IM}}(\Theta,\mathcal{R})=1-\mathrm{S3IM}(\hat{\mathcal{R}},\mathcal{R}) = =1-\frac1M\sum_{m=1}^M\mathrm{SSIM}(\mathcal{P}^{(m)}(\hat{\mathcal{C}}),\mathcal{P}^{(m)}(\mathcal{C}))$

## Volume Rendering

### [ReTR: Modeling Rendering Via Transformer for Generalizable Neural Surface Reconstruction](https://arxiv.org/pdf/2305.18832)

**使用Transformer 代替渲染过程，并且添加了深度监督**

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20231109094904.png)


## Mixture of Experts (MoE)

### [Boost Your NeRF: A Model-Agnostic Mixture of Experts Framework for High Quality and Efficient Rendering](https://arxiv.org/pdf/2407.10389#page=20.42)

> [混合专家模型（MoE）详解](https://huggingface.co/blog/zh/moe#%E4%BB%80%E4%B9%88%E6%98%AF%E6%B7%B7%E5%90%88%E4%B8%93%E5%AE%B6%E6%A8%A1%E5%9E%8B) MoE 层由两个核心部分组成: 一个门控网络(用于决定哪些令牌 (token) 被发送到哪个专家)和若干数量的专家(每个专家本身是一个独立的神经网络)

- 门控网络来决定采样点被输入哪 Top-K个专家网络(并且在filtering step抛弃low-density的点，密度值根据lowest resolution model 进行计算)
- 每个专家网络以采样点位置和方向作为输入，输出颜色和密度
- 根据该点到每个专家网络的权重(概率Probability Field)和颜色密度值，计算该点最终的颜色和密度值。最后通过体渲染得到pixel color，并联合优化resolution-weighted auxiliary loss(用于选取专家网络)

主要思想是在训练了一批不同分辨率的NeRF models后，优先使用low-resolution models，减少high-resolution models 的使用

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240806202919.png)


# Efficiency Improvement

- 完全抛弃MLP，存储显示的颜色/密度
- 加速渲染：对几何代理进行光栅化处理
- 加速渲染：使用前一个视图的信息来减少渲染像素的数量

## Explicit Grids with features(Efficiency of T&R)

**(减轻MLP架构)**

| Explicit Grids     | Related Works                                                                                                                                  |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| Feature grids      | [Plenoxels](https://arxiv.org/pdf/2112.05131), [DirectVoxGo](https://arxiv.org/pdf/2111.11215), [InstantNGP](https://arxiv.org/pdf/2201.05989) |
| Tri-planes         | [Tri-MipRF](https://arxiv.org/pdf/2307.11335)                                                                                                  |
| Multi-plane images | [MMPI](https://arxiv.org/pdf/2310.00249), [fMPI](https://arxiv.org/pdf/2312.16109)                                                             |
| Tensorial vectors  | [TensoRF](https://arxiv.org/pdf/2203.09517), [Strivec](https://arxiv.org/pdf/2307.13226)                                                       |

### [NGP-RT: Fusing Multi-Level Hash Features with Lightweight Attention for Real-Time Novel View Synthesis](https://arxiv.org/pdf/2407.10482)

InstantNGP 虽然训练速度(查询网格)很快，但是渲染的时候仍然需要大量的3D Point查询MLP，耗费大量时间

(*Deferred neural rendering*) SNeRG [12] 和 MERF [32] 通过将颜色和密度存在显示网格中，只对每条投射光线执行一次 MLP，从而大大加快了渲染过程。然而，他们对显式特征的处理显示出有限的表达能力，不适合 Instant-NGP 的多级特征。**将这些特征构建方法直接应用于 Instant-NGP 中的多级特征可能会导致其表示能力受损，并导致渲染质量下降**。

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240718142648.png)

提出了 NGP-RT，一种利用轻量级注意力机制高效渲染高保真新视图的新方法。NGP-RT 的注意力机制采用了简单而有效的加权和运算，可学习的注意力参数可自适应地优先处理显式多级哈希特征

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240718142838.png)


## Multi input and Multi output MLP(Efficiency of T&R)

**(减少MLP计算次数)**

### [MIMO-NeRF: Fast Neural Rendering with Multi-input Multi-output Neural Radiance Fields](https://arxiv.org/pdf/2310.01821)


## Points Sampling(Efficiency of T&R)

**(减少采样点数量 per ray)**

## Pixel Sampling(Efficiency of Training)

**(减少MLP计算次数)**

之前方法对train_data中所有的像素rgb三个值，进行预测+l1 loss+反向传播，训练速度很慢

通过量化rendering image 与 g.t. image 之间的差异，来指导在图像上采样像素的位置/数量：根据error map between rendering image and g.t. image，消除loss比较小的区域，对loss大的区域进行更多的采样

**idea**：
- Other sampling method 如果要加速训练的话，就要使用更高效的采样方法
  - LHS (Latin hypercube sampling)
- 这种方法只能对Train过程进行加速
- 由于优化的是MLP整体的参数，因此可能出现对误差大的像素优化时，降低误差小的像素的预测精度。
- 可能会对error大但是非重要区域的像素进行多次采样

> [马尔可夫链蒙特卡罗算法（MCMC） - 知乎](https://zhuanlan.zhihu.com/p/37121528) Basic
> [详解Markov Chain Monte Carlo (MCMC): 从拒绝-接受采样到Gibbs Sampling | Lemon's Blog](https://coderlemon17.github.io/posts/2022/05-11-mcmc/) **更清楚一点**
> [MCMC详解2——MCMC采样、M-H采样、Gibbs采样（附代码）](https://blog.csdn.net/u012290039/article/details/105696097)
> [采样（三）：重要性采样与接受拒绝采样 | Erwin Feng Blog](https://allenwind.github.io/blog/10466/)
> [The Markov-chain Monte Carlo Interactive Gallery](https://chi-feng.github.io/mcmc-demo/) **多种MCMC采样方法**

Monte Carlo采样无法得到复杂的分布(二维分布)，加入Markov Chain，马尔科夫链模型的状态转移矩阵收敛到的稳定概率分布与我们的初始状态概率分布无关$\pi(j)=\sum_{i=0}^\infty\pi(i)P_{ij}$(只与状态转移矩阵有关)，如果可以得到状态转移矩阵，就可以采样得到平稳分布的样本集。如何得到状态转移矩阵？
--> MCMC方法(与拒绝-接受采样的思路类似，其通过拒绝-接受概率拟合一个复杂分布, MCMC方法则通过拒绝-接受概率得到一个满足细致平稳条件的转移矩阵.)
- Metropolis-Hastings Sampling：需要计算接受率, 在高维时计算量大, 并且由于接受率的原因导致算法收敛时间变长. 对于高维数据, 往往数据的条件概率分布易得, 而联合概率分布不易得.
- Gibbs Sampling：



### [Accelerating Neural Field Training via Soft Mining](https://arxiv.org/pdf/2312.00075)

> [EGRA-NeRF: Edge-Guided Ray Allocation for Neural Radiance Fields](https://www.sciencedirect.com/science/article/pii/S0262885623000446) NeRF 的渲染显得过于模糊，并且在某些纹理或边缘中包含锯齿伪影，为此提出了边缘引导光线分配（EGRA-NeRF）模块，以**在训练阶段将更多光线集中在场景的纹理和边缘上** **(没有加速)**

To implement our idea we use Langevin Monte-Carlo sampling. We show that by doing so, regions with higher error are being selected more frequently, leading to more than 2x improvement in convergence speed.
$\mathcal{L}=\frac{1}{N}\sum_{n=1}^{N}\mathrm{err}(\mathbf{x}_{n})\approx\mathbb{E}_{\mathbf{x}\sim P(\mathbf{x})}\left[\mathrm{err}(\mathbf{x})\right]=\int\mathrm{err}(\mathbf{x})P(\mathbf{x})d\mathbf{x}$. P is the distribution of the sampled data points $x_{n}$. 之前方法大多是均匀分布
具体做法：
1. Soft mining with **importance sampling**. 引入了 importance distribution$Q(x)$

> 补充知识 [[蒙特卡洛方法] 02 重要性采样（importance sampling）及 python 实现_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1SV4y1i7bW/?spm_id_from=333.788&vd_source=1dba7493016a36a32b27a14ed2891088) 可以从一个任意分布中进行采样

误差可以重写为：$\int\operatorname{err}(\mathbf{x})P(\mathbf{x})d\mathbf{x} =\int\frac{\mathrm{err}(\mathbf{x})P(\mathbf{x})}{Q(\mathbf{x})}Q(\mathbf{x})d\mathbf{x}  =\mathbb{E}_{\mathbf{x}\sim Q(\mathbf{x})}\left[\frac{\mathrm{err}(\mathbf{x})P(\mathbf{x})}{Q(\mathbf{x})}\right].$ 由于均匀分布$P(x)$的PDF通常为常数，因此$\mathcal{L}=\frac{1}{N}\sum_{n=1}^{N}\frac{\mathrm{err}(\mathbf{x}_{n})}{Q(\mathbf{x}_{n})}$ $\mathrm{where} \quad\mathbf{x}_{n}\sim Q(\mathbf{x}).$
但是$\mathcal{L}$无法用于训练，采用[stop gradient operator](https://arxiv.org/pdf/1711.00937)(在正向计算时定义为同一值，且偏导数为零)
$\mathcal{L}\approx\mathbb{E}_{\mathbf{x}\sim\mathrm{sg}(Q(\mathbf{x}))}\left[\frac{\mathrm{err}(\mathbf{x})}{\mathrm{sg}(Q(\mathbf{x}))}\right].$
选择的$Q(x)$必须与$err(x)$成比例关系(消除error尺度的影响)：$\mathrm{err}(\mathbf{x})=\|f_{\boldsymbol{\psi}}(\mathbf{x})-f_{\mathrm{gt}}(\mathbf{x})\|_{2}^{2},\\Q(\mathbf{x})=\|f_{\boldsymbol{\psi}}(\mathbf{x})-f_{\mathrm{gt}}(\mathbf{x})\|_{1}.$
**根据**$Q(x)$的定义，会在error大的地方多采样一些$x$即像素点
Soft mining. $\mathcal{L}=\frac1N\sum_{n=1}^N\left[\frac{\mathrm{err}(\mathbf{x}_n)}{\mathrm{sg}(Q(\mathbf{x}_n))^\alpha}\right],\quad\text{where }\alpha\in[0,1]$(在关注error大的区域的同时也要关注一下其他区域，不然可能会学歪)
- $\alpha = 0$：hard mining
- $\alpha = 1$：(pure) importance sampling

2. Sampling via **Langevin Monte Carlo**

> [The promises and pitfalls of Stochastic Gradient Langevin Dynamics](https://arxiv.org/pdf/1811.10072) 数学分析LMC, SGLD, SGLDFP and SGD
> [MCMC using Hamiltonian dynamics](https://arxiv.org/pdf/1206.1901) 

从任意分布$Q(x)$中采样，使用MCMC方法中的Langevin Monte Carlo (LMC)
$\mathbf{x}_{t+1}=\mathbf{x}_t+a\nabla\log Q(\mathbf{x}_t)+b\boldsymbol{\eta}_{t+1},$
- a>0 is a hyperparameter defining the step size for the gradient-based walks
- b>0 is a hyperparameter defining the step size for the random walk $\boldsymbol{\eta}_{t\boldsymbol{+}1}\boldsymbol{\sim}\mathcal{N}(0,\mathbf{1})$
- 采样是局部的，因此采样的开销很小
- log的作用应该是把乘除转换成加减, eg: $w_i=\frac{p(x_i)}{q(x_i)}, \log w_i=\log p(x_i)-\log q(x_i)$

Sample (re-)initialization.(采样的初始化很重要)：We first initialize the sampling distribution to be uniform over the domain of interest as $\mathbf{x}_{0}{\sim}\mathcal{U}(\mathcal{R})$. We further re-initialize samples that either move out of $\mathcal{R}$ or have too low error value causing samples to get ‘stuck’. We use uniform sampling as well as edge-based sampling for 2D workloads.
Warming up soft mining. Start with $\alpha=0$, i.e., no correction, then linearly increase it to the desired $\alpha$ value at 1k iterations.
Alternative: multinomial sampling. To use multinomial sampling, one needs to do a forward pass of all data points to build a probability density function, which is computationally expensive. Hence an alternative strategy, such as those based on Markov Chain Monte Carlo (MCMC) is required. 为了防止对所有像素点进行前向计算以计算PDF的高耗费，使用MCMC采样

Ablation studies中: 即使LMC采样的精度比multinomial sampling低一点，但仍然比Uniform sampling更高，且更effective(效率与精度的折中(compromise/trade-off))

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240708102314.png)


### [Fast Learning Radiance Fields by Shooting Much Fewer Rays](https://arxiv.org/pdf/2208.06821)

![image.png|888](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240620160134.png)
之前的从图片中采样像素or光线的方法：$\mathbf{r}_i(u,v)\sim U(I_i),u\in[0,H_i],v\in[0,W_i],$ 按均匀分布随机采样
- trivial areas：对于背景(大部分颜色相同)，这样图片的像素就是非均匀的分布，均匀采样方式会导致采样到一些无意义的像素。 **Therefore**, we only need to shoot fewer rays in the trivial areas where the color changes slightly to perceive the radiance fields
- nontrivial areas：color changes greatly contain more information, so more rays are required to capture the detailed information and learn how to distinguish these pixels’ colors from its neighboring ones

trivial areas会很快收敛，而nontrivial areas不容易收敛
Based on the above observation, we propose two strategies to optimize ray distribution on input images. 
- The first one is to calculate a **prior probability distribution based on the image context** GT图片的先验分布
- the second one is to apply an **adaptive quadtree subdivision algorithm** to dynamically adjust ray distribution. 自适应的QSA

具体做法：
1. Context based Probability Distribution：we use the color variation of pixels relative to their surroundings to quantitatively identify the image context.

计算每个像素点跟周围八个点的std：$g(u,v)=\operatorname{std}(\mathbf{c}(u,v))=\sqrt{\frac19\sum_{x,y}[\mathbf{c}(x,y)-\overline{\mathbf{c}}]^2},\\x\in\{u-1,u,u+1\},y\in\{v-1,v,v+1\}.$ g越高表示像素颜色/密度变化越剧烈，通常在3D物体的表面边界处。优势：our image context based probability distribution function naturally helps to estimate where surfaces are located.
为了平衡$g_{max}$与$g_{min}$之间的差异，clamp后进行归一化：$g^{\prime}(u,v)=\frac{\mathrm{clamp}(s,\max(g(u,v)))}{\mathrm{max}(g(u,v))}$，we typically define threshold $\begin{aligned}s=0.01\times\text{mean}(g(u,v))\end{aligned}$. Values less than s will be clamped to s to avoid sampling too few rays at the corresponding positions.
**Sampling strategy**. In the lines of “Sampled Rays Distribution”, we sample 50% rays according to the context based probability distribution and randomly sample the other 50% rays, where each red point represents a sampled ray(**为什么要设置成50%**)
2. Adaptive QuadTree Subdivision：对于rendering error，只在error大的地方细分，在error小的地方不在细分(根据pre-defined threshold a来判断大小)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240620170649.png)
图片$I_{i}(H_{i} \times W_{i})$上总的采样数量：$M_i^l=Q_1^l\times\frac{H_i}{2^l}\times\frac{W_i}{2^l}+Q_2^l\times n_0,$ $l$ denote the times of subdivision, $Q_1^l$ and $Q_2^l$ denote the number of unmarked leaf nodes (error>a) and marked leaf nodes (error<a) separately. $n_{0} = 10(constant)$
> [iMAP: Implicit Mapping and Positioning in Real-Time](https://arxiv.org/pdf/2103.12352)

同时[iMap](https://arxiv.org/pdf/2103.12352)也使用rendering error来引导采样，不同点：
- The applications of rendering loss are different. iMap uses the rendering loss distribution on image blocks to decide how many points should be sampled on each block, while we use the **rendering loss** on each leaf node to decide whether this node should be subdivided into 4 child nodes.
- The number of sampled points and image blocks are different. In our method, the number of sampled points in each leaf node is identical, but the number and area of the image blocks (i.e. leaf nodes) changes during training. In contrast, iMap samples different numbers of rays according to a render loss based distribution in each one of the same size blocks.
- The sampling strategies in image blocks are different. In each image block, iMap uniformly samples points for rendering, while we sample points according to the image context. More points are sampled in the nontrivial areas where color changes a lot, while fewer points are sampled in the trivial areas where color changes slightly. Our sampling strategy helps to capture the detailed information in the nontrivial areas and reduce the training burden in the trivial areas.

3. Implementation Details
- 在每个epoch结束后，也就是对数据集中所有图像的像素都进行一次rendering，然后与g.t.进行对比得到rendering error
- In practice, we initially subdivide the quadtrees into 2 or 3 depths at the begin of training. This helps our method to distinguish the trivial and nontrivial areas faster among the quadtree leaf nodes

存在的问题：
- All-Pixel Sampling中，作者为了防止MLP对unmarked leaf nodes进行拟合的同时会改变已经拟合好的marked leaf nodes，在接近最后的epoch，使用了randomly sample rays from the whole image instead of using quadtrees for sampling, where the number of sampled rays is equal to the total number of pixels. **如何freeze已经拟合好的marked leaf nodes???**

### [iNeRF: Inverting Neural Radiance Fields for Pose Estimation](https://arxiv.org/pdf/2012.05877)

一种基于NeRF预训练模型的姿态估计的方法. 有了NeRF(MLP), 去refine位姿
- Sampling Rays: 计算所有pixel是非常耗时耗力的。目的是想要采样的点可以更好的包含在observed images和rendered images上。三种策略：
  - Random Sampling
  - Interest Point Sampling: employ interest point detectors to localize a set of candidate pixel locations in the observed image, this strategy makes optimization converge faster since less stochasticity is introduced. **But** we found that it is prone to local minima as it only considers interest points on the observed image instead of interest points from both the observed and rendered images.
  - Interest Region Sampling: After the interest point detector localizes the interest points, we apply a 5 × 5 **morphological dilation** for I iterations to enlarge the sampled region.

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240620151248.png)



