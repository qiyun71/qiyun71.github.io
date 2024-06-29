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

<!-- more -->

# Basic

> [NeuS: Learning Neural Implicit Surfaces by Volume Rendering for Multi-view Reconstruction](https://arxiv.org/pdf/2106.10689)

SDF内部-1，外部1，表面0
NeuS没有与NeRF一样直接使用MLP输出的不透明度$\sigma$作为$\rho$，而是使用预测的sdf进行相应计算得到$\rho$，以及权重
- $C(\mathbf{o},\mathbf{v})=\int_{0}^{+\infty}w(t)c(\mathbf{p}(t),\mathbf{v})\mathrm{d}t$ 
- $\rho(t)=\max\left(\frac{-\frac{\mathrm{d}\Phi_s}{\mathrm{d}t}(f(\mathbf{p}(t)))}{\Phi_s(f(\mathbf{p}(t)))},0\right)$ , MLP预测的sdf即$f(\mathbf{p}(t))$
- $\phi_s(x) =\frac{se^{-sx}}{(1+e^{-sx})^{2}}$, $\Phi_s(x)=(1+e^{-sx})^{-1},\text{i.e.,}\phi_s(x)=\Phi_s'(x)$

除了$\mathcal{L}1$和$\mathcal{L}_{mask}$损失之外还使用了$\mathcal{L}_{r e g}=\frac{1}{n m}\sum_{k,i}(\|\nabla f(\hat{\mathbf{p}}_{k,i})\|_{2}-1)^{2}.$ (Eikonal term)

# Paper

开题报告
- 三维重建算法的模块化框架构建(已有)
- 提升重建速度的像素选取方法研究
  - *目前方法*：对train_data中所有的像素rgb三个值，进行预测+l1 loss+反向传播
  - **改进：**
    - error map，对比render image与g.t. image(L1)，对误差大的像素进行优先优化。
      - 但是由于优化的是MLP整体的参数，因此可能出现对误差大的像素优化时，降低误差小的像素的预测精度。
      - 可能会对error map大但是非重要区域的像素进行多次采样
- 提高重建精度的混合编码方式研究(已有)
- 三维重建数据采集平台的搭建(设备)

## Camera Pose Estimation(Precision)

## Point Sampling(Precision)

### 混合采样方式
NerfAcc：占据+逆变换采样
先使用占据网格确定哪些区域需要采样，再通过粗采样得到的权重使用逆变换采样进行精采样得到采样点

## Pixel Sampling(Efficiency)
通过量化rendering image 与 g.t. image 之间的差异，来指导在图像上采样像素的位置/数量
**idea**：
- LHS (Latin hypercube sampling)
- 根据loss between rendering image and g.t. image，消除loss比较小的区域，对loss大的区域进行更多的采样

### [iNeRF: Inverting Neural Radiance Fields for Pose Estimation](https://arxiv.org/pdf/2012.05877)
一种基于NeRF预训练模型的位姿估计的方法. 有了NeRF(MLP), 去refine位姿
- Sampling Rays: 计算所有pixel是非常耗时耗力的。目的是想要采样的点可以更好的包含在observed images和rendered images上。三种策略：
  - Random Sampling
  - Interest Point Sampling: employ interest point detectors to localize a set of candidate pixel locations in the observed image, this strategy makes optimization converge faster since less stochasticity is introduced. **But** we found that it is prone to local minima as it only considers interest points on the observed image instead of interest points from both the observed and rendered images.
  - Interest Region Sampling: After the interest point detector localizes the interest points, we apply a 5 × 5 **morphological dilation** for I iterations to enlarge the sampled region.

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240620151248.png)
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

### [Accelerating Neural Field Training via Soft Mining](https://arxiv.org/pdf/2312.00075)
> [EGRA-NeRF: Edge-Guided Ray Allocation for Neural Radiance Fields](https://www.sciencedirect.com/science/article/pii/S0262885623000446)

To implement our idea we use Langevin Monte-Carlo sampling. We show that by doing so, regions with higher error are being selected more frequently, leading to more than 2x improvement in convergence speed.
$\mathcal{L}=\frac{1}{N}\sum_{n=1}^{N}\mathrm{err}(\mathbf{x}_{n})\approx\mathbb{E}_{\mathbf{x}\sim P(\mathbf{x})}\left[\mathrm{err}(\mathbf{x})\right]=\int\mathrm{err}(\mathbf{x})P(\mathbf{x})d\mathbf{x}$. P is the distribution of the sampled data points $x_{n}$. 之前方法大多是均匀分布
具体做法：
1. Soft mining with **importance sampling**. 引入了 importance distribution$Q(x)$

> 补充知识 [[蒙特卡洛方法] 02 重要性采样（importance sampling）及 python 实现_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1SV4y1i7bW/?spm_id_from=333.788&vd_source=1dba7493016a36a32b27a14ed2891088)

误差可以重写为：$\int\operatorname{err}(\mathbf{x})P(\mathbf{x})d\mathbf{x} =\int\frac{\mathrm{err}(\mathbf{x})P(\mathbf{x})}{Q(\mathbf{x})}Q(\mathbf{x})d\mathbf{x}  =\mathbb{E}_{\mathbf{x}\sim Q(\mathbf{x})}\left[\frac{\mathrm{err}(\mathbf{x})P(\mathbf{x})}{Q(\mathbf{x})}\right].$ 由于均匀分布$P(x)$的PDF通常为常数，因此$\mathcal{L}=\frac{1}{N}\sum_{n=1}^{N}\frac{\mathrm{err}(\mathbf{x}_{n})}{Q(\mathbf{x}_{n})}$ $\mathrm{where} \quad\mathbf{x}_{n}\sim Q(\mathbf{x}).$
但是$\mathcal{L}$无法用于训练，采用[stop gradient operator](https://arxiv.org/pdf/1711.00937)(在正向计算时定义为同一值，且偏导数为零)
$\mathcal{L}\approx\mathbb{E}_{\mathbf{x}\sim\mathrm{sg}(Q(\mathbf{x}))}\left[\frac{\mathrm{err}(\mathbf{x})}{\mathrm{sg}(Q(\mathbf{x}))}\right].$
选择的$Q(x)$必须与$err(x)$成比例关系(消除error尺度的影响)：$\mathrm{err}(\mathbf{x})=\|f_{\boldsymbol{\psi}}(\mathbf{x})-f_{\mathrm{gt}}(\mathbf{x})\|_{2}^{2},\\Q(\mathbf{x})=\|f_{\boldsymbol{\psi}}(\mathbf{x})-f_{\mathrm{gt}}(\mathbf{x})\|_{1}.$
**根据**$Q(S)$的定义，会在error大的地方多采样一些$x$即像素点
Soft mining. $\mathcal{L}=\frac1N\sum_{n=1}^N\left[\frac{\mathrm{err}(\mathbf{x}_n)}{\mathrm{sg}(Q(\mathbf{x}_n))^\alpha}\right],\quad\text{where }\alpha\in[0,1]$(在关注error大的区域的同时也要关注一下其他区域，不然可能会学歪)
- $\alpha = 0$：hard mining
- $\alpha = 1$：(pure) importance sampling

2. Sampling via **Langevin Monte Carlo**

从任意分布$Q(x)$中采样，使用MCMC方法中的Langevin Monte Carlo (LMC)
$\mathbf{x}_{t+1}=\mathbf{x}_t+a\nabla\log Q(\mathbf{x}_t)+b\boldsymbol{\eta}_{t+1},$
- a>0 is a hyperparameter defining the step size for the gradient-based walks
- b>0 is a hyperparameter defining the step size for the random walk $\boldsymbol{\eta}_{t\boldsymbol{+}1}\boldsymbol{\sim}\mathcal{N}(0,\mathbf{1})$
- 采样是局部的，因此采样的开销很小

Sample (re-)initialization.(采样的初始化很重要)：We first initialize the sampling distribution to be uniform over the domain of interest as $\mathbf{x}_{0}{\sim}\mathcal{U}(\mathcal{R})$. We further re-initialize samples that either move out of $\mathcal{R}$ or have too low error value causing samples to get ‘stuck’. We use uniform sampling as well as edge-based sampling for 2D workloads.
Warming up soft mining. Start with $\alpha=0$, i.e., no correction, then linearly increase it to the desired $\alpha$ value at 1k iterations.
Alternative: multinomial sampling. To use multinomial sampling, one needs to do a forward pass of all data points to build a probability density function, which is computationally expensive. Hence an alternative strategy, such as those based on Markov Chain Monte Carlo (MCMC) is required.
