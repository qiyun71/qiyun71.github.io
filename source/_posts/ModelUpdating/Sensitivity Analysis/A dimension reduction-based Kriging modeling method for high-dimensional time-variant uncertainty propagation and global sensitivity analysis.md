---
zotero-key: I2A9MG56
zt-attachments:
  - "2754"
title: A dimension reduction-based Kriging modeling method for high-dimensional time-variant uncertainty propagation and global sensitivity analysis
created: 2025-06-12 02:51:33
modified: 2025-08-11 10:54:28
tags:
  - /done
  - ⭐⭐⭐
collections: GSA+AK+DR+SIS  Dimensional reduction
year: 2024
publication: Mechanical Systems and Signal Processing
citekey: songDimensionReductionbasedKriging2024
---

| Title        | "A dimension reduction-based Kriging modeling method for high-dimensional time-variant uncertainty propagation and global sensitivity analysis"                                                                                                                                                                                                                                                                                                                         |
| ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Author       | [Zhouzhou Song,Hanyu Zhang,Qiangqiang Zhai,Boqin Zhang,Zhao Liu,Ping Zhu]                                                                                                                                                                                                                                                                                                                                                                                               |
| Organization | a State Key Laboratory of Mechanical System and Vibration, **Shanghai Jiao Tong University**, Shanghai 200240, PR China  b National Engineering Research Center of Automotive Power and Intelligent Control, Shanghai Jiao Tong University, Shanghai 200240, PR China  c School of Design, Shanghai Jiao Tong University, Shanghai 200240, PR China  d Department of Industrial Systems Engineering and Management, National University of Singapore, 119077, Singapore |
| Paper        | [Zotero pdf](zotero://select/library/items/I2A9MG56) [attachment](<file:///D:/Download/Zotero_data/storage/2TRZV9MT/Song%20%E7%AD%89%20-%202024%20-%20A%20dimension%20reduction-based%20Kriging%20modeling%20method%20for%20high-dimensional%20time-variant%20uncertainty%20pr.pdf>)<br><br>                                                                                                                                                                            |
| Project      |                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |

<!-- more -->

## Background

将高维数据降维（构建精确代理模型需要的样本数量随着维度增加呈指数增加）

Q:为什么Kriging可以解决小样本问题？
A:Kriging与测试根据输入样本的插值进行计算

之前的方法只考虑了高维输入(TimeSeries-based 回归、预测、分类问题)，而没有考虑高维输出问题

## Innovation

- 不仅考虑了高维输入，还考虑了高维输出(SVD-based PCA)
  - “Although several dimension reduction-based Kriging methods have been proposed for scalar outputs, methods for highdimensional outputs have not yet been fully investigated.” ([Song 等, 2024, p. 2](zotero://select/library/items/I2A9MG56)) ([pdf](zotero://open-pdf/library/items/2TRZV9MT?page=2&annotation=ZZ44HFWK))
- 主成分数量的确定方法 ladle estimator，结合了特征值和特征向量进行秩估计(将主成分确定问题转换成秩确定问题)
- 将输入的 project matrix(ISDR) 嵌入到 Kriging model(Kernel function) 中
- 广义的敏感性指标，可以评估输入参数对时变响应整体的敏感性程度

## Outlook

- PCA是一种线性的降维方法，无法处理强非线性问题(latent 输入和输出的数量非常多，计算量大)

## Cases

- “A mathematical problem” ([Song 等, 2024, p. 13](zotero://select/library/items/I2A9MG56)) ([pdf](zotero://open-pdf/library/items/2TRZV9MT?page=13&annotation=99YR5DQ4))
- “A truss bridge problem” ([Song 等, 2024, p. 14](zotero://select/library/items/I2A9MG56)) ([pdf](zotero://open-pdf/library/items/2TRZV9MT?page=14&annotation=HNVQHJYN))
- “A heat conduction problem” ([Song 等, 2024, p. 17](zotero://select/library/items/I2A9MG56)) ([pdf](zotero://open-pdf/library/items/2TRZV9MT?page=17&annotation=VL4TEDBJ))
- “An aluminum-CFRP hybrid joint structure problem” ([Song 等, 2024, p. 18](zotero://select/library/items/I2A9MG56)) ([pdf](zotero://open-pdf/library/items/2TRZV9MT?page=18&annotation=YZ5GXN3V))

## Equation

Data：$\mathcal{D}=\{(\mathbf{x}_{i},\mathbf{y}_{i}),i=1,2,\dots N\}$ total N samples
- Input：$\mathbf{x} = [x_1,\dots ,x_p]^{\mathrm{T}}\in \mathbb{R}^p$
- Output: $\mathbf{y} = [y_1,\dots ,y_{N_{t}}]^{\mathrm{T}}\in \mathbb{R}^{N_{t}}$

$\mathbf{Y}=\left.\left[\begin{array}{cccc}\mathbf{y}_1(t_1)&\mathbf{y}_1(t_2)&\cdots&\mathbf{y}_1(t_{N_t})\\\mathbf{y}_2(t_1)&\mathbf{y}_2(t_2)&\cdots&\mathbf{y}_2(t_{N_t})\\\vdots&\vdots&\ddots&\vdots\\\mathbf{y}_N(t_1)&\mathbf{y}_N(t_2)&\cdots&\mathbf{y}_N(t_{N_t})\end{array}\right.\right]$

### SVD-based PCA

> [【转载】奇异值分解(SVD)计算过程示例 - marsggbo - 博客园](https://www.cnblogs.com/marsggbo/p/10155801.html)
> [(10 封私信) 主成分分析（PCA）原理详解 - 知乎](https://zhuanlan.zhihu.com/p/37777074)

高维输出响应
$$ \mathbf{y} = [\mathbf{y}_1,\mathbf{y}_2,\dots ,\mathbf{y}_N]^{\mathrm{T}} = \left[ \begin{array}{cccc}\mathbf{y}_1(t_1) & \mathbf{y}_1(t_2) & \dots & \mathbf{y}_1(t_{N_t})\\ \mathbf{y}_2(t_1) & \mathbf{y}_2(t_2) & \dots & \mathbf{y}_2(t_{N_t})\\ \vdots & \vdots & \ddots & \vdots \\ \mathbf{y}_N(t_1) & \mathbf{y}_N(t_2) & \dots & \mathbf{y}_N(t_{N_t}) \end{array} \right]$$

对输出的协方差矩阵 $\begin{array}{r}\mathbf{C} = \frac{1}{N - 1}\Big(\mathbf{y} - \boldsymbol {\mu}_{\mathbf{y}}\Big)^{\mathrm{T}}\Big(\mathbf{y} - \boldsymbol {\mu}_{\mathbf{y}}\Big) \end{array}$ 执行SVD，即PCA，得到 $\mathbf{C} = \mathbf{U}\boldsymbol {\Lambda}\mathbf{V}^{\mathrm{T}}$。假设前 $N_{\mathrm{LD}}$ 主方向可以捕捉到 $\mathbf{y}$ 的变化，则低维潜在空间可以表示为：
$$ \mathbf{y}_{\mathrm{LD}} = \mathbf{y}\mathbf{U}_{\mathrm{LD}}$$
则 $\mathbf{y}$ 可以从原来的 $N$ 降维到 $N_{LD}$，并且低维潜在空间可以通过 $\mathbf{U}_{\mathrm{LD}}$ 还原：

$$\mathbf{y} = \mathbf{y}_{\mathrm{LD}}\mathbf{U}_{\mathrm{LD}}^{\mathrm{T}}$$

将主成分数量确定问题，转换为协方差矩阵的秩估计问题，矩阵的秩信息隐藏在特征值和特征向量中，“Therefore, to estimate the rank of a matrix, we must consider both the eigenvalues and eigenvectors, and not just the eigenvalues” ([Song 等, 2024, p. 7](zotero://select/library/items/I2A9MG56)) ([pdf](zotero://open-pdf/library/items/2TRZV9MT?page=7&annotation=KICAIV29))

“The ladle estimator” ([Song 等, 2024, p. 7](zotero://select/library/items/I2A9MG56)) ([pdf](zotero://open-pdf/library/items/2TRZV9MT?page=7&annotation=D3Q7TE7M))

Let $\eta_{1},\eta_{2},\dots ,\eta_{N_{t}}$ be the eigenvectors corresponding to $\lambda_{1},\lambda_{2},\dots ,\lambda_{N_{t}}$ and let $\mathbf{B}_k = (\eta_1,\eta_2,\dots ,\eta_k)$ be the $N_{t}\times k$ matrix consisting the principal $k$ eigenvectors of C. 

To quantify the variability of eigenvectors, we can draw r **bootstrap samples(重新放回的采样)** of size $N$ from the response data in the training set $\mathcal{D}$ to form $r$ output matrices $\mathbf{y}_i^* (1\leq i\leq r)$ . For each $\mathbf{y}_i^*$ , we can obtain the corresponding sample covariance matrix $\mathbf{C}_i^*$ and the corresponding eigenvector matrix $\mathbf{B}_{k,i}^*$ which consists the principal $k$ eigenvectors of $\mathbf{C}_i^*$ . The bootstrap variability of $\mathbf{B}_k$ is then defined by the following function:

$$ f_{r}^{0}(k) = \left\{ \begin{array}{c}0,k = 0,\\ r^{-1}\sum_{i = 1}^{r}\Big\{1 - \Big|\operatorname *{det}\big(\mathbf{B}_{k}^{\mathrm{T}}\mathbf{B}_{k,i}^{*}\big)\Big|\Big\} ,k = 1,\dots N_{t} - 1. \end{array} \right. $$
$f_{r}^{0}(k)$ reaches 0 if each $\mathbf{B}_{k,i}^{*}$ spans the same column space as $\mathbf{B}_k$ and 1 if each $\mathbf{B}_{k,i}^{*}$ spans a space orthogonal to $\widehat{\mathbf{B}}_k$ . In other words, the greater the discrepancy between the column spaces of $\mathbf{B}_k$ and $\mathbf{B}_{k,i}^{*}$ the larger the value of $f_{r}^{0}(k)$ .

- 当 $k<N_{LD}$ 时，所有特征值 $\lambda_{k}$ 是分明的，特征向量之间的差异很小，因此 $f_{r}^{0}(k)$ 很小
- 当 $k>N_{LD}$ 时，特征值接近 0，因此空间中的特征向量任意，导致 $f_{r}^{0}(k)$ 很大

但是，上述方法仅考虑了特征向量来获得秩，这种方法是启发式的，并不严格。在某些情况下，特征向量的特征值或变异性彼此接近，并且没有明显的差距，从而导致估计错误。

为此 the ladle estimator 被提出融合 $\lambda_{k}$ and $f_{r}^{0}(k)$ 进行秩估计. 

To do this, $f_{r}^{0}(k)$ first needs to be normalized as follows:

$$ f_{r}(k) = \frac{f_{r}^{0}(k)}{1 + \sum_{i = 0}^{N_{t} - 1}f_{r}^{0}(i)},k = 0,1,\dots ,N_{t} - 1$$
特征值：
$$ \phi_{r}(k) = \frac{\lambda_{k + 1}}{1 + \sum_{i = 0}^{N_{t} - 1}\lambda_{i + 1}},k = 0,1,\dots ,N_{t} - 1$$
Then the ladle estimator for estimating the matrix rank $N_{\mathrm{LD}}$ is defined as follows:
$$ \begin{array}{r}\widehat{N}_{\mathrm{LD}} = \underset {k = 0,1,\dots ,N_t - 1}{\operatorname{argmin}}g_r(k) = \underset {k = 0,1,\dots ,N_t - 1}{\operatorname{argmin}}(f_r(k) + \phi_r(k)). \end{array} \tag{26} $$

Note that the eigenvalues are unambiguous when $k< N_{\mathrm{LD}}$ and the bootstrapped eigenvector variations are unambiguous when $k > N_{\mathrm{LD}}$ . Thus, $g_{r}(k)$ is unambiguous for $k = 0,1,\dots ,N_{t} - 1$ and its minimum value is easy to identify.


### ISDR

“Sufficient dimension reduction and sliced inverse regression” ([Song 等, 2024, p. 4](zotero://select/library/items/AECRTI4F)) ([pdf](zotero://open-pdf/library/items/BDXU2TUW?page=4&annotation=3Z246K3Z))

输入和输出：$\mathbf{X} \in\mathbb{R}^{p},\mathbf{Y} \in \mathbb{R}$，经典的线性 SDR 方法目标是寻找一个 $p\times d$ 的矩阵 $\mathbf{W}$，使其满足：$Y\coprod\mathbf{X}|\mathbf{W}^\mathrm{T}\mathbf{X}$。
- 其中 $\coprod$ 是指随机变量的独立性，$|$ 指条件。
- 含义：给定条件下，$Y$ 与 $\mathbf{X}$ 是相互独立的。即在已知 $\mathbf{W}^\mathrm{T}\mathbf{X}$ 后，$\mathbf{X}$ 不在提供新的信息

$Y\coprod\mathbf{X}|\mathbf{W}^\mathrm{T}\mathbf{X}$ 被定义为 the central subspace $\mathscr{S}_{\widetilde{Y}|\mathbf{X}}$。With the assumption of linearity condition：
$$
\Sigma_{\mathbf{X}\mathbf{X}}^{-1}\Big\{\mathbb{E}\Big[\mathbf{X}\Big|\widetilde{Y}\Big]-\mathbb{E}[\mathbf{X}]\Big\} \in \mathscr{S}_{\widetilde{Y}|\mathbf{X}},$$

where $\Sigma_{\mathbf{X}\mathbf{X}}$ is the covariance matrix of $\mathbf{X}$. If $\mathbf{X}$ are standardized to $\mathbf{Z}$:

$$\mathbf{Z} = \Sigma_{\mathbf{X}\mathbf{X}}^{-1/2}\big(\mathbf{X}-\mathbb{E}[\mathbf{X}]\big),$$

则 $\mathbb{E}\left[\mathbf{Z}|\widetilde{Y}\right]\in\mathscr{S}_{\tilde{Y}|\mathbf{Z}}.$

### Kriging(Ordinary)

> [DeepLearning](../../../../Learn/Neural%20Network/DeepLearning.md#Kriging) 多种 Kriging 核函数

降维输入$\mathbf{X} \in \mathbb{R}^{p \times 1}$，投影矩阵$\mathbf{W}\in \mathbb{R}^{p \times d}$，$\mathbf{Z}=\mathbf{W}^{\top}\mathbf{X}\in \mathbb{R}^{d \times 1}$，不降维的Kriging模型核函数：$R(\mathbf{x},\mathbf{x}^{\prime})=\exp\left[-\sum_{m=1}^p\theta_m\left(x_m-x_m^{\prime}\right)^2\right]$

低维空间直接Kriging拟合（正常使用降维后的$\mathbf{Z}$与$\mathbf{Y}$构建Kriging模型），核函数为：$R(\mathbf{z},\mathbf{z}^{\prime})=\exp\left[-\sum_{l=1}^d\theta_l\left(z_l-z_l^{\prime}\right)^2\right],$ 其中$z_{l}=\sum_{m=1}^{p}w_{ml}x_{m},l=1,2,\dots d$，可以写成：$R_\mathbf{W}(\mathbf{x},\mathbf{x}^{^{\prime}})=R(\mathbf{z},\mathbf{z}^{\prime})=\exp\left[-\sum_{l=1}^d\theta_l\left(\sum_{m=1}^{p}w_{ml}x_{m}-\sum_{m=1}^{p}w_{ml}x_{m}^{\prime}\right)^2\right]$
即$R_\mathbf{W}(\mathbf{x},\mathbf{x}^{^{\prime}})=\exp\left[-\sum_{l=1}^d\theta_l\left(\sum_{m=1}^{p}w_{ml}(x_{m}-x_{m}^{\prime})\right)^2\right]$

代码中RBF核函数：$k(x_i, x_j) = \text{exp}\left(- \frac{d(x_i, x_j)^2}{2l^2} \right)$

```python
Z = W^{\top} * X
# RBF kernel hyperparameter is l = sqrt(1/(2 * \theta))
kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
gpr = GaussianProcessRegressor(kernel,...)
gpr.fit(Z,Y)
```

“A more ideal way is to embed the projection matrix W into the Kriging kernel function so that the information obtained from the ISDR can be considered while training the Kriging model” ([Song 等, 2024, p. 10](zotero://select/library/items/I2A9MG56)) ([pdf](zotero://open-pdf/library/items/2TRZV9MT?page=10&annotation=2ZLXEUIJ)) 这种方法不显式创建中间变量Z。它直接在原始的高维空间X中定义一个全新的距离度量，这个度量本身就包含了降维的思想。
这篇论文的embed投影矩阵方式，将核函数写为了：
$\mathbf{F}_l(\mathbf{x})=\begin{pmatrix}w_{1l}x_1,w_{2l}x_2,\cdots,w_{pl}x_p\end{pmatrix}^\mathrm{T}.$
$R_l(\mathbf{F}_l(\mathbf{x}),\mathbf{F}_l(\mathbf{x}^{\prime}))=\exp\left[-\theta_l\sum_{m=1}^p\left(w_{ml}x_m-w_{ml}x_m^{\prime}\right)^2\right].$
$R_\mathbf{W}(\mathbf{x},\mathbf{x}^{^{\prime}})=\prod_{l=1}^dR_l(\mathbf{F}_l(\mathbf{x}),\mathbf{F}_l(\mathbf{x}^{^{\prime}}))=\exp\left[-\sum_{l=1}^d\left(\theta_l\sum_{m=1}^p\left(w_{ml}x_m-w_{ml}x_m^{^{\prime}}\right)^2\right)\right].$
可以写成：$R_\mathbf{W}(\mathbf{x},\mathbf{x}^{^{\prime}})=\exp\left[-\sum_{l=1}^d\left(\theta_l \sum_{m=1}^pw_{ml}^{2}\left(x_m-x_m^{^{\prime}}\right)^2\right)\right].$
交换求和顺序：$R_\mathbf{W}(\mathbf{x},\mathbf{x}^{^{\prime}})=\exp\left[-\sum_{m=1}^p\left( \sum_{l=1}^d \theta_l w_{ml}^{2}\right)\left(x_m-x_m^{^{\prime}}\right)^2\right].$

```python
Z = W^{\top} * X
kernel = CustomRBF()
gpr = GaussianProcessRegressor(kernel,...)
gpr.fit(Z,Y)

from scipy.spatial.distance import pdist, cdist, squareform
from sklearn.gaussian_process.kernels import Kernel, StationaryKernelMixin, Hyperparameter

class EmbeddedKernel(StationaryKernelMixin, Kernel):
    def __init__(self, W, length_scale=1.0, length_scale_bounds=(1e-5, 1e5)):
        self.W = np.asarray(W)
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds
    @property
    def anisotropic(self): return np.iterable(self.length_scale) and len(self.length_scale) > 1
    @property
    def hyperparameter_length_scale(self):
        d_out = self.W.shape[1]
        if self.anisotropic: return Hyperparameter("length_scale", "numeric", self.length_scale_bounds, d_out)
        return Hyperparameter("length_scale", "numeric", self.length_scale_bounds)
    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.atleast_2d(X)
        if self.anisotropic: length_scale = self.length_scale
        else: length_scale = np.full(self.W.shape[1], self.length_scale)
        composite_weights = np.sum(self.W**2 / length_scale**2, axis=1)
        if Y is None:
            dists = pdist(X, metric="sqeuclidean", w=composite_weights)
            K = np.exp(-0.5 * dists)
            K = squareform(K)
            np.fill_diagonal(K, 1)
        else:
            if eval_gradient: raise ValueError("梯度只能在 Y 为 None 时计算")
            dists = cdist(X, Y, metric="sqeuclidean", w=composite_weights)
            K = np.exp(-0.5 * dists)
        if eval_gradient:
            if not self.hyperparameter_length_scale.fixed:
                K_gradient = np.zeros((K.shape[0], K.shape[1], len(length_scale)))
                for i in range(len(length_scale)):
                    grad_dist_i = pdist(X, metric='sqeuclidean', w=self.W[:, i]**2)
                    grad_i = K * squareform(grad_dist_i / length_scale[i]**2)
                    K_gradient[..., i] = grad_i
                return K, K_gradient
            else: return K, np.empty((X.shape[0], X.shape[0], 0))
        else: return K
```

***区别***：低维空间直接Kriging的距离指标是和的平方$\left(\sum_{m=1}^pw_{ml}(x_m-x_m^{\prime})\right)^2$ ，而嵌入式的方法为平方和$\sum_{m=1}^p\left(w_{ml}x_m-w_{ml}x_m^{\prime}\right)^2$
