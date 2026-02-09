---
title: Global Sensitivity Analysis
date: 2025-07-14 15:14:37
tags:
  - 
categories: Blog&Book&Paper/Write/Write Paper/Model Updating
---

| Title     | Global Sensitivity Analysis |
| --------- | --------------------------- |
| Author    |                             |
| Conf/Jour |                             |
| Year      |                             |
| Project   |                             |
| Paper     |                             |

敏感性本质上也是进行输入参数降阶的，通过敏感性指数(指标)选择对输出响应影响最大的几个参数

<!-- more -->

## PeerReview

可以直接看这个公式，i就是输出响应的维度，等于他就是把高维响应的每一个值Yi的敏感性指标Svi=Dvi/Di都求出来，然后分子分母分别求和后，再相除。
代码里我是用python一个自带的库，对高维响应的每一个值Yi计算敏感性指标Svi=Dvi/Di （别人代码封装好的），然后计算了Di=Var(Yi) （自己写的），根据Dvi=Svi * Di 反过来拿到Dvi。拿到所有的Di和Dvi后，分子分母求和就可以得到generalized sensitivity indices Sv=Dv/D
至于一阶generalized sensitivity indices为什么数量级很小，我感觉像是因为高维FRF的每一个值算下来的一阶指标Svi=Dvi/Di本身就是有正有负（-0.1,0.1），乘以Y的方差Di(正数)后得到的Dvi也是有正有负，这样求和一下可能Dv就会由于相互抵消变得很小，这样再除以方差的求和D，数量级就变成0.00x了

## Method

- Nonlinear Dimension Reduction——AutoEncoders
- Advanced sampling——
- Sensitivity Index/Loss Function

Data：$\mathcal{D}=\{(\mathbf{x}_{i},\mathbf{y}_{i}),i=1,2,\dots N\}$ total N samples
- Input：$\mathbf{x} = [x_1,\dots ,x_p]^{\mathrm{T}}\in \mathbb{R}^p$
- Output: $\mathbf{y} = [y_1,\dots ,y_{N_{t}}]^{\mathrm{T}}\in \mathbb{R}^{N_{t}}$

$\mathbf{Y}=\left.\left[\begin{array}{cccc}\mathbf{y}_1(t_1)&\mathbf{y}_1(t_2)&\cdots&\mathbf{y}_1(t_{N_t})\\\mathbf{y}_2(t_1)&\mathbf{y}_2(t_2)&\cdots&\mathbf{y}_2(t_{N_t})\\\vdots&\vdots&\ddots&\vdots\\\mathbf{y}_N(t_1)&\mathbf{y}_N(t_2)&\cdots&\mathbf{y}_N(t_{N_t})\end{array}\right.\right]$


>  [A dimension reduction-based Kriging modeling method for high-dimensional time-variant uncertainty propagation and global sensitivity analysis](../../../../ModelUpdating/Sensitivity%20Analysis/A%20dimension%20reduction-based%20Kriging%20modeling%20method%20for%20high-dimensional%20time-variant%20uncertainty%20propagation%20and%20global%20sensitivity%20analysis.md)

- [x] ~~修改高斯核函数，说明为什么？~~
- [x] ~~修改ladle估计器量化方式~~
- [x] ~~Kriging训练损失函数——负对数似然~~ 集成到了scikit-learn库中
- [x] ~~Kriging预测+再训练。~~
  - [x] ~~Importance Sampling Method --> LMC~~
  - [x] MSE --> SIS
- [x] ~~敏感性指标：尝试替换sobol外的其他方法~~

| Name                      | Equation                                                                                                                                                |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Constant Kernel           | $k(x_i, x_j) = constant\_value \;\forall\; x_i, x_j$                                                                                                    |
| White Kernel              | $k(x_i, x_j) = noise\_level \text{ if } x_i == x_j \text{ else } 0$                                                                                     |
| RBF/Gaussian Kernel       | $k(x_i, x_j) = \text{exp}\left(- \frac{d(x_i, x_j)^2}{2l^2} \right)$                                                                                    |
| Matérn kernel             | $k(x_i, x_j) = \frac{1}{\Gamma(\nu)2^{\nu-1}}\Bigg(\frac{\sqrt{2\nu}}{l} d(x_i , x_j )\Bigg)^\nu K_\nu\Bigg(\frac{\sqrt{2\nu}}{l} d(x_i , x_j )\Bigg),$ |
| Rational quadratic kernel | $k(x_i, x_j) = \left(1 + \frac{d(x_i, x_j)^2}{2\alpha l^2}\right)^{-\alpha}$                                                                            |
| Exp-Sine-Squared kernel   | $k(x_i, x_j) = \text{exp}\left(- \frac{ 2\sin^2(\pi d(x_i, x_j) / p) }{ l^ 2} \right)$                                                                  |
| Dot-Product kernel        | $k(x_i, x_j) = \sigma_0 ^ 2 + x_i \cdot x_j$                                                                                                            |

- [x] Dimension reduction based on AutoEncoder
- [x] Multi-fidelity dataset

Two method fusion(DR+MF)：
- train AE in multi-fidelity dataset， then train AESM in multi-fidelity dataset


### NN structure

“(a) CAE-based ROM and (b) FCAE-based” ([Zhang 等, 2025, p. 8](zotero://select/library/items/UMBI6YQW)) ([pdf](zotero://open-pdf/library/items/ME4DQSQS?page=8&annotation=AVUBNZF7))

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20251020161606.png)



“LSTM-Autoencoder” ([Wei 等, 2022, p. 5](zotero://select/library/items/NZYEFEM4)) ([pdf](zotero://open-pdf/library/items/63Q5DF4S?page=5&annotation=KHIREXWE))

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20251020161502.png)


### Framework

As shown in Fig. 1, the procedure of the AE-MLP framework for global sensitivity analysis comprises four main steps:
1. Generate the training dataset $\mathcal{D}=\{\mathbf{x}_{i},\mathbf{y}_{i}\}_{i=1}^{N}$ by performing FE simulations for N different input parameter sets $\mathbf{x}_{i}$, where $\mathbf{y}_{i}$ represents the corresponding high-dimensional response features.
2. Train an Autoencoder $g_{\varphi}(f_{\theta}(\cdot))$ to learn the low-dimensional representation $\mathbf{z}$ of the high-dimensional response data $\mathbf{y}_{i}$. For different types of response data, appropriate AE architectures (e.g., fully connected AE, convolutional AE, recurrent AE) can be selected to effectively capture the underlying patterns.
3. Train surrogate models, specifically MLP $S_{\phi}$, to learn the mapping from the input parameters $\mathbf{x}_{i}$ to the low-dimensional representation $\mathbf{z}$ obtained from the encoder $f_{\theta}$ of the AE.
4. Global sensitivity analysis is performed using the trained AE-MLP model to evaluate the impact of input parameter $\mathbf{x}$ variations on the low-dimensional latent representation $\mathbf{z}$.

For detailed implementation, the training procedure of AE-MLP is summarized in Algorithm 1. After sampling mini-batch $\mathcal{D}_{b}^{AE}$ from the whole training dataset, the AE firstly compute the reconstructed response $g_{\varphi}(f_{\theta}(\mathbf{y}))$ based initialized parameters $\theta$ and $\varphi$. Then, the loss function and its gradients with respect to the parameters are calculated to update the AE parameters using Adam optimization algorithms. After training the AE, the low-dimensional representations $\mathbf{z}_{i}$ are obtained by passing the high-dimensional response data $\mathbf{y}_{i}$ through the trained encoder $f_{\theta}$. Subsequently, the MLP-based surrogate model $S_{\phi}$ is trained using the mini-batch $\mathcal{D}_{b}^{MLP}$. Similarly, the loss function and its gradients are computed to update the MLP parameters using Adam optimization. The train process of both the AE and the MLP is repeated until reaching the maximum epoch. 

Figure 1. The framework of AE-MLP for global sensitivity analysis.

Algorithm 1. The training procedure of AE-MLP.

### Equation

encoder: $f_{\theta}$
decoder: $g_{\varphi}$
surrogate model: $S_{\phi}$

### Algorithm

```
\begin{algorithm}[H]
\caption{Standard Sobol' indices computation directly in origin space}
\label{alg:standard_sobol}
\SetAlgoLined
\setcounter{AlgoLine}{0}
\KwIn{Number of samples $N$, the initial interval $[\underline{X}_i, \overline{X}_i], i \in [1,d]$ trained MLP surrogate model $S_{\phi}(\cdot)$ and decoder $g_{\varphi}(\cdot)$}
\KwOut{Estimated Sobol' indices ${S}_i$ and ${S}_{Ti}$ for each input variable $X_i$}
Generate two independent sample matrices $\mathbf{A}$ and $\mathbf{B}$ of size $N \times d$; \par
\For{$i = 1$ \KwTo $d$}{
  Create a matrix $\mathbf{C}_i$ where the $i$-th column is from $\mathbf{B}$ and all other columns are from $\mathbf{A}$\;
}
Evaluate the model for all rows in all matrices: $ y_A = g_{\varphi}(S_{\phi}(\mathbf{A})),~y_B = g_{\varphi}(S_{\phi}(\mathbf{B})),~y_{C_i} = g_{\varphi}(S_{\phi}(\mathbf{C}_i))$\; \par
% \tcp{This requires $N \times (d+2)$ model evaluations.}
Compute the total variance: ${V}(Y) = \frac{1}{N-1}\sum_{j=1}^{N}(y_{A_j} - \bar{y}_A)^2, ~ \bar{y}_A = \frac{1}{N} \sum_{j=1}^{N} y_{A_j}$\; \par
Compute the first-order index for each input: $ {S}_i = \frac{\frac{1}{N} \sum_{j=1}^{N} y_{B_j} (y_{C_{i,j}} - y_{A_j})}{{V}(Y)} $\; \par
Compute the total-order index for each input: $ {S}_{Ti} = \frac{\frac{1}{2N} \sum_{j=1}^{N} (y_{A_j} - y_{C_{i,j}})^2}{{V}(Y)} $\; \par
\end{algorithm}


\renewcommand{\thealgorithm}{2:}
\begin{algorithm}[H]
\caption{Pull-back of indices computation in origin space}
\label{alg:pullback_sobol}
\SetAlgoLined
\setcounter{AlgoLine}{0}
\KwIn{Number of samples $N$, the initial interval $[\underline{X}_i, \overline{X}_i], i \in [1,d]$ trained MLP surrogate model $S_{\phi}(\cdot)$ and decoder $g_{\varphi}(\cdot)$}
\KwOut{Estimated Sobol' indices ${S}_i^{\mathbf{Y}}$ and ${S}_{Ti}^{\mathbf{Y}}$ for each input variable $X_i$}
Generate two independent sample matrices $\mathbf{A}$ and $\mathbf{B}$ of size $N \times d$; \par
\For{$i = 1$ \KwTo $d$}{
  Create a matrix $\mathbf{C}_i$ where the $i$-th column is from $\mathbf{B}$ and all other columns are from $\mathbf{A}$\;
}
Evaluate latent variables: $\mathbf{Z}_A = S_{\phi}(\mathbf{A}), \mathbf{Z}_B = S_{\phi}(\mathbf{B}), \mathbf{Z}_{C_i} = S_{\phi}(\mathbf{C}_i)$\;\par
Compute mean latent vector: $\boldsymbol{\mu}_z = \frac{1}{N} \sum_{j=1}^{N} \mathbf{Z}_{A,j}$\;\par
Compute total latent covariance matrix: $\mathbf{\Sigma}_z = \frac{1}{N-1} (\mathbf{Z}_A - \boldsymbol{\mu}_z)^T (\mathbf{Z}_A - \boldsymbol{\mu}_z)$\;\par
Compute Jacobian at the mean: $\mathbf{J}_{dec} = \nabla_{z} g_{\varphi}(z) |_{z=\boldsymbol{\mu}_z}$\; \par
Estimate total variance in origin space: $V_{total} = \text{tr}\left( \mathbf{J}_{dec} \cdot \mathbf{\Sigma}_z \cdot \mathbf{J}_{dec}^T + \mathbf{\Sigma}_\varepsilon \right)$\; \par
\For{$i = 1$ \KwTo $d$}{
    $\mathbf{V}_{z,i} = \frac{1}{N} (\mathbf{Z}_B)^T (\mathbf{Z}_{C_i} - \mathbf{Z}_A)$\; \par
    $\mathbf{V}_{z, \sim i} = \frac{1}{N} (\mathbf{Z}_A)^T (\mathbf{Z}_{C_i} - \mathbf{Z}_B)$\;\par
    Compute the first-order index for each input: $S_i^{\mathbf{Y}} = \frac{\text{tr}\left( \mathbf{J}_{dec} \cdot \mathbf{V}_{z,i} \cdot \mathbf{J}_{dec}^T \right)}{V_{total}}$\;\par
    Compute the total-order index for each input: $S_{Ti}^{\mathbf{Y}} = 1 - \frac{\text{tr}\left( \mathbf{J}_{dec} \cdot \mathbf{V}_{z, \sim i} \cdot \mathbf{J}_{dec}^T \right)}{V_{total}}$\;\par
}
\end{algorithm}
```

## Case

To demonstrate the effectiveness of the proposed AE-MLP framework for global sensitivity analysis, three cases including a mathematical problem and two engineering applications are presented in this section. 

All cases in this section are implemented on a computer with Intel i5-13490F CPU and RTX 4060 Ti GPU. The Autoencoder and MLP surrogate models are constructed using the PyTorch library. The Adam optimization algorithm is employed for training both the AE and MLP models, with an initial learning rate of 0.001 and a batch size of 64. The maximum number of training epochs is set to 5000 for both models, with early stopping based on validation loss to prevent overfitting. In addition, 1000 samples are utilized to perform global sensitivity analysis using the SALib library [✨✨].

> [Iwanaga et al., Toward SALib 2.0: Advancing the accessibility and interpretability of global sensitivity analyses, 2022-05-31, Socio-environmental Systems Modelling](zotero://select/library/items/3WE3XILM) 
> [Herman et al., SALib: An open-source Python library for Sensitivity Analysis, 2017-01-10, The Journal of Open Source Software](zotero://select/library/items/WW7X9WY5)

To quantitatively evaluate the performance of surrogate models, the mean relative error (MRE) is employed as the accuracy metric, defined as:
$\mathrm{MRE}=\frac{1}{N_{test}n_{y}}\sum_{i=1}^{N_{test}}\sum_{j=1}^{n_y} \frac{|y_{ij}-\hat{y}_{ij}|}{|y_{ij}|}$
where $N_{test}$ is the number of test samples, $n_y$ is the number of output features, $y_{ij}$ and $\hat{y}_{ij}$ are the actual and predicted output responses for the i-th test sample at the j-th time instant, respectively.

### Numerical case

**data-driven method need substantial samples**， so the accuracy of data-driven method is lower than model-based method in small sample.
- Neural network for large sample
- Kriging for small sample

#### Ishigami function

***输入3输出1***

“Ishigami function” ([Shang 等, 2023, p. 9](zotero://select/library/items/APWQBCLI)) ([pdf](zotero://open-pdf/library/items/4FTIT5GX?page=9&annotation=GKWN7TWA)) **Cokriging**

高保真度模型HF：$y_h(\mathbf{x})=\sin(\pi x_1)+7\sin^2(\pi x_2)+0.1(\pi x_3)^4\sin(\pi x_1)$
低保真度模型LF：$y_l(\mathbf{x})=\sin(\pi x_1)+7.3\sin^2(\pi x_2)+0.08(\pi x_3)^4\sin(\pi x_1)$

uniform distribution $x_{1},x_{2},x_{3}\in[-1, 1]$ 
number of train samples: HF 150, LF 225

| Index | Analytical | Proposed method（Cokriging） |             |             | LHS         |
| ----- | ---------- | -------------------------- | ----------- | ----------- | ----------- |
| s1    | 0.314      | 0.242(23.0%)               | 0.314(0.1%) | 0.314(0.1%) | 0.305(2.8%) |
| s2    | 0.442      | 0.585(32.3%)               | 0.443(0.1%) | 0.442(0.1%) | 0.467(5.7%) |
| s1,3  | 0.244      | 0.160(34.3%)               | 0.245(0.3%) | 0.243(0.4%) | 0.245(0.3%) |
| sT    | 0.557      | 0.368(33.9%)               | 0.564(1.2%) | 0.558(0.1%) | 0.521(6.5%) |
| sT2   | 0.442      | 0.647(46.4%)               | 0.449(1.5%) | 0.442(0)    | 0.454(2.7%) |
| sT3   | 0.244      | 0.159(34.9%)               | 0.250(2.4%) | 0.243(0.2%) | 0.242(0.8%) |
| Nh    | \-         | 30                         | 150         | 300         | 6.0e3       |

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20250801153844.png)


“Ishigami function” ([Palar 等, 2018, p. 182](zotero://select/library/items/3LQAGAB7)) ([pdf](zotero://open-pdf/library/items/P3CAB384?page=8&annotation=HJZGI5K7)) **MF-PCE or MPCE**

解析解：
$\begin{aligned}\mathrm{D}&=\frac{a^{2}}{8}+\frac{b\pi^{4}}{5}+\frac{b^{2}\pi^{8}}{18}+\frac{1}{2},\\D_{1}&=\frac{b\pi^{4}}{5}+\frac{b^{2}\pi^{8}}{50}+\frac{1}{2},D_{2}=\frac{a^{2}}{8},D_{3}=0,\\D_{1,2}&=D_{2,3}=0,D_{1,3}=\frac{8b^{2}\pi^{8}}{225},D_{1,2,3}=0.\end{aligned}$

| Index     | Analytical | MF-PCE | Kriging(RBF) | MLP       | Real(Sampling) |
| --------- | ---------- | ------ | ------------ | --------- | -------------- |
| SU1       | 0.3138     |        | 0.317        | 0.382     | 0.318          |
| SU2       | 0.4424     |        | 0.439        | 0.293     | 0.442          |
| ~~SU3~~   | ~~0~~      |        | ~~0.007~~    | ~~0.025~~ | ~~0.004~~      |
| ~~SU12~~  | ~~0~~      |        | ~~0.005~~    | ~~0.035~~ | ~~0.001~~      |
| SU13      | 0.2436     |        | 0.236        | 0.161     | 0.241          |
| ~~SU23~~  | ~~0~~      |        | ~~0.001~~    | ~~0.043~~ | ~~0.001~~      |
| ~~SU123~~ | ~~0~~      |        |              |           |                |
| SUT1      | 0.5574     |        | 0.560        | 0.641     | 0.558          |
| SUT2      | 0.4424     |        | 0.448        | 0.436     | 0.443          |
| SUT3      | 0.2436     |        | 0.245        | 0.295     | 0.244          |

“Ishigami function” ([Antoniadis 等, 2021, p. 10](zotero://select/library/items/XUVZDR4I)) ([pdf](zotero://open-pdf/library/items/BP52YK32?page=10&annotation=U9ZUW4TH)) Random forests for global sensitivity analysis: A selective review

---

不同kriging核函数的代理模型精度(MSE)、样本数量(Samples)和fit训练时间(Time Cost)

| Name                      | MSE             | Samples | Time cost    |
| ------------------------- | --------------- | ------- | ------------ |
| Constant Kernel           | 12.87041368     | 1248    | 31.74592     |
| White Kernel              | 31.05298668     | 1248    | 64.74881     |
| RBF/Gaussian Kernel       | **0.001510469** | **384** | **11.36942** |
| Matérn kernel             | 0.432411261     | 1075.2  | 151.51033    |
| Rational quadratic kernel | 0.003733838     | 441.6   | 14.56246     |
| Dot-Product kernel        | 12.37060261     | 1248    | 44.64468     |
| Exp-Sine-Squared kernel   | 0.082443308     | 672     | 140.3159467  |

---

***需要在此算例上证明：MLP多保真比单保真要好，并且MLP可以达到跟Kriging相同的精度***

Multi-fidelity训练：
- 方式1：先在low-fidelity数据集上预训练(1000步) ，再在high-fidelity数据集上微调/校正(1000步)
- 方式2：直接在multi-fidelity(low + high)数据集上混合训练(1000步)
High-fidelity训练：直接在high-fidelity数据集上训练(1000步)

训练集：225个low-fidelity样本，150个high-fidelity样本 (同配置)
测试集：1000个high-fidelity样本 (CoKriging论文没有详细说明)

**对于简单算例，样本量少的时候，MLP的精度肯定不如Kriging的精度**

![[J3 Case results#Ishigami|A1:D10<300>{html}]]

---

**提高MLP的低保真训练样本量**

混合训练低保真样本太多了也不行，网络就算能很好地拟合LF model，但是对于HF model的预测还是错误的

![[J3 Case results#Ishigami|A12:D15<300>{html}]]

如果在15000Total训练的基础上，再使用HF微调5000 epochs呢？
RMSE：0.2588，有点进步，但是估计已经到顶
❎MLP在大量的LF上拟合后，再在HF上进行微调后，并不会对HF训练集之外的点进行很好地预测，泛化性差。这也是data-driven方法的通病，即小样本数据

---

另一组测试数据的测试情况：

![[J3 Case results#Ishigami|A17:C21<300>{html}]]

同样的随着样本数量的增加，虽然RMSE逐渐减小，但是还是不如kriging的精度

---

根据sobol总阶敏感性指标进行敏感性大小：
$x_{1}>x_{2}>x_{3}$

#### Math problem 

> 算例描述 训练样本：[80, 120, 160, 400, 700, 1000]

The proposed method is validated on a mathematical problem with 40 input parameters and 101 output responses at different time instants. The problem function is defined as follows:

$\mathbf{Y}(t)=\sin(4\pi t)\sum_{i=1}^{25}\left(\frac{X_i+a_i}{1+a_i}\right)+\frac{1}{25}\sum_{i=26}^{30}\left(X_i-5\pi X_it\right)+\cos(5\pi t)\mathrm{atan}\left(t+\sum_{i=31}^{40}X_i^3b_{i-30}\right)$

where $X_i, i=1,2,\dots,40$ represents independent input parameters and $Y(t)$ represents output responses at time instant t. Following [✨Song 等, 2024], the first 30 input parameters follow a standard normal distribution $\mathcal{N}(0,1)$, while the last 10 input parameters follow a normal distribution $\mathcal{N}(0,0.5)$. The output ranges from 0.015 to 1.5 seconds and is uniformly sampled at 101 time nodes. The coefficients $a_i$ and $b_i$ are defined as: 

$$
a_i = \begin{cases}1,& i=1,2\\ 50,& i=3,4,\dots,25\end{cases}\quad b_i = \begin{cases}6,& i=1,2 \\0.1,& i=3,4,\dots,10\end{cases}
$$

The train and test datasets are generated by evaluating the problem function at different input parameter sets sampled using Monte Carlo sampling. The train dataset consists of varying sample sizes [80, 120, 160, 400, 700, 1000], while the test dataset contains 100 samples to evaluate the accuracy of surrogate models.

> 模型的架构

In this case, the architecture of the Autoencoder used for dimension reduction of output responses is a recurrent autoencoder (RAE) with Long Short-Term Memory (LSTM) layers, which is effective in capturing temporal dependencies in sequential data. The encoder consists of four LSTM layers with 128 neurons each, followed by a dense layer that maps the output to a 16-dimensional latent space. The decoder mirrors the encoder structure, reconstructing the original 101-dimensional output from the 16-dimensional latent representation. For the surrogate model, a Multi-Layer Perceptron (MLP) with three hidden layers containing 128 neurons each is employed to map the 40-dimensional input parameters to the 16-dimensional latent space.

The accuracy of AE-MLP surrogate models is compared with other models including MLP, Kriging, and AE-Kriging. The MLP and kriging directly model the mapping from 40-dimensional input parameters to 101-dimensional output responses. The MLP has the same architecture as the surrogate model in AE-MLP but with an output layer of 101 neurons. The kriging model uses a radial basis function (RBF) kernel for interpolation. The AE-Kriging model employs the same RAE architecture for dimension reduction of output responses as in AE-MLP, followed by a kriging model that maps the 40-dimensional input parameters to the 16-dimensional latent space.

> 不同样本量下四个代理模型的精度，小样本下kriging的精度更高、且AE-kriging的精度比kriging高，随着样本量的增加，AE-MLP的精度逐渐提高超过AE-kriging
> 图

The accuracy of different surrogate models under varying sample sizes is compared. As shown in Figure ⭐, the kriging model outperforms other models in small sample scenarios due to its strong interpolation capabilities. The AE-Kriging model achieves higher accuracy than the standard kriging model by effectively reducing the output dimensionality while preserving essential features. As the sample size increases, the AE-MLP model's accuracy improves significantly, eventually surpassing that of the AE-Kriging model. This indicates that the AE-MLP model benefits from larger datasets, allowing it to learn complex mappings more effectively. Notetably, the standard MLP model exhibits the lowest accuracy across all sample sizes, highlighting the challenges of directly modeling high-dimensional outputs without dimension reduction.

Figure ⭐. Mean Relative Error of different surrogate models at varying sample sizes.

> 1000样本量下代理模型的精度，mean和std of predicted response in test datasets
> 图

Figure ⭐ shows that the mean and standard deviation of output responses obtained from different surrogate models are compared with the theoretical results computed by math problem function. Despite the MLP, all other surrogate models predict accurately the mean of output responses. And, only proposed AE-MLP capture accurately the standard deviation of output responses.

Figure ⭐. Mean and standard deviation of output responses at 1000 sample size.

> GSA分析的结果，排序
> 图

Figure ⭐ presents the generalized first and total sensitivity analysis results obtained from different surrogate models at a sample size of 1000. The sensitivity indices computed by the AE-MLP model closely match the theoretical values, accurately identifying the most influential input parameters on the output responses. In contrast, other surrogate models show discrepancies in sensitivity rankings.


##### Experiment

***输入40输出101***

“A mathematical problem” ([Song 等, 2024, p. 13](zotero://select/library/items/I2A9MG56)) ([pdf](zotero://open-pdf/library/items/2TRZV9MT?page=13&annotation=99YR5DQ4))

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20250903191545.png)

---

可以看出，将输入40降维到16，输出101降维到16的情况下，DR-MLP和DR-Kriging的精度都不是很理想
- 不降维时，MLP的精度比Kriging高
- 降维时，DR-MLP的精度在小样本下不如DR-kriging，但是在大样本情况下精度高于DR-Kriging。样本量为160时DR-Kriging的精度不如DR-MLP应该是训练误差。
- DR-MLP的精度不如不降维的MLP？WHY？ 可能是`40--[128, 64, 32]--16` 网络隐藏层参数太大的原因直接从40到128了
- DR-Kriging的精度比Kriging高，这是正常的。
![[J3 Case results#Mathproblem|A1:J18<300>{html}]]

---

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20250904094521.png)


将input和output降维到跟文献一样，实验：`sample80, input: 40--[128, 64, 32]--2, output: 401--[128, 64, 32]--6`
DR-MLP的结果MSE:1.1912，还是不行，差不多收敛(epoch从5000增加到50000时，MSE也才下降到1.1743)

修改Autoencoder的网络结构，实验：`sample80, input: 40--[40, 24, 12]--2, output: 401--[256, 64, 16]--6`
DR-MLP的结果MSE:2.4661，更差了

网络改大点呢？实验：`sample80, input: 40--[256, 128, 64]--2, output: 401--[512, 256, 64]--6`
DR-MLP的结果MSE: 1.9315，已经收敛（此时增加迭代的次数epoch也不会提高精度）

---

为什么对于DR-kriging，随着样本数量的增加，精度的表现很不稳定(先增后减)

---

```ad-note
对于$X_{31}$和$X_{32}$的敏感性分析结果不太好？
当$b_i=6$时，假设$X_{31},X_{32}=0.5$，t的范围为[0, 1.5]s时，$\arctan$也就是$\tan^{-1}$的取值/波动范围必定比$b_{i}=0.1$时的大
```

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20251103194317.png)


### Experimental case

#### Nasa challenge

> 算例描述+数据集

```ad-note
The NASA Langley Research Center introduced the Optimization under Uncertainty Challenge [23], primarily aimed at the development of new aerospace vehicles and systems designed to function within harsh environments under a range of operational conditions. Its subproblem A, which pertains to model calibration and UQ of Subsystems, is illustrated in Fig. 6. This subproblem specifically addresses the methodology for deriving uncertainty models for aleatory and epistemic parameters based on time-domain response data. This complex case serves to illustrate the accuracy and feasibility of the proposed inverse neural network calibration approach. 
In this problem, the physical system is represented as a black-box analytical model without losing any fidelity. The system response is a time-domain sequential data defined as y(t) = y(a, e, t), where a and e are the aleatory and epistemic variables, respectively. The uncertainty model for a is defined as ai ∼ fa,i = 1,⋯,5, where fa is a unknown Probability Density Function (PDF), presenting aleatory uncertainty. The uncertainty model for e is defined as ej ∈ E, j = 1, ⋯4, where E is an interval representing epistemic uncertainty. 

100 sets of time-domain response data are provided by the Challenge host as the observation set D1, which will be employed to identify the uncertainty characteristics of parameters a and e based on the proposed neural network model.
In the original setup of the Challenge, the aleatory random variables ai, i = 1, ⋯, 5 follow unknown distributions within the range [0.2], and the epistemic parameters ej, j = 1, ⋯4 are unknown-but-fixed constants that lie in the same range [0,2]. A total of 10,000 samples of a and e are generated by evenly sampling within the pre-defined range, as shown in Fig. 7. Using the provided function y(t), their corresponding time responses y(t) are calculated, resulting in a total of 10,000 pairs of datasets. For example, taking a= [1,1,1,1,1] and e = [0.5,0.5,0.5,0.5] as an example, the time response y(t) is computed for a total duration of 5 s. The time response is sampled at 1 ms intervals, yielding a time dimension vector of length 5001. This results in a 10000-by-5001 dimension matrix serving as the input to the proposed inverse neural network model. The vectors a and e are concatenated into a single one-dimensional vector of length 9, which is used as the training label for the network.
```

The proposed AE-MLP framework is further validated on an engineering application involving the global sensitivity analysis of a NASA Langley Research Center challenge problem [✨reference]. The challenge focuses on the calibration and uncertainty quantification of subsystems in aerospace vehicles operating under varying conditions. The system is modeled as a black-box analytical model, with the output being time-domain response data influenced by both aleatory and epistemic uncertainties. The time-domain response is represented as:
$$y(t) = y(a, e, t)$$
where $a$ denotes the aleatory variables and $e$ represents the epistemic variables. The aleatory variables $a_i$ for $i=1,\dots,5$ follow unknown probability density functions within the range [0, 2], while the epistemic variables $e_j$ for $j=1,\dots,4$ are unknown-but-fixed constants also within the range [0, 2]. The time responses are sampled at 1 ms intervals over a duration of 5 seconds, resulting in output vectors of length 5001.

For training the AE-MLP surrogate model, a train dataset comprising 1000 samples is generated by evenly sampling the aleatory and epistemic variables within their specified ranges. Each sample consists of a 9-dimensional input vector formed by concatenating the aleatory and epistemic variables, and a corresponding 5001-dimensional output vector representing the time-domain response. Otherwise, a test dataset containing 100 samples is created similarly to evaluate the accuracy of the surrogate models.

> 模型的架构，由于输出响应的时间相关性和高维度，Autoencoder采用LSTM-based RAE结构进行降维，MLP结构同前。

Due to the temporal dependencies and high dimensionality of the output responses, a Long Short-Term Memory (LSTM)-based Recurrent Autoencoder (RAE) is employed for dimension reduction. The encoder consists of four LSTM layers with 128 neurons each, followed by a dense layer that maps the output to a 16-dimensional latent space. The decoder mirrors the encoder structure, reconstructing the original 5001-dimensional output from the 16-dimensional latent representation. The surrogate model utilizes a Multi-Layer Perceptron (MLP) with three hidden layers containing 128 neurons each to map the 9-dimensional input parameters to the 16-dimensional latent space.

> 1000样本量下，对比不同代理模型的精度，由于Kriging在处理高维输出时面临计算复杂度和内存需求的挑战，因此仅比较MLP、AE-kriging和AE-MLP三种模型的精度。结果表明，AE-MLP模型在捕捉时间域响应特征方面表现出色，显著优于传统的MLP和AE-Kriging模型。

The accuracy of different surrogate models under a sample size of 100 is compared. Due to the computational complexity and memory requirements associated with kriging in handling high-dimensional outputs, only the MLP, AE-Kriging, and AE-MLP models are evaluated. As shown in Table ⭐, the AE-MLP model demonstrates superior performance in capturing the characteristics of the time-domain responses, significantly outperforming both the traditional MLP and AE-Kriging models. This highlights the effectiveness of the AE-MLP framework in modeling complex temporal dynamics in engineering applications. In addition, the time-consuming for function and different surrogate models are also listed in Table ⭐.  The MLP and AE-MLP models exhibit relatively low computational costs due to their efficient neural network architecture. However, due to the large train samples and memory requirements, the kriging model is not suitable for this case and the AE-Kriging requires much more running time than the AE-MLP model.

Table ⭐. Mean Relative Error and time-consuming of different surrogate models.
Methods	MLP	AE-MLP	AE-Kriging	Kriging
Mean relative error	1.2032	0.8567	0.9224	※


From, Tables , we can find that the traditional Kriging model is not applicable to this NASA challenge problem and fails to generate usable results. 
Beside, 加入一句对精度的描述。
The AE-MLP model achieves the highest accuracy among the evaluated surrogate models, with a mean relative error of 0.8567, significantly outperforming the traditional MLP and AE-Kriging models, which have mean relative errors of 1.2032 and 0.9224, respectively.

Besides, the AE-MLP models exhibit relatively low computational costs due to their efficient neural network architecture, while the MLP and the AE-Kriging requires much more running time when dealing with high-dimensional sequential outputs.

> 敏感性分析结果，一阶和总阶敏感性指数雷达图+敏感性排序
> 通过一阶敏感性指标所有models可以辨识最敏感的四个参数$a_{3},e_{2},e_{1},a_{1}$，并且实现很好的排序。但对于其他敏感性指数非常小的参数，所有模型都无法很好进行排序，但这没问题，因为其他不敏感的参数是无关紧要的。
> 通过总阶敏感性指标，只有AE-MLP可以准确辨识最敏感的四个参数，并实现很好排序，其他方法对于第四个敏感的参数$a_{2}$排序不正确。

Figure ⭐ presents the global sensitivity analysis results obtained from different surrogate models. The AE-MLP and AE-Kriging compute the generalized sensitivity indices based on the high-dimensional output responses reconstructed from the decoder of the trained Autoencoder. Otherwise, the latent variables can be also utilized to compute the generalized sensitivity indices for different input parameters. And the sensitivity indices results of these two modes in latent space are called as "AE-MLP-Latent" and "AE-Kriging-Latent" in Figure ⭐. In first-order sensitivity indices, all models accurately identify and rank the four most influential parameters $a_{3}, e_{2}, e_{1}, a_{1}$. However, for other parameters with very small sensitivity indices, all models struggle to provide accurate rankings, which is acceptable as these insensitive parameters are trivial. In total-order sensitivity indices, only the AE-MLP model accurately identifies and ranks the four most influential parameters, while other methods incorrectly rank the fourth most sensitive parameter $a_{2}$.

图：不同方法与Function的generalized 一阶和总阶敏感性指数雷达图对比及排序
Figure ⭐. Generalized first-order and total-order sensitivity indices obtained from different surrogate models.

> 对比三种方法与Function在不同时刻(t=2, 3s)的一阶和总阶敏感性指数对比及排序
> 对于t=2s时的响应，只有AE-MLP可以很好地根据一阶和总阶敏感性辨识和排序最敏感的四个参数，而其他方法对于敏感性参数排序不正确。其中MLP和AE-Kriging对一阶敏感性第四敏感的参数错误地辨识为了$a_2$，对总阶敏感性，MLP错误地排序了$a_{1}$和$a_{2}$，而AE-Kriging对$e_{1},a_{1}$和$a_{2}$排序错误。对于t=3s时的响应，MLP和AE-Kriging对一阶敏感性中$e_{1}$和$e_{2}$的排序错误，并且对总阶敏感性中$a_{1}$和$a_{2}$排序错误，只有AE-MLP可以很好地辨识和排序最敏感的四个参数。

Figure ⭐ presents the global sensitivity analysis results obtained from different surrogate models at specific time instants t=2s and t=3s. For the response at t=2s, only the AE-MLP model accurately identifies and ranks the four most influential parameters based on both first-order and total-order sensitivity indices. Other methods exhibit incorrect rankings for sensitive parameters. Specifically, the MLP and AE-Kriging models incorrectly identify $a_{2}$ as the fourth most sensitive parameter in first-order sensitivity. In total-order sensitivity, the MLP model misranks $a_{1}$ and $a_{2}$, while the AE-Kriging model misranks $e_{1}, a_{1},$ and $a_{2}$. For the response at t=3s, both MLP and AE-Kriging models misrank $e_{1}$ and $e_{2}$ in first-order sensitivity, as well as $a_{1}$ and $a_{2}$ in total-order sensitivity. Only the AE-MLP model successfully identifies and ranks the four most influential parameters. This also demonstrated that the proposed AE-MLP framework is effective in capturing the temporal dynamics of the system and accurately quantifying the influence of input parameters on the output responses over time.

Figure ⭐. First-order and total-order sensitivity indices at t=2s and t=3s obtained from different surrogate models.

> 考虑是不是要与之前论文对比，另一个敏感性分析指标
> 图+表

敏感性指标：
$D(t)=\max_ny(t)-\min_ny(t)$
$\Delta D_i(t)=D_0(t)-D_i(t), 0s\leq t\leq5s$

$e_2>e_1>e_3\approx e_4.$
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20250904101040.png)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20250904100904.png)


#### Folding fin

> 算例描述+数据集
> 如图⭐a)所示，折叠翼结构主要由前翼板、后翼板和转轴组成。
> 折叠翼的动力学特性受多个设计参数影响，包括翼板弹性模量$E_{1}$，翼板密度$\rho_{1}$，翼板泊松比$\nu_{1}$，转轴弹性模量$E_{2}$，转轴密度$\rho_{2}$，转轴泊松比$\nu_{2}$，前翼板翼展$a_{1}$，前翼板弦长$b_{1}$，后翼板翼展$a_{2}$，后翼板弦长$b_{2}$和转轴直径$d$，参数的名义值如表⭐所示。通过动力学仿真得到10个不同位置的XYZ三方向频响函数，作为折叠翼结构的输出响应，如图⭐b)所示。

A folding fin structure is employed to further validate the proposed method. In engineering applications, frequency response functions (FRFs) of folding fin structure are crucial for monitoring the health status of the structure, as they reflect the dynamic characteristics of the structure. Usually, a numerical model is established to emulate the dynamic behavior of real structure, and the FRFs are obtained through simulations. Then the model is updated based on the differences between the simulated FRFs and the measured FRFs from experiments. For determining the main parameters to updated, it is essential to understand the sensitivity of FRFs to various structural parameters of the folding fin structure.

The folding fin structure, as illustrated in Figure ⭐a), mainly consists of a front wing plate, a rear wing plate, and a rotating shaft. And the dynamic characteristics of the folding fin are influenced by several structural parameters, including the wing plate Young's modulus $E_{1}$, wing plate density $\rho_{1}$, wing plate Poisson's ratio $\nu_{1}$, shaft Young's modulus $E_{2}$, shaft density $\rho_{2}$, shaft Poisson's ratio $\nu_{2}$, front wing span $a_{1}$, front wing chord length $b_{1}$, rear wing span $a_{2}$, rear wing chord length $b_{2}$, and shaft diameter $d$. The nominal values of these parameters are listed in Table ⭐. The output dynamic responses of the folding fin are represented by the FRFs in the XYZ directions at 10 different locations, as shown in Figure ⭐b). The real part of FRFs is considered in this case study. The frequency range of interest is from 256 to 1024 Hz and is uniformly sampled at 97 frequency nodes.

Figure ⭐. The case study of folding fin structure: a) Folding fin structure; b) Locations of FRF measurements.

Table ⭐. Structural parameters of folding fin structure.

| Parameter  | Description                | Nominal Value |
| ---------- | -------------------------- | ------------- |
| $E_{1}$    | Wing plate Young's modulus | 71 GPa        |
| $\rho_{1}$ | Wing plate density         | 2700 kg/m³    |
| $\nu_{1}$  | Wing plate Poisson's ratio | 0.33          |
| $E_{2}$    | Shaft Young's modulus      | 200 Gpa       |
| $\rho_{2}$ | Shaft density              | 7850 kg/m³    |
| $\nu_{2}$  | Shaft Poisson's ratio      | 0.3           |
| $a_{1}$    | Front wing span            | 70 mm         |
| $b_{1}$    | Front wing chord length    | 420 mm        |
| $a_{2}$    | Rear wing span             | 160 mm        |
| $b_{2}$    | Rear wing chord length     | 205 mm        |
| $d$        | shaft diameter             | 6 mm          |

A train dataset comprising 1000 samples is generated by the folding fin simulation model at different structural parameter sets sampled using Latin Hypercube Sampling (LHS) within specified ranges, as listed in Table ⭐. Each sample consists of a set of 11 structural parameters as input and the corresponding FRFs at 10 locations in XYZ directions as output. Additionally, a test dataset containing 100 samples is generated similarly to evaluate the accuracy of surrogate models.

Table ⭐. Ranges of structural parameters for train and test dataset.

| Parameter  | Range        |
| ---------- | ------------ |
| $E_{1}$    | [65, 75]     |
| $\rho_{1}$ | [2500, 3000] |
| $\nu_{1}$  | [0.3, 0.36]  |
| $E_{2}$    | [190, 210]   |
| $\rho_{2}$ | [7500, 8000] |
| $\nu_{2}$  | [0.27, 0.33] |
| $a_{1}$    | [75, 85]     |
| $b_{1}$    | [415, 435]   |
| $a_{2}$    | [150, 170]   |
| $b_{2}$    | [200, 210]   |
| $d$        | [5.8, 6.1]   |


> 模型的架构: 考虑到输出的高维97x10x3，通过3.4.1节Multi-dimensional data storage and encoding将其转换为RGB图片，然后使用CAE进行降维，MLP结构同前。

Considering the high dimensionality of the output FRFs (97 frequency nodes × 10 locations × 3 directions), the output data is reshaped into RGB images as described in Section 3.4.1. A CAE is employed for dimension reduction of these RGB images. The encoder consists of several convolutional layers followed by linear layers to map the FRFs to a 64-dimensional latent vector. The decoder mirrors the encoder structure, reconstructing the original RGB images from the 64-dimensional latent representation. The detailed architecture of the CAE is illustrated in Figure ⭐. In addition, a MLP with three hidden layers containing 128 neurons each is utilized to map the structural parameters to the latent vector.

Figure ⭐. Architecture of CAE network for folding fin case.

Due to the high dimensionality of the output FRFs, MLP and kriging models are not feasible for constructing surrogate model in this case. Therefore, the AE-MLP is utilized to validate the proposed method. The accuracy of the AE-MLP surrogate model is evaluated by comparing the reconstructed FRFs from the decoder with the simulated FRFs in the test dataset. In this case, the MRE represented by error heatmap is calculated element-wise for each frequency nodes at every location and direction, averaging the relative error across all samples in test dataset. The mean absolute error (MAE) heatmap is illustrated in Figure ⭐, showing that the AE-MLP surrogate model achieves high accuracy in reconstructing the FRFs across all frequency nodes, locations, and directions. 

> 1. 误差大的地方主要集中在FRF的实部值最大处，表明AE-MLP无法准确预测FRF的实部最大值，但是对于FRF最大值的位置预测较为准确。
> 2. 由于激振力的方向，折叠翼主要在Y方向振动，Y方向实部值绝对误差较大，因此X和Z方向的实部值较小，绝对误差MAE较小。
> 3. 为了更详细地观察AE-MLP预测FRF与目标值的差异，绘制了测试集中某3个样本Node1和6的三方向FRF实部的对比图，如图⭐所示。可以看出，AE-MLP能够很好地捕捉FRF的动态特性，在大部分频率范围内与目标值高度吻合，仅在某些频率点存在较小的偏差。

Figure ⭐. Mean absolute error (MAE) heatmap of FRF real part obtained from AE-MLP surrogate model on test dataset.

As shown in Figure ⭐, the maximum MAE values mainly occur at frequency nodes where the real part of FRFs has the largest values. This indicates that the AE-MLP fails to accurately predict the maximum values of the real part of FRFs. However, it can accurately predict the locations of these maximum values, which is crucial for engineering applications. Moreover, due to the excitation force direction, the folding fin primarily vibrates in the Y direction, resulting in larger absolute errors in the real part of FRFs in the Y direction. Consequently, the absolute errors in the X and Z directions are smaller due to their smaller real part values of FRFs.

To further examine the differences between the FRFs predicted by the AE-MLP surrogate model and the target values, the real parts of FRFs at Node 1 and Node 6 in XYZ directions for three samples from the test dataset are plotted in Figure ⭐. It can be observed that the AE-MLP effectively captures the dynamic characteristics of the FRFs, showing high agreement with the target values across most frequency ranges, with only minor deviations at certain frequency points.

Figure ⭐. Comparison of FRF real parts at Node 1 and Node 6 for three samples from test dataset.

> Time-consuming: AE-MLP, 12.964s VS Simulation, 200s × 100

In addition, the computational efficiency of the AE-MLP surrogate model is evaluated by comparing the time taken to predict FRFs for the entire test dataset with that of the folding fin simulation model. The AE-MLP surrogate model takes only 12.964 seconds to predict the FRFs for all 100 samples in the test dataset, while the folding fin simulation model requires approximately 20000 seconds for the same task. As the number of samples increases, the computational time advantage of the AE-MLP surrogate model becomes even more pronounced. This demonstrates that the AE-MLP surrogate model significantly reduces the computational time while maintaining high accuracy, making it suitable for sensitivity analysis and other real-time applications.

> 敏感性分析结果
> 一阶敏感性：
>     AE-MLP：$a_{2}>E_{1}>a_{1}>d>\nu_{2}>E_{2}>\nu_{1}>\rho_{2}>b_{2}>b_{1}>\rho_{1}$
>     AE-MLP-Latent：$a_{2}>a_{1}>E_{1}>b_{1}>d>\nu_{2}>E_{2}>\nu_{1}>\rho_{2}>b_{2}>\rho_{1}$
> 总阶敏感性：
>     AE-MLP：$a_{2}>E_{1}>a_{1}>b_{2}>E_{2}>b_{1}>\rho_{1}>d>\rho_{2}>\nu_{1}>\nu_{2}$
>     AE-MLP-Latent：$a_{2}>E_{1}>a_{1}>E_{2}>b_{2}>b_{1}>\rho_{1}>d>\rho_{2}>\nu_{1}>\nu_{2}$

The global sensitivity analysis is performed using the trained AE-MLP surrogate model. The generalized first-order and total-order sensitivity indices for the 11 structural parameters are computed based on the latent vector and the FRFs reconstructed from the decoder of the trained CAE, called "AE-MLP-Latent" and "AE-MLP" respectively. As shown in Figure ⭐ and ⭐, the first-order sensitivity indices obtained from both methods consistently identify the front wing span $a_{2}$, wing plate Young's modulus $E_{1}$, front wing chord length $a_{1}$, and shaft diameter $d$ as the most influential parameters affecting the FRFs. The rankings of other parameters show slight variations between the two methods but generally agree on the relative importance of the parameters. In total-order sensitivity indices, both methods again identify $a_{2}$ and $E_{1}$ as the most influential parameters, followed by $a_{1}$, rear wing chord length $b_{2}$ and shaft Young's modulus $E_{2}$. The rankings of other parameters are similar between the two methods, with minor differences in the order of less influential parameters. Overall, the sensitivity analysis results demonstrate the effectiveness of the AE-MLP surrogate model in capturing the influence of structural parameters on the dynamic responses of the folding fin structure.

Figure ⭐. Generalized first-order sensitivity indices obtained from AE-MLP and AE-MLP-Latent.

Figure ⭐. Generalized total-order sensitivity indices obtained from AE-MLP and AE-MLP-Latent.

> Pointwise sensitivity analysis， rear wing span $a_{2}$  and  wing plate density $\rho_{1}$ 对FRFs图片 

The first-order and total-order sensitivity indices for the front wing chord length $a_{2}$ at each frequency node, location, and direction are illustrated in Figure ⭐ and ⭐. It can be observed that the front wing chord length exhibit significant influence on the FRFs across a wide range of frequency nodes, locations, and directions. Notably, certain frequency nodes show particularly high sensitivity indices for the front wing chord length, indicating their critical role in shaping the dynamic responses of the folding fin structure. The spatial distribution of sensitivity indices also reveals that specific locations and directions are more affected by changes in $a_{2}$, highlighting the complex interplay between structural parameters and dynamic behavior.

Figure ⭐. First-order sensitivity indices for front wing chord length $a_{1}$ at each frequency node, location, and direction.

Figure ⭐. Total-order sensitivity indices for front wing chord length $a_{1}$ at each frequency node, location, and direction.



##### Experiment

*本实验选用Simcenter SCADAS Mobile 05数据采集系统和PCB Piezotronics 086C03力锤传感器对折叠翼结构进行冲击激励测试，采集频率响应函数（FRF）数据。如图⭐所示，通过移动力锤法，在折叠翼的不同测点施加冲击力，并使用两个三向加速度传感器测量折叠翼在XYZ方向的动力学响应。实验频率范围为0至3203.1250 Hz，采样点数为1026。*

低频段与实验对应的还可以：(0, 2000)Hz

实验：频率范围(0, 3203.1250) Hz
数据维度（30，1026，10，1，2）： 30组数据，FRF采样点1026，(力锤)测点数10，测量方向1（Y方向），传感器数量(2)
- 前641 frequency

仿真：频率范围(0, 3200) Hz，采样间隔为8Hz
v1数据维度(1000，401, 10, 3)：1000组数据，FRF采样点401，测量节点数10，输出方向3(XYZ)
- 前251 frequency：(1000, 251, 10, 3) --data preprocessing--> (1000, 3, 251, 10) --> NN
- 10个节点`[21977, 22043, 10392, 10368, 20784, 9261, 21056, 9440, 20980, 9593]`
- 0对应0Hz，251对应2000Hz，$\Delta index=8$Hz。索引(32, 128+1)对应(256, 1024)Hz的点

---

样本只有1000个，网络参数设置太大，导致网络无法收敛

---

Autoencoder 在使用ConvTranspose2d后可以达到很好的精度，但由于其没有约束latent space，导致z的数值很大，难以用于MLP-based surrogate model的训练
- 要么使用VAE达到相同的精度
- 要么尝试直接监督latent vector


train AE
- encoder如果最后没有激活函数，AE output accuracy高，但latent vector数值很大，
- encoder如果最后有激活函数tanh，虽然latent vector 数值限制在[-1,1], 同样会导致预测同样的数，是学习率的问题，调小学习率后可以很好地预测并且latent space实现限制

train VAE
- 同理添加tanh和调小学习率后，可以很好地实现重建+约束latent space

train surrogate model based on freeze/unfreeze Decoder/Encoder
- 监督FRF data
  - **Freeze Deocder**：也可以达到很好地精度
  - Do not freeze decoder：
    - good，but difficult to learn distribution of large number
- 直接监督latent vector的方法
  - **Freeze Encoder**：
    - 由于AE的latent space都是large number，导致loss会非常大，反向传播时梯度爆炸？
    - 学习率的问题，以及没有激活函数限制，解决后good可以很好地训练代理模型
  - Do not freeze Encoder：如果不冻结encoder，loss(f(x), encoder(y))会同时优化MLP和Encoder，导致encoder的参数变化，甚至出现对所有y都预测同样的数，虽然loss值很小，但失去了意义


#### Piston (Discard)

***输入7输出1***

> [Piston Simulation Function](https://www.sfu.ca/~ssurjano/piston.html) 活塞模拟函数模拟活塞在气缸内的周期运动。它涉及一系列非线性函数。输出响应是周期时间，单位为秒。

$\begin{gathered}C(\mathbf{x})=2\pi\sqrt{\frac{M}{k+S^{2}\frac{P_{0}V_{0}}{T_{0}}\frac{T_{a}}{V^{2}}}},\mathrm{~where}\\V=\frac{S}{2k}\left(\sqrt{A^{2}+4k\frac{P_{0}V_{0}}{T_{0}}T_{a}}-A\right)\\A=P_{0}S+19.62M-\frac{kV_{0}}{S}\end{gathered}$

| 参数                        | 名称                               |
| ------------------------- | -------------------------------- |
| $M ∈ [30, 60]$            | piston weight $(kg)$             |
| $S ∈ [0.005, 0.020]$      | piston surface area $(m^{2})$    |
| $V_{0} ∈ [0.002, 0.010]$  | initial gas volume $(m^{3})$     |
| $k ∈ [1000, 5000]$        | spring coefficient $(N/m)$       |
| $P_{0} ∈ [90000, 110000]$ | atmospheric pressure $(N/m^{2})$ |
| $T_{a} ∈ [290, 296]$      | ambient temperature $(K)$        |
| $T_{0} ∈ [340, 360]$      | filling gas temperature $(K)$    |

“Piston model” ([Constantine和Diaz, 2017, p. 9](zotero://select/library/items/3MX3A9VH)) ([pdf](zotero://open-pdf/library/items/4UNWCK67?page=9&annotation=4JXFDA7E)) Active subspace --> activity scores
“a cylindrical piston model” ([Zhou 等, 2022, p. 7](zotero://select/library/items/WYUCSUFY)) ([pdf](zotero://open-pdf/library/items/I78RP8RT?page=7&annotation=JR6M7PEQ)) Active subspace + 
Kriging 

“Piston function (PT)” ([Shang 等, 2024, p. 6](zotero://select/library/items/BF4VY3T6)) ([pdf](zotero://open-pdf/library/items/ULWYF55Z?page=6&annotation=QXRTGGLY)) Ensemble PCE based on active learning hybrid criterion

| Name                      | MSE             | Samples | Time cost       |
| ------------------------- | --------------- | ------- | --------------- |
| Constant Kernel           | 0.021021728     | 1248    | 55.72251        |
| White Kernel              | 0.25521185      | 1248    | 95.1271         |
| RBF/Gaussian Kernel       | 0.014940862     | 1190.4  | 124.0729        |
| Matérn kernel             | 0.011875885     | 1017.6  | 49.33325        |
| Rational quadratic kernel | **0.002784104** | **96**  | 0.515690.51569  |
| Dot-Product kernel        | 0.003248555     | 556.8   | **0.053839923** |
| Exp-Sine-Squared kernel   | 0.045786151     | 1248    | 16.28446429     |

---


