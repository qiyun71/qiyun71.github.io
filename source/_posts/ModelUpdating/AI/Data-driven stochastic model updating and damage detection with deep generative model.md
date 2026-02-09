---
zotero-key: BQV3XQBN
zt-attachments:
  - "2665"
title: Data-driven stochastic model updating and damage detection with deep generative model
created: 2025-06-06 13:57:08
modified: 2025-09-23 12:39:35
tags: /done  ⭐⭐⭐⭐⭐
collections: 3.1Frequentist  3Latent space-based active learning framework for likelihood-free Bayesian  model updating  Neural Network  Origin  Data-driven
year: 2025
publication: Mechanical Systems and Signal Processing
citekey: wangDatadrivenStochasticModel2025
author:
  - Tairan Wang
  - Sifeng Bi
  - Yanlin Zhao
  - Laurent Dinh
  - John Mottershead
---

| Title        | "Data-driven stochastic model updating and damage detection with deep generative model"                                                                                                                                                                                                            |
| ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Author       | [Tairan Wang,Sifeng Bi,Yanlin Zhao,Laurent Dinh,John Mottershead]                                                                                                                                                                                                                                  |
| Organization | a Department of Aeronautics and Astronautics, University of Southampton, **UK**  b School of Mechanical Engineering, University of Science and Technology Beijing, China  c Apple, Cupertino, CA, United States  d Department of Mechanical and Aerospace Engineering, University of Liverpool, UK |
| Paper        | [Zotero pdf](zotero://select/library/items/BQV3XQBN) [attachment](<file:///D:/Download/Zotero_data/storage/ZKIEB4MZ/Wang%20%E7%AD%89%20-%202025%20-%20Data-driven%20stochastic%20model%20updating%20and%20damage%20detection%20with%20deep%20generative%20model.pdf>)<br><br>                      |
| Project      |                                                                                                                                                                                                                                                                                                    |

<!-- more -->

## Background

“Is there a calibration algorithm beyond the dominant Bayesian sampling approach and sensitivitybased optimisation in model updating? Can a neural network serve **not only as a surrogate mode**l but also **possess its own calibration capacity**, independent of the Bayesian or optimisation framework?” ([Wang 等, 2025, p. 1](zotero://select/library/items/BQV3XQBN)) ([pdf](zotero://open-pdf/library/items/ZKIEB4MZ?page=1&annotation=LP5UXMBG))

Bayesian缺点：$P(\theta|y_{obs})=\frac{P_L(y_{obs}|\theta)\cdot P\theta}{P(y_{obs})}$
- “Nevertheless, in many instances, the establishment of **the likelihood function** is always either **computationally expensive** due to the high-dimensional integral calculations and the quantification of hybrid uncertainties, or **analytically intractable** because of the complexity of the model” ([Wang 等, 2025, p. 2](zotero://select/library/items/BQV3XQBN)) ([pdf](zotero://open-pdf/library/items/ZKIEB4MZ?page=2&annotation=FS9499MB))
  - 使用“the approximate Bayesian computation (ABC)” ([Wang 等, 2025, p. 2](zotero://select/library/items/BQV3XQBN)) ([pdf](zotero://open-pdf/library/items/ZKIEB4MZ?page=2&annotation=ZI2JED3N))
- “obtaining this integral $P(y_{obs})$ remains difficult” ([Wang 等, 2025, p. 4](zotero://select/library/items/BQV3XQBN)) ([pdf](zotero://open-pdf/library/items/ZKIEB4MZ?page=4&annotation=VAW6DQPT))  $P(y_{obs})$可以保证后验分布的积分为1
  - “MCMC and TMCMC are commonly used. These methods **bypass the need to calculate the normalising factor** by generating samples directly from the unknown posterior distribution.” ([Wang 等, 2025, p. 4](zotero://select/library/items/BQV3XQBN)) ([pdf](zotero://open-pdf/library/items/ZKIEB4MZ?page=4&annotation=FQ8GKHLQ))

ABC “still faces some problems like **over-reliance on the accuracy of sampling method**s and is **time-consuming** when **handling some high-dimensional tasks** that cannot fulfil the timeliness and accuracy required in SHM” ([Wang 等, 2025, p. 2](zotero://select/library/items/BQV3XQBN)) ([pdf](zotero://open-pdf/library/items/ZKIEB4MZ?page=2&annotation=8V7ZJD9H))

## Innovation

- “conditional Invertible Neural Network (cINN)” ([Wang 等, 2025, p. 1](zotero://select/library/items/BQV3XQBN)) ([pdf](zotero://open-pdf/library/items/ZKIEB4MZ?page=1&annotation=J3XAD7JG))
  - “The cINN consists of two parts known as the conditional network and the invertible neural network (INN).” ([Wang 等, 2025, p. 1](zotero://select/library/items/BQV3XQBN)) ([pdf](zotero://open-pdf/library/items/ZKIEB4MZ?page=1&annotation=VDT872U5))
  - “Rather than directly calibrating physical parameters, this multilevel framework focuses on their statistical moments, e.g. mean and variance, referred to as hyperparameters.” ([Wang 等, 2025, p. 1](zotero://select/library/items/BQV3XQBN)) ([pdf](zotero://open-pdf/library/items/ZKIEB4MZ?page=1&annotation=MJ4JNGQQ))

“The introduction of the latent distribution helps avoid a many-to-one relationship between input and observation data, where an element of input data is connected to a combination of observation data and latent space elements” ([Wang 等, 2025, p. 3](zotero://select/library/items/BQV3XQBN)) ([pdf](zotero://open-pdf/library/items/ZKIEB4MZ?page=3&annotation=K9Y79HAF))


## Outlook

- 环境因素(季节性温度变化，不同测试工程师系统性测量误差)“introduce uncertainties that were not considered in this study” ([Wang 等, 2025, p. 24](zotero://select/library/items/BQV3XQBN)) ([pdf](zotero://open-pdf/library/items/ZKIEB4MZ?page=24&annotation=TLK8EDHZ))
- 没有高维算例验证“its scalability remains to be systematically evaluated, as the case studies in this paper involve only lowdimensional examples.” ([Wang 等, 2025, p. 24](zotero://select/library/items/BQV3XQBN)) ([pdf](zotero://open-pdf/library/items/ZKIEB4MZ?page=24&annotation=8YLAUESK))
- 此外高维输入参数场景的应用没有验证，实际中需要“ensuring its efficiency and stability in complex engineering applications.” ([Wang 等, 2025, p. 24](zotero://select/library/items/BQV3XQBN)) ([pdf](zotero://open-pdf/library/items/ZKIEB4MZ?page=24&annotation=8JU2QDIF))

## Cases

### A 3-DOF mass-spring example

![image.png|333](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20250829133305.png)

observation data 使用target value of parameters 生成

无损伤时的参数($k_{1},k_{2},k_{3}$)修正结果，损伤情况1($k_{1}$ reduce 10%)，情况2($k_{2},k_{3}$ reduce 10% 20%)，情况3($k_{1},k_{2},k_{3}$ reduce 10% 10% 20%)下的修正结果。以及三种损伤情况的PoD对比

对比sensitivity-based，bayesian approach(TMCMC+Bhattacharyya distance)和proposed cINN-based method的修正结果，相差不大，但是cINN-based method虽然训练时间相较于sensitivity-based method很长，推理时间很少

### A 3-DOF experimental rig

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20250829133320.png)



## Equation

使用同样的cINN网络参数
- Training：嵌入simulatino data
- Inference：嵌入observation data


![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20250819101706.png)

### INN

“the Change of Variable Rule (CoVR)” ([Wang 等, 2025, p. 5](zotero://select/library/items/BQV3XQBN)) ([pdf](zotero://open-pdf/library/items/ZKIEB4MZ?page=5&annotation=X2DIHTV5))
$P_x(x)=P_Z(Z)\cdot\det\left(\frac{\partial f(x)}{\partial x}\right)=P_Z(f(x))\cdot\det\left(\frac{\partial f(x)}{\partial x}\right)$ 
- $f(x)$表示x和Z之间的可逆变换
- “The transformation requires the calculation of the Jacobian determinant” ([Wang 等, 2025, p. 6](zotero://select/library/items/BQV3XQBN)) ([pdf](zotero://open-pdf/library/items/ZKIEB4MZ?page=6&annotation=Q6XA3DNL))

变换必须要 bijective (invertible+unique)，且在forward和inverse 计算时高效
==> **Affine Coupling Layer (ACL)**

input $x \in \mathbb{R}^{D}$ 被随机地分为两部分 $x_{1} \in \mathbb{R}^{d}, x_{2} \in \mathbb{R}^{D-d}$
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20250819105116.png)
Forward：
$z_1=x_1$
$z_2=x_2\odot\exp(s_1(x_1))+t_1(x_1)$
- “The use of the **exponential** function **ensures that the transformation remains invertible**” ([Wang 等, 2025, p. 6](zotero://select/library/items/BQV3XQBN)) ([pdf](zotero://open-pdf/library/items/ZKIEB4MZ?page=6&annotation=5MZI96Q3))

Inverse：
$x_1=z_1$
$x_2=(z_2-t_1(z_1))\odot\exp(-s_1(z_1))$

⊙ denotes the element-wise multiplication. The scaling function $s_{1}( • )$ and the translation function $t_{1}( • )$ are neural network

“In addition, the special structure of the ACL enables an easy-to-compute Jacobian determinant, a critical component in the CoVR.” ([Wang 等, 2025, p. 6](zotero://select/library/items/BQV3XQBN)) ([pdf](zotero://open-pdf/library/items/ZKIEB4MZ?page=6&annotation=MJYT2QR5))

$J=\begin{bmatrix} 1 &0\\\frac{\partial z_2}{\partial x_1}&\mathrm{diag}(\exp(s_1(x_1)))\end{bmatrix}$

$\det(\boldsymbol{J})=\exp\left(\sum s_1(x_1)\right)$

### Conditional network

融合simulated and observed response data到输入参数x和latent variables Z之间地变换中

![image.png|555](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20250819130312.png)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20250819130736.png)

Forward： **simulation data as condition**
$z_1=x_1\odot\exp(s_2(z_2,y_{sim}))+t_2(z_2,y_{sim})$
$z_2=x_2\odot\exp(s_1(x_1,y_{sim}))+t_1(x_1,y_{sim})$

Inverse： **observation data as condition**
$x_1=(z_1-t_2(z_2,y_{obs}))\oslash\exp(s_2(z_2,y_{obs}))$
$x_2=(z_2-t_1(x_1,y_{obs}))\oslash\exp(s_1(x_1,y_{obs}))$

训练过程中估计的后验分布：
$P(x|y_{sim})=P_Z(Z)\cdot\prod_{i=1}^N\left|\det\left(\frac{\partial f_i(Z_{i-1},\mathbb{S}_\varphi(y_{sim}))}{\partial Z_{i-1}}\right)\right|$

使用基于KL散度的loss function来优化INN（$\phi$）和conditional network($\varphi$)的参数：
$L=KL[P_x(x_{train})\|P(x|y_{sim})]$，优化目标是使得先验分布$P_{x}(x_{train})$和经过逆变换后的参数(后验)分布$P(x|y_{sim})$之间的差异最小
$\widehat{\phi},\widehat{\varphi}=\underset{\phi,\varphi}{\operatorname*{\operatorname*{argmin}}}\mathbb{E}_{P\left(y_{sim}\right)}\{\mathbb{K}\mathbb{L}[P_x(x_{train})\|P(x|y_{sim})]\}$
$\widehat{\phi},\widehat{\varphi}=\underset{\phi,\varphi}{\operatorname*{\operatorname*{argnuin}}}\mathbb{E}_{P\left(y_{sim}\right)}\left\{\mathbb{E}_{P_X(x_{train})}[\mathrm{log}P_x(x_{train})-\mathrm{log}P(x|y_{sim})]\right\}$
$\widehat{\phi},\widehat{\varphi}=\underset{\phi,\varphi}{\operatorname*{\operatorname*{argmax}}}\mathbb{E}_{P\left(y_{sim}\right)}\left\{\mathbb{E}_{P_{x}\left(x_{train}\right)}[\mathrm{log}P(x|y_{sim})]\right\}$ ， training sample distribution $P_x(x_{train})$ is independent of the estimated posterior distribution $P(x|y_{sim})$

$\begin{aligned}&\widehat{\phi},\widehat{\varphi}=\underset{\phi,\varphi}{\operatorname*{\operatorname*{argmax}}}\iint P(x_{train},y_{sim})\cdot\mathrm{log}P(x|y_{sim})dx_{train}dy_{sim}\\&=\underset{\phi,\varphi}{\operatorname*{\operatorname*{\mathrm{argmax}}}}\iint P(x_{rain},y_{sim})\left(\mathrm{log}P_Z\left(Z=f_{\phi,\varphi}\left(x\right)\right)+\mathrm{log}\left|\mathrm{det}J_{\phi,\varphi}\right|\right)dx_{train}dy_{sim}\end{aligned}$

$\widehat{\phi},\widehat{\varphi}=\underset{\phi,\psi}{\operatorname*{\operatorname*{argmin}}} \frac{1}{M} {\operatorname*{\operatorname*{\sum}}}^M_{{m=1}}\left(\frac{\left|f_{\phi,\psi}(x_{train}^m;\mathbb{S}_\varphi(\boldsymbol{y}_{sim}^m))\right|^2}{2}-\log\left|\det\boldsymbol{J}_{\phi,\psi}^m\right|\right)$， adopt a standard Gaussian distribution as the latent distribution
- M是训练样本的数量


### Damage detection

the main application of this paper is damage detection.

initial/prior distribution (intact)
updated/posterior distribution (arbitrarily damaged)

根据参数(刚度or...)初始分布$\mu,\sigma$，设定损伤阈值$\delta=\delta|_{p_{thre}=0.05}=\mu_{ud}-1.645\sigma_{ud}$(低于阈值表示进入damage状态)
如果后验分布的pdf中大部分都位于阈值以下，则Probability of Damage (PoD)很大，反之PoD则很小，标志结构更可能仍然处于安全状态


### Train

数据集：X（n, params）, Y(n, n_y)
- 每个批次大小bs，训练轮数epochs

训练后的推理/修正：X_exp（m, params），Y_exp(m, n_y)
