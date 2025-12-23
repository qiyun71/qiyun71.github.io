---
zotero-key: 48BE68BJ
zt-attachments:
  - '5216'
title: Latent space-based stochastic model updating
created: 2025-08-17T10:39:55.000Z
modified: 2025-08-17T10:39:55.000Z
collections: >-
  Latent space-based active learning framework for likelihood-free Bayesian 
  model updating
year: 2025
publication: Mechanical Systems and Signal Processing
citekey: leeLatentSpacebasedStochastic2025
authors:
  - Sangwon Lee
  - Taro Yaoyama
  - Masaru Kitahara
  - Tatsuya Itoi
date updated: 2025-08-19T18:49:26.000Z
---

| Title        | "Latent space-based stochastic model updating"                                                                                                                                                                        |
| ------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Author       | [Sangwon Lee,Taro Yaoyama,Masaru Kitahara,Tatsuya Itoi]                                                                                                                                                               |
| Organization | a The University of Tokyo, Department of Architecture, 7-3-1, Hongo, Bunkyo-ku, Tokyo, Japan b The University of Tokyo, Department of Civil Engineering, 7-3-1, Hongo, Bunkyo-ku, Tokyo, Japan                        |
| Paper        | [Zotero pdf](zotero://select/library/items/48BE68BJ) [attachment](file:///D:/Download/Zotero_data/storage/6Q7TSL4Z/Lee%20%E7%AD%89%20-%202025%20-%20Latent%20space-based%20stochastic%20model%20updating.pdf)<br><br> |
| Project      |                                                                                                                                                                                                                       |

<!-- more -->

## Background

- “Addressing these uncertainties poses significant challenges, particularly when data and simulations are limited” ([Lee 等, 2025, p. 1](zotero://select/library/items/48BE68BJ)) ([pdf](zotero://open-pdf/library/items/6Q7TSL4Z?page=1&annotation=ZAVQB8MG))

传统的近似贝叶斯(ABC)中，likelihood 是基于distance 计算的：
“The distance-based methods discretize data into finite intervals or bins, enabling the calculation of statistical distances” ([Lee 等, 2025, p. 5](zotero://select/library/items/48BE68BJ)) ([pdf](zotero://open-pdf/library/items/6Q7TSL4Z?page=5&annotation=YY6NZMJC))
- (可靠的PMF需要大量数据)“Reliable PMF estimates require a significant amount of data” ([Lee 等, 2025, p. 5](zotero://select/library/items/48BE68BJ)) ([pdf](zotero://open-pdf/library/items/6Q7TSL4Z?page=5&annotation=INCFVMGS))
- (计算量随数据维度增加)“the computational burden increases with the dimensionality of the data.” ([Lee 等, 2025, p. 5](zotero://select/library/items/48BE68BJ)) ([pdf](zotero://open-pdf/library/items/6Q7TSL4Z?page=5&annotation=WP6UJLV4))
- (超参数的选择)“The number of bins (Nbin) and the width factor should be carefully chosen” ([Lee 等, 2025, p. 5](zotero://select/library/items/48BE68BJ)) ([pdf](zotero://open-pdf/library/items/6Q7TSL4Z?page=5&annotation=WQP83VS5))

probability mass functions (PMFs) 离散的PDF

## Innovation

主要是Bayesian framework 中 likelihood 计算的创新

- “the encoder of a pre-trained VAE [18] is used as q to approximate PDF p” ([Lee 等, 2025, p. 3](zotero://select/library/items/48BE68BJ)) ([pdf](zotero://open-pdf/library/items/6Q7TSL4Z?page=3&annotation=QIFACWYW))，Latent space-based likelihood estimation： $p(\mathbf{x_{obs}}|\mathbf{\theta})\propto\int_{\mathcal{Z}}\frac{p(\mathbf{z}|\mathbf{x_{obs}})p(\mathbf{z}|\mathbf{\theta})}{p(\mathbf{z})}d\mathbf{z}$
- Stochastic model updating in a multi-observation, multi-simulation scenario：$p(\mathbf{X}_{\mathrm{obs}}|\mathbf{\phi})\simeq\prod_{i=1}^{N_{obs}}\sum_{j=1}^{N_{sim}}\int_{\mathcal{Z}}\frac{q(\mathbf{z}|\mathbf{\theta}_{j}^{*})q(\mathbf{z}|\mathbf{x}_{\mathrm{obs}}^{(i)})}{q(\mathbf{z})}d\mathbf{z}$

相当于使用VAE的编码器部分，将$\mathbf{x_{obs}}$降维到$\mathbf{z}$，代替$p(\mathbf{z}|\mathbf{x_{obs}})$部分的计算，然后通过将$\mathbf{x_{sim}}$同样降维到$\mathbf{z}$，代替$p(\mathbf{z}|\mathbf{x_{sim}})=p(\mathbf{z}|h(\theta))=p(\mathbf{z}|\theta)$的计算，然后两者共同计算likelihood用于Bayesian model updating

## Outlook

## Cases

## Equation

simulataion: $x=h(\theta)$

### likelihood estimation

Latent space-based likelihood estimation：
$p(\mathbf{x_{obs}}|\mathbf{z},\mathbf{\theta})=p(\mathbf{x_{obs}}|\mathbf{z})$
$p(\mathbf{x_{obs}}|\mathbf{\theta})=\int_{\mathcal{Z}}p(\mathbf{x_{obs}}|\mathbf{z})p(\mathbf{z}|\mathbf{\theta})d\mathbf{z}$
$p(\mathbf{x_{obs}}|\mathbf{\theta})\propto\int_{\mathcal{Z}}\frac{p(\mathbf{z}|\mathbf{x_{obs}})p(\mathbf{z}|\mathbf{\theta})}{p(\mathbf{z})}d\mathbf{z}$ ， base: $p(\mathbf{x_{obs}}|\mathbf{z})=\frac{p(\mathbf{x_{obs}})\cdot p(\mathbf{z}|\mathbf{x_{obs}})}{p(\mathbf{z})}$, $p(\mathbf{x_{obs}})$ is constant
$\int_{\mathcal{Z}}\frac{p(\mathbf{z}|\mathbf{x_{obs}})p(\mathbf{z}|\mathbf{\theta})}{p(\mathbf{z})}d\mathbf{z}\simeq\int_{\mathcal{Z}}\frac{q(\mathbf{z}|\mathbf{x_{obs}})q(\mathbf{z}|\mathbf{\theta})}{q(\mathbf{z})}d\mathbf{z}$

Stochastic model updating in a multi-observation, multi-simulation scenario：
$\mathbf{\theta}\sim p(\mathbf{\theta}|\mathbf{\phi})$，$\phi$ is hyperparameter of probability distribution
$\mathbf{X}_{\mathrm{obs}}=[\mathbf{x}_{\mathrm{obs}}^{(1)},\mathbf{x}_{\mathrm{obs}}^{(2)},...,\mathbf{x}_{\mathrm{obs}}^{(N_{obs})}]$

$p(\mathbf{X_{obs}}|\phi)=\prod_{i=1}^{N_{obs}}p(\mathbf{x_{obs}^{(i)}}|\phi)$
$p(\mathbf{X}_{\mathrm{obs}}|\mathbf{\phi})\propto\prod_{i=1}^{N_{obs}}\int_{\mathcal{Z}}\frac{p(\mathbf{z}|\mathbf{\phi})p(\mathbf{z}|\mathbf{x}_{\mathrm{obs}}^{(i)})}{p(\mathbf{z})}d\mathbf{z}$
$p(\mathbf{X_{obs}}|\mathbf{\phi})\propto\prod_{i=1}^{N_{obs}}\int_{\mathcal{Z}}\frac{\left(\int_{\theta}p(\mathbf{z}|\mathbf{\theta})p(\mathbf{\theta}|\mathbf{\phi})d\mathbf{\theta}\right)p(\mathbf{z}|\mathbf{x_{obs}^{(i)}})}{p(\mathbf{z})}d\mathbf{z}$， 对$\theta$求积分：$p(\mathbf{z}|\phi)=\int p(\mathbf{z},\theta|\phi) \, d\theta=\int p(\mathbf{z}|\theta,\phi) p(\theta|\phi) \, d\theta$, $p(\mathbf{z}|\theta,\phi)=p(\mathbf{z}|\theta)$

- $p(A|B)=\frac{p(A,B)}{p(B)}$ ==> $p(\mathbf{z},\theta)=p(\mathbf{z}|\theta)p(\theta)$ ==> $p(\mathbf{z},\theta|\phi)=p(\mathbf{z}|\theta,\phi)p(\theta|\phi)$

$p(\mathbf{X_{obs}}|\phi)\simeq\prod_{i=1}^{N_{obs}}\sum_{j=1}^{N_{sim}}\int_{\mathcal{Z}}\frac{p(\mathbf{z}|\mathbf{\theta_{j}^{*}})p(\mathbf{z}|\mathbf{x_{obs}^{(i)}})}{p(\mathbf{z})}d\mathbf{z}$，转化为对$N_{sim}$个$\theta$样本的求和
$p(\mathbf{X}_{\mathrm{obs}}|\mathbf{\phi})\simeq\prod_{i=1}^{N_{obs}}\sum_{j=1}^{N_{sim}}\int_{\mathcal{Z}}\frac{q(\mathbf{z}|\mathbf{\theta}_{j}^{*})q(\mathbf{z}|\mathbf{x}_{\mathrm{obs}}^{(i)})}{q(\mathbf{z})}d\mathbf{z}$

p通过VAE的编码器部分来近似：
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20250819165442.png)

likelihood: $L=\prod_{i=1}^{N_{obs}}\sum_{j=1}^{N_{sim}}l_{i,j}$， $l_{i,j}=\mathcal{L}_{\mathrm{cal}}(q(\mathbf{z}|\mathbf{x}_{\mathrm{obs}}^{(i)}),q(\mathbf{z}|\mathbf{x}_{\mathrm{sim}}^{(j)}))$

通过VAE将高维数据降维，解决了高维计算量大的问题