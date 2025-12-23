---
zotero-key: E3R6NA8N
zt-attachments:
  - "7748"
title: A model-free deep learning-based health prognosis methodology with epistemic and aleatoric uncertainties
created: 2025-10-27 12:36:58
modified: 2025-10-27 12:36:59
tags: Epistemic uncertainty  Aleatoric uncertainty  Bayesian deep learning  Health prognosis  Model-free
collections: Inverse SM
year: 2025
publication: Expert Systems with Applications
citekey: sunModelfreeDeepLearningbased2025
author:
  - Bo Sun
  - Junlin Pan
  - Zeyu Wu
  - Qiang Feng
  - Chen Lu
  - Zili Wang
---
| Title        | "A model-free deep learning-based health prognosis methodology with epistemic and aleatoric uncertainties"                                                                                                                                                                                    |
| ------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Author       | [Bo Sun,Junlin Pan,Zeyu Wu,Qiang Feng,Chen Lu,Zili Wang]                                                                                                                                                                                                                                      |
| Organization |                                                                                                                                                                                                                                                                                               |
| Paper        | [Zotero pdf](zotero://select/library/items/E3R6NA8N) [attachment](<file:///D:/Download/Zotero_data/storage/Y8PT5WG8/Sun%20%E7%AD%89%20-%202025%20-%20A%20model-free%20deep%20learning-based%20health%20prognosis%20methodology%20with%20epistemic%20and%20aleatoric%20uncertain.pdf>)<br><br> |
| Project      |                                                                                                                                                                                                                                                                                               |

<!-- more -->

## Background

退化数据中的aleatory uncertainty和model中的epistemic uncertainty “affect the trustworthiness of health prognosis, specifically in terms of the state of health (SOH) and remaining useful life (RUL).” ([Sun 等, 2025, p. 1](zotero://select/library/items/E3R6NA8N)) ([pdf](zotero://open-pdf/library/items/Y8PT5WG8?page=1&annotation=IKHC6NIQ))

## Innovation

“The most significant feature of model-free BDL is that it **does not require assumptions about the distribution types** of the neural network model parameters and output parameters” ([Sun 等, 2025, p. 3](zotero://select/library/items/E3R6NA8N)) ([pdf](zotero://open-pdf/library/items/Y8PT5WG8?page=3&annotation=6EL33W7K))

- “an **advanced dropout approach** is introduced to quantify the epistemic uncertainty” ([Sun 等, 2025, p. 1](zotero://select/library/items/E3R6NA8N)) ([pdf](zotero://open-pdf/library/items/Y8PT5WG8?page=1&annotation=EDYQMQGD))
- “**data augmentation approach** and an arbitrary polynomial chaos expansion method are used in a model-free manner to quantify the aleatoric uncertainty” ([Sun 等, 2025, p. 1](zotero://select/library/items/E3R6NA8N)) ([pdf](zotero://open-pdf/library/items/Y8PT5WG8?page=1&annotation=JKY2JJ4D))
- “a model-free BDL model is constructed for health probability prognosis that incorporates two types of uncertainty.” ([Sun 等, 2025, p. 1](zotero://select/library/items/E3R6NA8N)) ([pdf](zotero://open-pdf/library/items/Y8PT5WG8?page=1&annotation=PCLHLHRJ))
- “an incremental learning approach based on an asynchronous calibration strategy is presented to adaptively update the prediction model.” ([Sun 等, 2025, p. 1](zotero://select/library/items/E3R6NA8N)) ([pdf](zotero://open-pdf/library/items/Y8PT5WG8?page=1&annotation=APFMWMKZ))


## Outlook

目前最终的uncertainty被表示为了一个distribution (mean和variance)，

可以用P-box吗？
- 将aleatoric 量化成distribution (data augmentation)
- 将epistemic 量化成interval (model dropout)

最终输出的数据y中的uncertainty被表示为P-box


## Cases

## Equation

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20251028194511.png)


![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20251028214049.png)


model/epistemic uncertainty通过dropout来进行量化
data/aleatoric uncertainty通过data augmentation和arbitrary polynomial chaos expansion来进行量化

### Model-free epistemic uncertainty quantification

dropout mask distribution to approximate any distribution

$$ g \left(m _ {j} ^ {(l)}\right) = \frac {1}{\sqrt {2 \pi} \sigma_ {l}} e ^ {- \frac {\left(\ln m _ {j} ^ {(l)} - \ln \left(1 - m _ {j} ^ {(l)}\right) - \mu_ {l}\right) ^ {2}}{2 \sigma_ {l} ^ {2}}} \times \left(\frac {1}{m _ {j} ^ {(l)} \left(1 - m _ {j} ^ {(l)}\right)}\right) $$
- dropout mask value of l-th layer: $m_j^{(l)}=Sigmoid(r_j^{(l)})=\frac{1}{1+exp(-r_j^{(l)})}$
- the intermediate hidden variable $r_{j}^{(l)}$ follow the gaussian distribution: $h\left(r_{j}^{(l)}\right)=\frac{1}{\sqrt{2 \pi} \sigma_{l}} e^{-\frac{\left(r_{j}^{(l)}-\mu_{l}\right)^{2}}{2 \sigma_{l}^{2}}}$ 
- the trainable parameters $\mu_l$ and $\sigma_l$ control the shape of the dropout mask distribution.
- 根据概率论，如果两个随机变量存在 $m=k(r)$ 的关系，那么它们的概率密度函数满足 $g(m)=h(r)\left|\frac{d r}{d m}\right|$

### Model-free aleatoric uncertainty quantification

“a small sample time-series data augmentation and probabilistic cognitive approach based on a worm Wasserstein generative adversarial network (WWGAN)” ([Sun 等, 2025, p. 5](zotero://select/library/items/E3R6NA8N)) ([pdf](zotero://open-pdf/library/items/Y8PT5WG8?page=5&annotation=KID5GJ5Q))
- “During the digestion phase” ([Sun 等, 2025, p. 5](zotero://select/library/items/E3R6NA8N)) ([pdf](zotero://open-pdf/library/items/Y8PT5WG8?page=5&annotation=KFDTKQFK))：sliced raw data用于训练WWGAN
- “In the assimilation phase,” ([Sun 等, 2025, p. 5](zotero://select/library/items/E3R6NA8N)) ([pdf](zotero://open-pdf/library/items/Y8PT5WG8?page=5&annotation=P89QIGR2))： **subsequent sliced data输入到训练好的WWGAN中，生成的data与原始data进行wasserstein距离计算，从而进行聚类，划分为不同的group**
- “In the output phase” ([Sun 等, 2025, p. 5](zotero://select/library/items/E3R6NA8N)) ([pdf](zotero://open-pdf/library/items/Y8PT5WG8?page=5&annotation=IS46DPP4))：最后对每个group有一个 trained generator，可以生成与该group相似的data

通过arbitrary polynomial chaos expansion (aPCE)来进行aleatoric uncertainty的不确定性传播
*为什么要通过aPCE*

“Generally, two parameters (a, b) can be used to represent the distribution function of the output in health prognosis” ([Sun 等, 2025, p. 5](zotero://select/library/items/E3R6NA8N)) ([pdf](zotero://open-pdf/library/items/Y8PT5WG8?page=5&annotation=7WQEKVJ8))

### Model-free BDL model construction

integrate epistemic and aleatoric uncertainty into Bayesian Deep Learning (BDL) model:
$p(y|x,X,Y)= \int p(y;a,b)p(a,b|x,\omega )p(\omega|X,Y) \, dadbd\omega$

loss function:
KL divergence ==> maximize the evidence lower bound (ELBO) ==> inference loss function
$loss \propto -ELBO = - \mathbb{E}_{q(\omega|X,Y)}[\log p(Y|X,\omega)] + KL(q(\omega|X,Y)||p(\omega))$
$loss \propto - \frac{1}{N}\sum_{i=1}^{N}\log(p(y_{i};a(x_{i},\omega),b(x_{i},\omega)))+\sum_{i=1}^{L}\lambda \mid\mid \omega^{(l)}\mid\mid^{2},\omega \sim q_{\sigma}(\omega)$

### Adaptive probabilistic health prognosis

“a **dynamic sliding window** algorithm” ([Sun 等, 2025, p. 6](zotero://select/library/items/E3R6NA8N)) ([pdf](zotero://open-pdf/library/items/Y8PT5WG8?page=6&annotation=PFITU77F))

采样K个不同的model parameter $\omega_k$，对每个network，循环M次从$p(y_{i};a(x_{i},\omega),b(x_{i},\omega))$中采样并计算predictive distribution $p(y|x,X,Y)$
循环完$Q=K\times M$次后，计算平均和方差：
$mean = \frac{1}{Q}\sum_{k=1}^{K}\sum_{m=1}^{M}\hat{y}_{m}^{k}$
$var_{total}=\frac{1}{Q}\sum_{k=1}^{K}\sum_{m=1}^{M}\left( \hat{y}_{m}^{k}- mean \right)$
$var_{alearoric}\frac{1}{Q}\sum_{k=1}^{K}\sum_{m=1}^{M}\left( \hat{y}_{m}^{k}- \frac{1}{M}\sum_{m=1}^{M}\hat{y}_{m}^{k} \right)$
$var_{epistemic}=var_{total}-var_{aleatoric}$

