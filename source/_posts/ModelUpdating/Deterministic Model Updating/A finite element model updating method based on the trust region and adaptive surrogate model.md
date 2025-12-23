---
zotero-key: DNHR3V6L
zt-attachments:
  - "4941"
title: A finite element model updating method based on the trust region and adaptive surrogate model
created: 2025-08-12 02:50:20
modified: 2025-08-12 02:50:42
tags:
  - Kriging
  - model
  - Adaptive
  - surrogate
  - modeling
  - Finite
  - element
  - model
  - updating
  - Global
  - optimization
  - Surrogate
  - model
  - optimization
  - Trust
  - region
collections: Advanced sampling
year: 2023
publication: Journal of Sound and Vibration
citekey: baiFiniteElementModel2023
---
| Title        | "A finite element model updating method based on the trust region and adaptive surrogate model"                                                                                                                                                                                              |
| ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Author       | [Yu Bai,Zhenrui Peng,Zenghui Wang]                                                                                                                                                                                                                                                           |
| Organization | a School of Mechanical Engineering, Lanzhou Jiaotong University, Lanzhou, Gansu, 730070, China  b School of Mechanical Engineering, Xi’an Jiaotong University, Xi’an, Shaanxi, 710049, China                                                                                                 |
| Paper        | [Zotero pdf](zotero://select/library/items/DNHR3V6L) [attachment](<file:///D:/Download/Zotero_data/storage/I4EMSK2Z/Bai%20%E7%AD%89%20-%202023%20-%20A%20finite%20element%20model%20updating%20method%20based%20on%20the%20trust%20region%20and%20adaptive%20surrogate%20model.pdf>)<br><br> |
| Project      |                                                                                                                                                                                                                                                                                              |

<!-- more -->

⭐⭐⭐本质上是一个通过TR-based MPV infill criterion进行选择新样本的优化算法。替换了SSA、PSO等优化算法根据候选样本选择新样本的方法。***TR-based MPV infill criterion要优于优化算法中的采样方法吗?***

## Background

The traditional finite element (FE) model updating method based on a static surrogate model often **suffers from high calculation costs and low updating accuracy**

traditional model updating：
1. construct surrogate model $F_{\Theta}$ with structural parameters x_train and dynamic response y_train
2. find the optimal $x^{*}$ to minimize the residuals between $y^{*}_{pred}$ and $y_{exp}$ with fixed surrogate model $y_{pred}=F_{\Theta}(x)$

如果想要提升surrogate model的精度，则需要继续添加样本。这就需要adaptive sampling，但是在采样时需要比较，选择最合适的样本，需要使用FE model计算候选样本的response，非常耗时。**牺牲效率来提高精度** ***本文没有解决此问题***

## Innovation

***将adaptive kriging(小失效概率可靠性问题)训练代理模型的思想引入到了模型修正***

- 将FE model与实验测量响应之间的残差作为代理模型的输出，将model updating问题(寻找最优参数使得残差最小)转化为surrogate model optimization问题(不断优化kriging，使得残差最小)。目标函数是最小化残差和
  - “The residuals between the calculated responses of the FE model and experimental responses were used as the outputs, and the FE model updating problem was converted into a surrogate model optimization problem.” ([Bai 等, 2023, p. 3](zotero://select/library/items/DNHR3V6L)) ([pdf](zotero://open-pdf/library/items/I4EMSK2Z?page=3&annotation=JM646THZ))
- 使用TR-based MPV infill criterion进行adaptive sampling


Q: 基于Kriging的模型修正"代理模型"训练好后，如何寻找让"代理模型"输出为0对应的输入参数？
A: 关键就是TR-based MPV infill criterion，选择合适的新样本，新样本如果使得
- ✅猜测：应该尽可能通过adaptive sampling method 采样让residuals为0的样本对 
- ✅实际：根据FE计算完候选样本的residuals后，从小到大排序，选择最小的$\eta$个样本，并对这$\eta$个样本取平均得到 $\bar{x}_{\eta}$，即为optimal solution(updated parameters)

## Outlook

- “However, the sampling strategy does not fully utilize the modeling error information of the surrogate model, which may be a drawback of this method.” ([Bai 等, 2023, p. 17](zotero://select/library/items/DNHR3V6L)) ([pdf](zotero://open-pdf/library/items/I4EMSK2Z?page=17&annotation=Z3SG63FF)) --> **a more effective global optimal solution**
- 提升效率只是相较于传统方法需要先构建代理模型，然后用优化算法寻找最优参数的过程。但用adaptive sampling需要对候选样本进行FE simulation，也非常耗时。***使用adaptive surrogate model 可以达到相同的效果***

## Cases

“Fig. 2. Simply supported beam and test.” ([Bai 等, 2023, p. 8](zotero://select/library/items/DNHR3V6L)) ([pdf](zotero://open-pdf/library/items/I4EMSK2Z?page=8&annotation=BGUQVHDL))
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20250814153729.png)

“Fig. 10. Steel truss bridge and test.” ([Bai 等, 2023, p. 13](zotero://select/library/items/DNHR3V6L)) ([pdf](zotero://open-pdf/library/items/I4EMSK2Z?page=13&annotation=8ZQKI4QI))
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20250814153736.png)

## Equation

![image.png|444](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20250814145858.png)
