---
zotero-key: XU6ETC7S
zt-attachments:
  - "7875"
title: Graph Neural Networks (GNNs) based accelerated numerical simulation
created: 2025-10-29 08:53:02
modified: 2025-10-29 08:53:03
tags: Surrogate model  Numerical simulation  Machine learning  Graph neural networks
collections: SM
year: 2023
publication: Engineering Applications of Artificial Intelligence
citekey: jiangGraphNeuralNetworks2023
author:
  - Chunhao Jiang
  - Nian-Zhong Chen
---
| Title        | "Graph Neural Networks (GNNs) based accelerated numerical simulation"                                                                                                                                                                                   |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Author       | [Chunhao Jiang,Nian-Zhong Chen]                                                                                                                                                                                                                         |
| Organization |                                                                                                                                                                                                                                                         |
| Paper        | [Zotero pdf](zotero://select/library/items/XU6ETC7S) [attachment](<file:///D:/Download/Zotero_data/storage/XWWYSAN7/Jiang%E5%92%8CChen%20-%202023%20-%20Graph%20Neural%20Networks%20(GNNs)%20based%20accelerated%20numerical%20simulation.pdf>)<br><br> |
| Project      |                                                                                                                                                                                                                                                         |

<!-- more -->

## Background

“Finite element method (FEM) based high-fidelity simulation can be computationally demanding and timeconsuming as engineering problems become more complicated.” ([Jiang和Chen, 2023, p. 1](zotero://select/library/items/XU6ETC7S)) ([pdf](zotero://open-pdf/library/items/XWWYSAN7?page=1&annotation=6SRCQJ3C))

为了提升仿真的效率，有许多不同方面的改进：
- Efficient iterative approaches *limited improvement*
- “model order reduction (MOR).” ([Jiang和Chen, 2023, p. 1](zotero://select/library/items/XU6ETC7S)) ([pdf](zotero://open-pdf/library/items/XWWYSAN7?page=1&annotation=42QDE37M)) *same mesh*
- DL/NN-based surrogate models *different mesh* & *only for Euclidean data*
  - “the increase in offline train time and online simulation time due to the deeper neural networks is undesirable” ([Jiang和Chen, 2023, p. 2](zotero://select/library/items/XU6ETC7S)) ([pdf](zotero://open-pdf/library/items/XWWYSAN7?page=2&annotation=3GK3SQ7Y))

## Innovation

- “a novel surrogate model based on graph neural networks (GNNs)” ([Jiang和Chen, 2023, p. 2](zotero://select/library/items/XU6ETC7S)) ([pdf](zotero://open-pdf/library/items/XWWYSAN7?page=2&annotation=9KUB2MM2))
  - “mesh elements, edges, and nodes are integrated and their features are encoded into graph vertices to provide compressed graph embeddings.” ([Jiang和Chen, 2023, p. 2](zotero://select/library/items/XU6ETC7S)) ([pdf](zotero://open-pdf/library/items/XWWYSAN7?page=2&annotation=52VA974P))  - 
- “global attributes are introduced as another form of graph embedding.” ([Jiang和Chen, 2023, p. 2](zotero://select/library/items/XU6ETC7S)) ([pdf](zotero://open-pdf/library/items/XWWYSAN7?page=2&annotation=QM7B929J))  enhance the efficiency and accuracy
  - each vertex可以接受distant vertices的信息，防止gradient vanishing/efficiency degradation/over-smoothing（over smoothing表示信息传递过多导致节点特征趋于一致）

## Outlook

- 为什么case3的样本量远大于case1和2
- 只能用于简单的2维三角形规整网格吗？
  - 3维：graph不太好构建
  - 其他网格：应该也可以，只需要调整一下vertex 的 attribute
- 没有说明encoder/decoder/Graph conv的优势，无对比
- 相较于其他surrogate model，这种方式可以提高变几何时的泛化性

## Cases

a simply supported (on all of four sides) rectangular metal sheet

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20251030184331.png)


- case1：变量为力的大小
- case2：变量为力的大小和位置
- case3：变量为力的大小和位置，以及几何尺寸

厚度为4mm，弹性模量为$2.1 \times 10^{11}Pa$，泊松比为0.3

输入为mesh information和boundary condition，输出为Von Mises stress of each element

Loss function: $RMSLE=\sqrt{\frac{1}{n}\sum_{i=1}^{n}(\log(\widehat{\sigma}_{i}+1)-\log(\sigma_{i}+1))^{2}},$

Accuracy: $MAPE=\frac{100}{n}\sum_{i=1}^{n}\left|\frac{\sigma_{i}-\widehat{\sigma}_{i}}{\sigma_{i}+\varepsilon}\right|,$ $\varepsilon = 0.001$

数据集样本数量：
- case1/2： train128, validation8
- case3：train8192，validation512


## Equation

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20251030182944.png)


![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20251030182952.png)


![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20251030183022.png)


每个三角形vertex的attribute包括：
- vertex attribute vector:
  - 质心坐标
  - node1边界条件
  - node2边界条件
  - node3边界条件
  - 材料属性
- global attribute vector:
  - 外力大小
  - 外力位置
  - 几何?


Architecture of GNN：

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20251030184038.png)


Train process：

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20251030183842.png)


Flowchart:

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20251030184054.png)
