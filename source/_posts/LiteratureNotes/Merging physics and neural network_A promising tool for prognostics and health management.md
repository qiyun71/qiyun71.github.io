---
zotero-key: AC66GB5S
zt-attachments:
  - "8421"
title: "Merging physics and neural network: A promising tool for prognostics and health management"
created: 2025-12-09 08:37:40
modified: 2025-12-09 08:37:41
collections: SM  0Review
year: 2026
publication: Engineering Applications of Artificial Intelligence
citekey: wangMergingPhysicsNeural2026
author:
  - Fujin Wang
  - Weiyuan Liu
  - Meng Sun
  - Zhi Zhai
  - Zhibin Zhao
  - Xuefeng Chen
---
| Title        | "Merging physics and neural network: A promising tool for prognostics and health management"                                                                                                                                                                                          |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Author       | [Fujin Wang,Weiyuan Liu,Meng Sun,Zhi Zhai,Zhibin Zhao,Xuefeng Chen]                                                                                                                                                                                                                   |
| Organization |                                                                                                                                                                                                                                                                                       |
| Paper        | [Zotero pdf](zotero://select/library/items/AC66GB5S) [attachment](<file:///D:/Download/Zotero_data/storage/CD62T744/Wang%20%E7%AD%89%20-%202026%20-%20Merging%20physics%20and%20neural%20network%20A%20promising%20tool%20for%20prognostics%20and%20health%20management.pdf>)<br><br> |
| Project      |                                                                                                                                                                                                                                                                                       |

<!-- more -->

## Background


## Innovation

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20251209164133.png)


- I-1：通过PM提取测量响应数据(时序数据)中的特征(FRF)，然后将FRF输入NN预测结构参数“PM leverages physical principles to **extract valuable features from the measured raw data**, thereby eliminating the need for feature discovery in the end-to-end prediction using NN.” ([Wang 等, 2026, p. 5](zotero://select/library/items/AC66GB5S)) ([pdf](zotero://open-pdf/library/items/CD62T744?page=5&annotation=UJSN7JL6))
- I-2：同时将测量响应数据和提取特征作为输入，NN预测结构参数“the PM branch can be regarded as a data augmentation” ([Wang 等, 2026, p. 7](zotero://select/library/items/AC66GB5S)) ([pdf](zotero://open-pdf/library/items/CD62T744?page=7&annotation=GH8H8FCA))
- I-3：通过PM仿真响应数据，用于增强测量响应数据，NN预测结构参数“Simulated data also serves as a form of data augmentation.” ([Wang 等, 2026, p. 7](zotero://select/library/items/AC66GB5S)) ([pdf](zotero://open-pdf/library/items/CD62T744?page=7&annotation=2LSLFNGU))
- I-4：通过PM从测量响应数据中计算结构参数，然后计算PM输出与真实结构参数之间的误差，并使用NN建模从测量响应数据到误差之间的映射“the PM is employed for primary predictions, and the NN learns the residuals between the predicted values and the actual values to compensate for errors.” ([Wang 等, 2026, p. 7](zotero://select/library/items/AC66GB5S)) ([pdf](zotero://open-pdf/library/items/CD62T744?page=7&annotation=L83SBKJ4))
- II-1：用NN代替PM中难以获取的参数“one or multiple NNs are used to replace the parameters within the PM. After training, the NN is effectively regarded as the parameters of the PM, allowing the model to generate the predicted value ypred through the PM,” ([Wang 等, 2026, p. 8](zotero://select/library/items/AC66GB5S)) ([pdf](zotero://open-pdf/library/items/CD62T744?page=8&annotation=FBGRQSQJ))
- II-2：用NN学习/输出PM的参数(参数辨识)“uses the NN to learn and output the parameters of the PM” ([Wang 等, 2026, p. 8](zotero://select/library/items/AC66GB5S)) ([pdf](zotero://open-pdf/library/items/CD62T744?page=8&annotation=TR6EJ4Y6))
- II-3：PINN “This method provides new physical insights under limited experimental measurement data, ensuring that the model’s predictions satisfy the laws of physics.” ([Wang 等, 2026, p. 8](zotero://select/library/items/AC66GB5S)) ([pdf](zotero://open-pdf/library/items/CD62T744?page=8&annotation=E2DKMSP8))
- II-4：将特定的计算过程用NN代替“designs the NN based on PM or attributes (Fig. 6(b)), allowing specific computational processes to be replaced by NN, such as Fourier transforms and wavelet transforms commonly used in signal analysis.” ([Wang 等, 2026, p. 8](zotero://select/library/items/AC66GB5S)) ([pdf](zotero://open-pdf/library/items/CD62T744?page=8&annotation=B8SQVZZD))

Physical model-centric中，II-1是将NN代替PM中的参数(NN作为PM的一部分)，而II-2是用NN学习/输出PM的参数(参数辨识)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20251209170256.png)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20251209171325.png)


## Outlook

## Cases

### Physical model-centric BattNN for voltage prediction (II-1 architecture)

> [wang-fujin/BattNN](https://github.com/wang-fujin/BattNN)

将公式中$\delta, V_{b}, R_{sp}$的计算用三个小型NN代替

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20251223172041.png)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20251223172048.png)

### Neural network-centric method for SOH estimation (II-3 architecture)

> [wang-fujin/PINN4SOH: A physics-informed neural network for battery SOH estimation](https://github.com/wang-fujin/PINN4SOH)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20251223172323.png)

## Equation
