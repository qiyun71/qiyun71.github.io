---
zotero-key: BX98YIW4
zt-attachments:
  - "5146"
title: Advanced Sobol sensitivity analysis of a 1:4-scale prestressed concrete containment vessel using an ANN-based surrogate model
created: 2025-08-15 12:37:33
modified: 2025-08-15 12:37:34
collections: NN
year: 2025
publication: Nuclear Engineering and Technology
citekey: juAdvancedSobolSensitivity2025
---
| Title        | "Advanced Sobol sensitivity analysis of a 1:4-scale prestressed concrete containment vessel using an ANN-based surrogate model"                                                                                                                                                              |
| ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Author       | [Bu-Seog Ju,Ho-Young Son,Jongryun Lee]                                                                                                                                                                                                                                                       |
| Organization | Department of Civil Engineering, College of Engineering, Kyung Hee University, Yongin-Si, Gyeonggi-Do, 17104, Republic of **Korea**                                                                                                                                                          |
| Paper        | [Zotero pdf](zotero://select/library/items/BX98YIW4) [attachment](<file:///D:/Download/Zotero_data/storage/9XWPT2JC/Ju%20%E7%AD%89%20-%202025%20-%20Advanced%20Sobol%20sensitivity%20analysis%20of%20a%2014-scale%20prestressed%20concrete%20containment%20vessel%20using%20an.pdf>)<br><br> |
| Project      |                                                                                                                                                                                                                                                                                              |

<!-- more -->

## Background

**The safety of nuclear power plants** can be enhanced by **performing sensitivity analyses to identify and evaluate factors that significantly impact their structural integrity**

## Innovation

 - an artificial neural network based surrogate model

使用sobol敏感性分析定量评估 "在各种内部压力水平上containment buildings" 材料不确定性对"internal pressure–displacement behavior"的影响。

the compressive strength of concrete and the prestressing force in the hoop direction 影响 the behavior of containment buildings，并且重要性根据the internal pressure level的改变而改变

## Outlook

## Cases

本文研究对象为prestressed concrete containment vessel
- “an inner liner to prevent the external release of radioactive materials” ([Ju 等, 2025, p. 2](zotero://select/library/items/BX98YIW4)) ([pdf](zotero://open-pdf/library/items/9XWPT2JC?page=2&annotation=PIC25K3D))
- “The structure incorporate concrete, rebars, and tendons to ensure sufficient load-carrying capacity.” ([Ju 等, 2025, p. 2](zotero://select/library/items/BX98YIW4)) ([pdf](zotero://open-pdf/library/items/9XWPT2JC?page=2&annotation=IW6CJTRK))

FE model：
![image.png|333](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20250815211725.png)

输入的12个不确定性参数：
- property of four component 为什么是normal分布？
  - “The components of a PCCV are meticulously controlled during manufacturing to ensure strict tolerances. Consequently, these component uncertainties can be assumed to follow a normal distribution” ([Ju 等, 2025, p. 3](zotero://select/library/items/BX98YIW4)) ([pdf](zotero://open-pdf/library/items/9XWPT2JC?page=3&annotation=ESNNAW9H))
  - “normal distributions can simplify the sampling process when the coefficient of variation is less than 0.3” ([Ju 等, 2025, p. 3](zotero://select/library/items/BX98YIW4)) ([pdf](zotero://open-pdf/library/items/9XWPT2JC?page=3&annotation=UZR6ZPXR))“In this study, concrete had the highest coefficient of variation (0.15) among the material uncertainty factors; therefore, all factors, except for prestress losses, were assumed to follow a normal distribution.” ([Ju 等, 2025, p. 3](zotero://select/library/items/BX98YIW4)) ([pdf](zotero://open-pdf/library/items/9XWPT2JC?page=3&annotation=CNCRBLNF))
- 垂直和水平方向的 prestress losses为什么是均匀分布？
  - “Based on the loss rate expected over the operational life of a nuclear power plant, the uncertainty distributions of vertical and horizontal prestress losses were assumed to be uniform” ([Ju 等, 2025, p. 3](zotero://select/library/items/BX98YIW4)) ([pdf](zotero://open-pdf/library/items/9XWPT2JC?page=3&annotation=VSN2QXHH))

![image.png|444](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20250815212053.png)

此外，输入FE model的characteristic还包括location information (azimuth and height), and internal pressure

输出： the radial displacement of the PCCV at 44 measurement points coinciding with the estimated sensor locations.


## Equation

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20250815205355.png)
