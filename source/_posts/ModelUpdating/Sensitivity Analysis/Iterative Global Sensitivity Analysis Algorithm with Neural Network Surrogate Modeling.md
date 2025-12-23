---
zotero-key: 68FH9DLD
zt-attachments:
  - "5147"
title: Iterative Global Sensitivity Analysis Algorithm with Neural Network Surrogate Modeling
created: 2025-08-15 12:37:35
modified: 2025-08-15 12:37:36
collections: NN
year: 2021
citekey: liuIterativeGlobalSensitivity2021
---
| Title        | "Iterative Global Sensitivity Analysis Algorithm with Neural Network Surrogate Modeling"                                                                                                                                                                                                                                                                                                                                                              |
| ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Author       | [Yen-Chen Liu,Jethro Nagawkar,Leifur Leifsson,Slawomir Koziel,Anna Pietrenko-Dabrowska]                                                                                                                                                                                                                                                                                                                                                               |
| Organization | 1 Department of Aerospace Engineering, Iowa State University, Ames IA 50011, **USA** {clarkliu,jethro,leifur}@iastate.edu  2 Engineering Modeling & Optimization Center, School of Science and Engineering, Reykjavik University, Menntavegur 1, 101 Reykjavik, Iceland koziel@ru.is  3 Faculty of Electronics Telecommunications and Informatics, Gdansk University of Technology, Narutowicza 11/12, 80-233 Gdansk, Poland anna.dabrowska@pg.edu.pl |
| Paper        | [Zotero pdf](zotero://select/library/items/68FH9DLD) [attachment](<file:///D:/Download/Zotero_data/storage/IDB4HWXS/Liu%20%E7%AD%89%20-%202021%20-%20Iterative%20Global%20Sensitivity%20Analysis%20Algorithm%20with%20Neural%20Network%20Surrogate%20Modeling.pdf>)<br><br>                                                                                                                                                                           |
| Project      |                                                                                                                                                                                                                                                                                                                                                                                                                                                       |

<!-- more -->

## Background

- GSA需要大量调用FE model simulation，计算量大
- NN-based surrogate model 如何确定合适的训练样本数量，以及计算GSA时model response的数量？

## Innovation

- NN-based surrogate model
- for NN：确定训练样本数量的方法
- for GSA：确定reponse数量的方法

## Outlook

## Cases

## Equation

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20250818114717.png)

收敛准则 “based on the convergence of the Sobol’ indices between successive iterations.” ([Liu 等, 2021, p. 7](zotero://select/library/items/68FH9DLD)) ([pdf](zotero://open-pdf/library/items/IDB4HWXS?page=7&annotation=WTCCC9Q2))
- inner-loop ：$d_r[s_i]=\left|\frac{s_i^{(\mathrm{n})}-s_i^{(\mathrm{n}-1)}}{s_i^{(1)}}\right|,$ “absolute relative change of Sobol’ indices” ([Liu 等, 2021, p. 7](zotero://select/library/items/68FH9DLD)) ([pdf](zotero://open-pdf/library/items/IDB4HWXS?page=7&annotation=WEZXVQ3S))
  - 当$d_r[s_i]\leq\epsilon_r=0.1$时终止循环
- outer-loop：$d_a[s_i]=\left|s_i^{(\mathrm{m})}-s_i^{(\mathrm{m}-1)}\right|,$ “absolute change of Sobol’ indices” ([Liu 等, 2021, p. 7](zotero://select/library/items/68FH9DLD)) ([pdf](zotero://open-pdf/library/items/IDB4HWXS?page=7&annotation=JE6X4V2L))
  - 当$d_r[s_i]\leq\epsilon_r\mathrm{~or~}d_a[s_i]\leq\epsilon_a=0.01$时终止循环

