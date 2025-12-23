---
zotero-key: 8CRYDZCD
zt-attachments:
  - "5364"
title: "Structural damage identification based on physics-guided deep learning: applications to large-scale structures"
created: 2025-08-19 01:33:37
modified: 2025-08-19 01:34:24
collections: Neural Network  Damage Identification
year: 2025
publication: Structural Health Monitoring
citekey: leiStructuralDamageIdentification2025
---
| Title        | "Structural damage identification based on physics-guided deep learning: applications to large-scale structures"                                                                                                                                                                          |
| ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Author       | [Yongzhi Lei,Jun Li,Hong Hao]                                                                                                                                                                                                                                                             |
| Organization | 1Centre for Infrastructural Monitoring and Protection, School of Civil and Mechanical Engineering, Curtin University, Bentley, WA, **Australia** 2Earthquake Engineering Research and Test Center, Guangzhou University, Guangzhou, China                                                 |
| Paper        | [Zotero pdf](zotero://select/library/items/8CRYDZCD) [attachment](<file:///D:/Download/Zotero_data/storage/W34VMLJ7/Lei%20%E7%AD%89%20-%202025%20-%20Structural%20damage%20identification%20based%20on%20physics-guided%20deep%20learning%20applications%20to%20large-scale.pdf>)<br><br> |
| Project      |                                                                                                                                                                                                                                                                                           |

<!-- more -->

## Background

- SHM中的damage identification方向，Physics-guided neural network (PGNN) 方法融合物理定律和数据，因而受到广泛关注
- 现有的PGNN方法“face the problems of poor generalization ability and lack of application to the large-scale structures.” ([Lei 等, 2025, p. 1](zotero://select/library/items/8CRYDZCD)) ([pdf](zotero://open-pdf/library/items/W34VMLJ7?page=1&annotation=7CFZ3GJ4))

## Innovation

- 基于大尺度结构的模型敏感性分析和模型剪枝(model reduction)“a new physics-based loss function” ([Lei 等, 2025, p. 1](zotero://select/library/items/8CRYDZCD)) ([pdf](zotero://open-pdf/library/items/W34VMLJ7?page=1&annotation=C35R6FXZ))

## Outlook

## Cases

## Equation

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20250819094444.png)


$X_{Ext}=[\lambda^{input};\varphi^{input};\lambda^0;\varphi^0;S_\lambda^R;S_\varphi^R]$
- $\lambda^{input};\varphi^{input}$ “denote the eigenvalues and eigenvectors” ([Lei 等, 2025, p. 6](zotero://select/library/items/8CRYDZCD)) ([pdf](zotero://open-pdf/library/items/W34VMLJ7?page=6&annotation=V92FAE72))
- $\lambda^0;\varphi^0$ “are eigenvalues and eigenvectors of the **health structure**,” ([Lei 等, 2025, p. 6](zotero://select/library/items/8CRYDZCD)) ([pdf](zotero://open-pdf/library/items/W34VMLJ7?page=6&annotation=J4LY7HFZ))
- $S_\lambda^R;S_\varphi^R$ “are the eigenvalue and eigenvector sensitivity matrices of the **reduced structure**,” ([Lei 等, 2025, p. 6](zotero://select/library/items/8CRYDZCD)) ([pdf](zotero://open-pdf/library/items/W34VMLJ7?page=6&annotation=CESPBZCE))

应该是damage identification领域的知识

⭐physics-based loss function

$Loss_{p}=Loss_{FCR}+Loss_{MAC}$
$Loss_{FCR}=\frac{1}{Nm}\frac{1}{ns}\sum_{k=1}^{ns}\sum_{i=1}^{Nm}\left|\frac{\lambda_{ki}^{input}-\lambda_{ki}^d}{\lambda_{ki}^d}\right|$
- $\lambda_{ki}^d=\lambda_i^0-\begin{bmatrix}S_\lambda^R\cdot\mathrm{ReLU}(\hat{y}_k)\end{bmatrix}$
$Loss_{MAC}=\frac{1}{Nm}\frac{1}{ns}\sum_{k=1}^{ns}\sum_{i=1}^{Nm}\frac{1-\sqrt{MAC_{ki}}}{MAC_{ki}}$
- $MAC_{ki}=\frac{\left\{\left[\varphi_{ki}^{input}\right]^T\varphi_{ki}^d\right\}^2}{\left\{\left[\varphi_{ki}^{input}\right]^T\varphi_{ki}^{input}\right\}\left\{\left[\varphi_{ki}^d\right]^T\varphi_{ki}^d\right\}}$
- $\varphi_{ki}^d=\varphi_i^0-S_\varphi^R\cdot\mathrm{ReLU}(\hat{y}_k)$
