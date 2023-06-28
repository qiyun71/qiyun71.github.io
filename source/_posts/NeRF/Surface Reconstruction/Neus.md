---
title: Neus
date: 2023-06-14 22:14:49
tags:
    - Neus
    - Surface Reconstruction
categories: NeRF/Surface Reconstruction
---

NeuS: Learning Neural Implicit Surfaces by Volume Rendering for Multi-view Reconstruction

<!-- more -->

[Totoro97/NeuS: Code release for NeuS (github.com)](https://github.com/Totoro97/NeuS)
[Project Page](https://lingjie0206.github.io/papers/NeuS/)

Neus的总目标是实现从2D图像输入中以高保真度重建对象和场景。
现有的神经表面重建方法，如DVR[Niemeyer等人，2020]和IDR[Yariv等人，2020]，需要前景掩码作为监督，很容易陷入局部最小值，因此在对具有严重自遮挡或薄结构的对象进行重建时面临困难。
最近的新视角合成神经方法，如NeRF[Mildenhall等人，2020]及其变体，使用体积渲染来产生具有优化鲁棒性的神经场景表示，即使对于非常复杂的对象也是如此。**然而，从这种学习的隐式表示中提取高质量的表面是困难的，因为在表示中没有足够的表面约束。**
在NeuS中，我们提出将表面表示为有符号距离函数（SDF）的零水平集，并开发了一种新的体积渲染方法来训练神经SDF表示。我们观察到，传统的体积渲染方法会导致固有的几何误差（即偏差）对于表面重建，因此提出了一个新的公式，它在一阶近似中没有偏差，从而即使在没有掩码监督的情况下也能实现更准确的表面重建。
在DTU数据集和BlendedMVS数据集上的实验证明，NeuS在高质量表面重建方面优于现有技术，尤其是对于具有复杂结构和自遮挡的对象和场景。
# 引言+相关工作

## SDF简单理解
SDF：输入一个空间中的点，输出为该点到某个表面（可以是曲面）最近的距离，符号在表面外部为正，内部为负。
给定一个物体的平面，我们定义SDF为空间某点到该平面距离为0的位置的点的集合（也就是物体表面的点）。如果空间中某点在物体表面之外，那么SDF>0；反之如果空间中某点在物体表面之内，那么SDF<0。这样子就可以找到物体表面在三维空间中的位置，自然而然的生成三维表面。
  1. mesh是一种由图表示的数据结构，基于顶点、边、面共同组成的多面体。它可以十分灵活的表示复杂物体的表面，在计算机图形学中有着广泛的应用。从nerf输出的物理意义就可以想到，density可以用来表示空间中沿光线照射方向的密度，**那么我们可以通过基于密度的阈值来控制得到的mesh**。这种方法的好处是，训练好一个nerf的模型就可以得到一个mesh了。但是这种方式也有很大的缺点：一是训练结果会有很多噪音而且生成的mesh会有很多的空洞，二是很难控制一个合理的阈值。
  2. 这里我们可以考虑使用有向距离场（Signed distance function 简称 SDF）来取代nerf建模。使用SDF的一大好处是，SDF函数本身在空间是连续的，这样子就不需要考虑离散化的问题。我们之后使用Marching cubes方法来生成mesh。
  3. NeRF生成一个带有密度和颜色信息的模型，通过使用SDF来代替密度，在密度大的地方表示为物体的表面，就可以生成一个mesh模型。

>[旷视3d CV master系列训练营二： 基于NeRF和SDF的三维重建与实践 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/553474332)

## 相较于以往的工作：

![Pasted image 20230531185214.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/Pasted%20image%2020230531185214.png)


IDR无法应对图片中深度突然变化的部分，这是因为它对每条光线只进行一次表面碰撞（上图（a）上），于是梯度也到此为止，这对于反向传播来说过于局部了，于是困于局部最小值，如下图IDR无法处理好突然加深的坑。

NeRF的体积渲染方法提出沿着每条光线进行多次采样（上图（a）下）然后进行α合并，可以应对突然的深度变化但NeRF是专注于生成新视点图像而不是表面重建所以有明显噪声。

>[NeuS: Learning Neural Implicit Surfaces by Volume Rendering for Multi-view Reconstruction - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/496752239)

新的体积渲染方案：Neus，使用SDF进行表面表示，并使用一种新的体积渲染方案来学习神经SDF表示。
## 本文方法

本文集中在通过经典的绘制技术从2D图像中学习编码3D空间中的几何和外观的隐式神经表示，限制在此范围内，相关工作可以大致分为基于表面渲染的方法和基于体积渲染的方法
- 基于表面渲染：假设光线的颜色仅依赖于光线与场景几何的交点的颜色，这使得梯度仅反向传播到交点附近的局部区域，很难重建具有严重自遮挡和突然深度变化的复杂物体，因此需要物体mask来进行监督
- 基于体积渲染：eg_NeRF，通过沿每条射线的采样点的α-合成颜色来渲染图像。正如在介绍中所解释的，它可以处理突然的深度变化并合成高质量的图像。提取学习到的隐式场的高保真表面是困难的，因为基于密度的场景表示对其等值集缺乏足够的约束。
相比之下，我们的方法结合了基于表面渲染和基于体积渲染的方法的优点，通过将场景空间约束为带符号距离函数SDF，但使用体积渲染来训练具有鲁棒性的representation。
- 同时UNISURF也通过体积渲染学习隐式曲面。在优化过程中，通过缩小体积渲染的样本区域来提高重建质量。
**UNISURF用占用值来表示表面，而我们的方法用SDF来表示场景**，(因此可以自然地提取表面作为场景的零水平集，产生比UNISURF更好的重建精度。)

# 方法

{% note info %} 构建了一个无偏，且collusion-aware的权重函数w(t) = T(t)ρ(t) {% endnote %}


给定一组三维物体的pose图像，目标是重建该物体的表面，该表面由神经隐式SDF的零水平集表示。
为了学习神经网络的权重，开发了一种**新的体积渲染方法**来渲染隐式SDF的图像，并最小化渲染图像与输入图像之间的差异。确保了Neus在重建复杂结构物体时的鲁棒性优化。

## 渲染步骤
### 场景表示

被重构物体的场景表示为两个函数：
- f：将空间点的空间坐标映射为该点到物体的符号距离
- c：编码与点x和观察方向v相关联的颜色
这两个函数都被MLP编码，物体的表面有SDF的零水平集表示 $$\mathcal{S}=\left\{\mathbf{x}\in\mathbb{R}^3|f(\mathbf{x})=0\right\}.$$

定义概率密度函数 $$\begin{aligned} \phi_s(x)& =se^{-sx}/(1+e^{-sx})^2  \end{aligned}$$ 

其为sigmoid函数的导数 $$\Phi_s(x)=(1+e^{-sx})^{-1},\text{i.e.,}\phi_s(x)=\Phi_s'(x)$$


### Neus与NeRF 体渲染函数对比

| Project          | Neus                                                                                                                   | NeRF                                                                                                                                                                                      |
| ---------------- | ---------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 渲染函数         | $$C(\mathbf{o},\mathbf{v})=\int_{0}^{+\infty}w(t)c(\mathbf{p}(t),\mathbf{v})\mathrm{d}t,$$                               | $$\mathrm{C}(r)=\int_{\mathrm{t}_{\mathrm{n}}}^{\mathrm{t}_{\mathrm{f}}} \mathrm{T}(\mathrm{t}) \sigma(\mathrm{r}(\mathrm{t})) \mathrm{c}(\mathrm{r}(\mathrm{t}), \mathrm{d}) \mathrm{dt}$$ |
| 权重             | $$w(t)=T(t)\rho(t),\text{where}T(t)=\exp\left(-\int_0^t\rho(u)\mathrm{d}u\right).$$ **无偏、且遮挡**                     | $$w(t)=T(t)\sigma(t) , \text { where } \mathrm{T}(\mathrm{t})=\exp \left(-\int_{\mathrm{t}_{\mathrm{n}}}^{\mathrm{t}} \sigma(\mathrm{r}(\mathrm{s})) \mathrm{ds}\right)$$   **遮挡但有偏**  |
| 不透明度密度函数 | $$\rho(t)=\max\left(\frac{-\frac{\mathrm{d}\Phi_s}{\mathrm{d}t}(f(\mathbf{p}(t)))}{\Phi_s(f(\mathbf{p}(t)))},0\right).$$ | $$\sigma(t)=\phi_s(f(\mathbf{p}(t)))$$                                                                                                                                                      |
| 离散化                 |       $$\hat{C}=\sum_{i=1}^n T_i\alpha_i c_i,$$  $$T_i=\prod_{j=1}^{i-1}(1-\alpha_j)$$ $$\alpha_i=\max\left(\frac{\Phi_s(f(\mathbf{p}(t_i))))-\Phi_s(f(\mathbf{p}(t_{i+1})))}{\Phi_s(f(\mathbf{p}(t_i)))},0\right).$$                                                                                                                 |                                                               $$\hat{C}(\mathbf{r})=\sum_{i=1}^{N} T_{i}\left(1-\exp \left(-\sigma_{i} \delta_{i}\right)\right) \mathbf{c}_{i} \text {, where } T_{i}=\exp \left(-\sum_{j=1}^{i-1} \sigma_{j} \delta_{j}\right)$$                                                                                                                            |

权重函数，Neus(右)，NeRF(左)
![Pasted image 20230606154119.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/Pasted%20image%2020230606154119.png)




## 训练

loss函数
$$\mathcal L=\mathcal L_{color}+\lambda\mathcal L_{reg}+\beta\mathcal L_{mask}.$$
$$\mathcal{L}_{color}=\frac{1}{m}\sum_k\mathcal{R}(\hat{C}_k,C_k).$$
类似[IDR](https://lioryariv.github.io/idr/)

---
$$\mathcal{L}_{r e g}=\frac{1}{n m}\sum_{k,i}(\|\nabla f(\hat{\mathbf{p}}_{k,i})\|_{2}-1)^{2}.$$
[IGR](https://github.com/amosgropp/IGR)

---

$$\mathcal{L}_{mask}=\mathrm{BCE}(M_k,\hat{O}_k)$$
沿着相机ray的权重之和：
$$\hat{O}_k=\sum_{i=1}^n T_{k,i}\alpha_{k,i}$$
是否使用mask监督: (BCE是二值交叉熵损失)
$$M_{k} ∈ {0, 1}$$

分层采样类似NeRF



# Code

## Runner().train流程图
<iframe frameborder="0" style="width:100%;height:1153px;" src="https://viewer.diagrams.net/?highlight=0000ff&edit=_blank&layers=1&nav=1&title=Runner.drawio#R7V1Zk6M4Ev41RFd1RBPcx6Ndx8zG7vTudE3s7DwRlC3bTBlwA66jf%2F1KQtiA0gbbXG57ImraiEtk6stM5SEJ6p3%2F%2Fkvkrha%2FhVO0FBRp%2Bi6o94KiaJZq4H9Iy0faohiWlrbMI2%2BatsnbhifvB2KNEmtde1MUFy5MwnCZeKti4yQMAjRJCm1uFIVvxctm4bL41pU7R1zD08Rd8q1%2FetNkkbZairlt%2FxV580X2Ztmw0zO%2Bm13MviReuNPwLdekPgjqXRSGSfrLf79DS0K9jC7pfY87zm46FqEgqXPDaKV9cxdf3e9%2F%2BN%2FXc%2FlpFKhfvrCnvLrLNftgQTGW%2BHnjWYgfi3udfDBSGN%2FXYXbiS0wZNcIXyMbqfXsS%2F5qTf7%2BtMTui7Fm4U%2Bnj0pOMHpsnK1G4DqaI9FPGp98WXoKeVu6EnH3D4wq3LRJ%2FyU6TJ7FxIhubp%2BXpkH0UihL0nmtidPkFhT5Kog98SXY2YxobpXLGs7cty2WJtS1y7NZYm8tG2Xzz6C0j8A%2FGiwP4YnA0QlM8LtlhGCWLcB4G7vJh2zreUlHCR9tr%2FhWGK0a7v1GSfDDiueskLFIWvXvJ%2F8jtos6O%2FsqduX9nT6YHH9lBgD83dxM5%2FCt%2FbnsbPcruq83FOFxHE7SHVAqTCm40R8me65jcIXTcOyYitHQT77WI%2F8YZrHDAW0XhBMVxKstm3pxIKw8zdjdYpH7AosuY1Xm4KDoAF1Xn4aK3BRfzCpe6cNFqwkUfFFw0Di73buLGKLm5PQOAqJA%2BgQDSmj6RrStC6iJEr4kQ2RwURHQOIl9R8hZGL%2FE5IMTqW4VYAPm%2BPd58%2Fhyj5UwkKlnQx598MsMRsWk7%2ByTo98OTPapctGRVmaerqnZpyWYdyBH26f6RDU2YvPF05gTpBWdCZaN3KvMTuScvmC%2FRf93Ic4MJ2kvwV3bReVFdA8a2ZnRKdd6K%2F4YwjSJM%2Br0Ej7KrzoziwDjXuh3nKkdxjmrxwl2Rn5N1tPwYR%2B7khajxKvIVfRAtEFOxlbLSk3WOnApATdlqi5wb39TVLKw0CzNzr9ouZKAZiF2Y9TsHmSSMJgsxXCWeL46mrj84wcMbiJrOY6VbAxGQNFesVECgEivZE4eCFZvDyle0fkqVOorOwtmgq31PpZSrO66%2B9zoLalUiZWD%2Ba37CsQzdKW6ZLNDkZRV6NN5Dwz%2FPURb5IQ5t5xlbZOvVGSDJkHpHEm%2FueoGXcLRLyPxtTnHUPeG0UvDMNEVAV0sb67dAOb0lyqlXy7a%2BDKobE1CGZdkqfFQgonFnMYlcLzgLbW2YvcuYa6y5PlLqxgaUYQGFjw1ELEFDoodSmq9xFoiRew8VKHysgAqcQatlwzD7V8rXKXR9UVN3Cq0Oawqt8FPop7Xvu9HHnxGGwHnIGFPrW8ZkL7tCpRoqat05tKoOCioqP4der6ZugpwlcqOAhIcifDRExKiKmIViGGQsKMfFVEWzU9TYV9TURo1aFzXWsFDD%2B0Rw%2Fx3Pd%2BfIWaHIHyJeiIaRC3ixe3cuadecsPpgqesi0YaVFKbyLpI5CrBaCaahj%2F%2F5iB13kDmUmmaIuilt%2FlNKygZyMVqGaJldaht%2BHthM9YRPJ5MSm1RKS4Kt8ymlEKWSpIMm7J1WU2QPvjIqjzBbKrAJyKbqmEsKzyV9%2FOwmk4VDGaHcEUZIgn5P%2FVaELfgCKsRCcbJaEzlGLqEtr%2FmWSbgMo%2FSn78Yv%2BC56%2B136HJk8Ub%2FnWIQpmRT5ECdR%2BILu6NPU%2ByAMiEqcectlqcldevMAH04wfxBuHxO%2BeBN3OWInfG863ekXKgrfVkCqlXgPGO9gAqPSGvOBSJc%2Fd7zpO%2BER5VR%2BJFwMp3S9yCkD1HxQ1mN7rDJ4nD7Ygj0SrEfhwRDsO2E8Ii1Y7lu28KCTdtsQHjRhbJFG3DK%2BE0b0hz0Wxg%2FCg0muxMr74ZGcsu7oNbYwNvY%2B2RCssTCy6cBgjx%2BN6AtVeocpjDX6HlOwNcGmz7AswaZvtnT6DJM8cvx4QUNKMYtDyqw9pMy2hpR%2BdQzXL%2BCqm4eYTTgHMhXR%2BDzEALmRMyN%2FEZ6NxKsFigbp7aKTEaOoMWHUdD0F0Xhve2oP5Syh6QXJtrK6lHWQTXKX%2BlLnZx9k4KccmpEfF8MeQ1VFreg3ViSIQ9YWcJ3wyJA5LlwVUIVeqVRAmQnRvQKyvv%2FD%2BLf3I%2FDff3%2F%2BcL%2BPvX8%2BhH2npeY4vOV3szxulqcwEY1BGRVZv%2FP6jyZrD9KMMFUxyx3ZrIcD6iciJrUOzQidTxDK%2FFhT7zXzYW0si8ysuGN6bMa0WXpHvHKD7BaS6zunhHei%2BXPON5a%2FKNecexvQAegNkzB23AB3ZEmCpV544DsuRvXqWbR5s2gHNPK6dfoY1ySH%2BhLaqKt1h1UooPPeolx9N6n1vlkl8RCltaGLVtFW1TQwtV2zxE4DtzoQdyo6yqXP%2BC9wYtdfLVG88aBSR7di4FZMo%2BA5Xl2WCDRkTZSL%2BlcDixWwAhbVLqcfOj%2BNr%2BAoYad6YUELgH0qyD5pa2d1o8eu7svaeiybFFa7L4c10zB4Lw5d2CFwPH%2BFuUfWz6CClolXZl9mEdS8ZluE%2FvM67kmr6bqoFJNdVRBEJJCfza7yIGptLQKDjwB%2BIH7top9XvulS0XepyRBftE4N9Gs%2BZX3BVjdFzBhWPqXBp4itV8zQGKQXhdjlSgEpuqodYJdrbZW3GEBgPO%2FEALKMiDT7wgQTyWdZolnCpxnlQjq8A6Rs0G%2BcMxWekHzbDwf3OT7snvJ78Uyu%2BgFHuXhgPftIXkHObEarEydodeBXkNFPbWvl82fv6jvaYXlLhmiWIKdAykm1YMdlewqKDyrPSKaX5JE%2FkpMXucEc3UAD5ZZC7mKYqBiaqBR9gLoETp8MzG27HSaCMYUzjL0VTIytxdF4XCazFQbs9YNZeoY87SrWBvEUJOLA7ETe6zhxEyc1HQZrKBYduAaYbNC5oWjyvoRymnPR3CG2DmfmXIzmAhx%2FugGaH9p2atCJ5lJaMvj3BV8hK7Yrmz9Ab84x97U3N1i6MWHyjUfnBMThJgvpohaE6hJs9x3Tj8s1%2FhVNNIvJj7AYxc2i3pLbHUQftBZsI%2Bjjq06KsbQxJ6DZfBT%2FKItpEqKhdSZNP7S8ttp1oAIDFQzYEsNAaWmWCg5UYJ0qmqTkTMIIOeE6ib3pIH1thlF0SZsqFCjYauemrSdY6fYabct%2BH5bH18J8EaQNUBsA07C32oB93W7LcUqvSKuEyNgmkfO9dhR0%2FbGmy9F9T%2B0sZ4YQ0KGc6kj1BhMibXaIaaCphy0umHypoxZFs6sjFZanllqSp2CIT1UUEYi8Gm2pp16DfMOWp3VdNQPzvml9cnQw3reTOCr3tlbUvm7z2ScbwX8eqSdaDfnXZuoJzGveH3dZqSeAh80EA3xtZZ%2FARupPv0DR8WJMBlbv6qhgh946irCtnLuALvge5578n3QF%2BG12U2nJCXsTe3zcdctmIdJdt%2BAfaT%2B2Y2zzQSdIA96vlJuuD3GaDqTEWAelqrc2Vwc2nwnCCxKs5VFvdZrTB7MEqB540IXRozC2M1fU8%2FyWrqdxT1bGSFfPGFlkP0S6Xsa9YMv0x5isyPFgkHU3xgpdoONRsNW3kK6xczlMNnRee4KZFbuWB2hPg%2FY6ERj21C4jerUGHdZMQOGN1X6Cjk0kHvLVSN3mLIIureb7fXzaY65WrjPK0hdP0atH6mmDTl5%2FPIHo0m6dk6hU16zelwudh0arXP%2Fc5WrhFrucNg250ylSp86EmQY7T52eZsBVlGPLY3eR%2BTVVIG8IlaxdOElYB2Ygcns2EL9ERGv6EhrEn%2Bi4%2FJQ%2BqeYYLbapQnHxw7vTIlKsWwXYsO4BUNoF8io0Q52GEh42%2FsvyV1V0n4os1u2c%2BCp19%2BBe1H39Gx27MesAO6LPb%2FSVlypKtrv87PUUa9BCXq3FyYDd%2BFqTJDukhlQtDapgP51lYGdJecAzoWpzuf47iD2dQYP%2B3tf33GvqPn8euVMPD67sHZtjMUJ09%2B%2BbPS%2Bh9LotyuAqmpGpBnuXLEoCy8ySvODV2fltJ9IQ44%2BlWrIXbxsapuZuUVb9DokW27QiYSebgTqpZuttSyPMQVG0gWGxMf%2FGA%2FjqBYQMbLFL9uBC26mfcbEqQyqpDAusa9fgFY5ay1HtdR%2FSgXvggCWN4Asb33f0yBhWKR1SlljDzhAWd4csdxDBygi7fzm31MRwZl6A8nYGbWjK2MgpTvp733MPEKJYN%2BT7HaEk7bXO1MYm6bpZHenEa7%2BoJklLQ9%2FEXuG77%2BwVSRhNFiI%2Bvino5KnnY5aSokly9ILQKm35I1qjW0wBqeLrD%2BjSTjurJnUPHC0DNURAWyA%2F5MoXFOl%2Fom2Qf1HpfLoPxnFDfGcI%2B4jdSDLDgE5uGtHldklsbo5zily321Lk79LMeP8hza3o7e7Xl%2FHfz787Tr9Fyt3p8ZKKPVKxA3uXwtZRv5E0ftU10l2yj0%2FMZ3%2F1nuWhW2XrwwaXzpZEG0ieayLFAwSGegVGbWBAO5WCRFX6xAWwLymgrWj0LVwlno%2BZFIn4L3SIJry5ra3wCM5EEsl5c6ND7iu9mtTE7b57aDA2uH23cruk5pNFbAmsUtvufdc4lHtdjne4UN6N0BpI1jpC8r5O7gcyRkBE60qZkUyu2221Ng8nDjsAQ3bDyZbLcLIsEE6Q30dvD0u9bjYwLCxVYsQ4ESOw80WRTc75Im8Wv8yekwKd3Vri%2BWGumN3wr0JggoI4jJ5DrAXFeOIuyXL6Z4E%2Bzue6E31QimuL6LuQkrZGNBnghQWvs%2FrUZLU8mrH7ipzJAk1eqCu3S5vwNBgp5f04BwKjfpfxOjMcAZWFMFGlPoG0e8%2FqHELwWW%2FqJsjxfHeOzgdH2kCNQblXJ0mxkk2qiaNiJZvcHY4yHlYDqSsvyd5u1kSSj%2BLF%2BQBJN4uld4MB0oB2DBo%2BkOr6KORenRRZN%2FcDab2iMFoiNwq8YE7ysM9IMXH7kkoSH9Ha5Dx2BaZ%2BN0Q8Bkyy0Jd1l62pUbkSr9ovmGo5HPAXpKads0KRf0YwMkoV4QqAIr1bFPVbY9ndKgXNoKjuJCnNte4NRfwsaY4Ch1Y4ukk7SwCc6IgzVFEx7c1%2FZgEn%2BL%2BOtxmFmXohq4I3E2CqPQ3q1TG3pxw5rUsCqiIIAL7QZKU0pd5YvfNFEXt8EMWCp51QrJHK1Aj0ymEGDQSbAqilJkpVYK5U2NRNbEUjZQXf6XN%2FqpVoTxsR5eIlxaw9Ipoog4RHxICM%2FeHLXmB1JfjCU0OXp%2FGUL0gjO3I7M%2FIXhT7LFB2kvaLJoqFu7RW7ABgV3qfB3G5D1I29UrEt%2BtFSNN2jiMrPbFVXulnRD291Q1pvr0KVT%2FQorZ26WQ6wsPNrpyL1p98dsUmRWjfMrPRqzvZbr9XdZlQd8%2FTU2Ty99dB6rW10gomMrPZvV7mWImV287E3qMzMaqq8ay%2FZuQUKB6nordJuw%2BDOPVAtR2tqXVXOD%2BP9ye26HrvOajn2djOHiXCdkLWYWD1j2Z4iZ4Mw8t0lu2BwwNE3QMmkkQZlBAGL6bSHnLOZRDaIgNpFG2ojWg5QY6XsMEMu8TftGZfDerK%2BzCqOdqo%2FWd97QzvqDyhP8fz5DpgziOMLBojwXOFHRkAZnPhuW7uBedsL3eQduJ7%2FFmHS09uOXo%2BPOYu9MIjZQJDcYIr%2Fn%2FI%2FrnjUnoqjRr%2F1i0CK7IOJm6CA5C9sxy3xp85JUHZ%2B3PZ07fQ1j56DOnUOUMtiZHn%2FgQnYoWZbMNOu4bADtHBtO9Tu0w5Vd%2BfXNhQO4xIJhxUNU7loGJCi0WksTBtQ8u3wYQaU7sNU7TWLPetmDmZY0UXuJHGydwzRGaLqumjJB2ZpkD0bjA6tP21AObaDx0sm3arx0mtaoN6vDMxxtH4hz%2FB5qveyDJmiZuIgc6mydLWd0%2FTyDXqVW1srOwLUDub1Gdl%2FLpkOrtHSvUy%2FhioPwD%2BQ6g1f2GuoUuNTvTO8zDy0nMbnhxY4ONQ9WiqqjY%2F2szzTDQt8kutBHD7s0H1n%2BXQoDpdr4sRKj7%2BvUfThzNbB5JoOwpcClEo9VRUUtFBCiNVWKYCuXKVsfSlb16FzspV1mizgHTprfJiinG2SZqtkO7SRSTZRIy2WMB7lwKw%2F0PP2SLA0sksavsrKLhyNyClLEcb25hqyMPuDLliSML4bqARXCtjTwJWWd4ltozX4XSeu9eGXbbBbDb9%2BJ658sC9VRCgWHh4JQOzRV%2BeVjs%2FcYrNbRZlEnhvMl4XLk9zlpYjTGaBNx7g6wEiy2uLMdYWrA9BWd0qh95pQrvNTCowfEnEQ%2F0j%2FHSJC8MDfOaMwVBleSEeF0SK3t7giP6mghEXvK4yD4dFVV6RalIS24DmKjPgwCsl0aetUwh%2B6%2BC2cInLF%2FwE%3D"></iframe>



## 网络Network

### NeRF

同NeRF网络
![Pasted image 20221206180113.png|600](https://raw.githubusercontent.com/yq010105/Blog_images/main/Pasted%20image%2020221206180113.png)


### SDFNetwork

激活函数 $\text{Softplus}(x) = \frac{\log(1 + e^{\beta x})}{\beta}$

网络结构：
![SDFNetwork.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/SDFNetwork.png)

input: pts, 采样点的三维坐标 batch_size * n_samples x 3
output: 257个数 batch_size * n_samples x 257

`sdf(pts) = output[:, :1]`:  batch_size * n_samples x 1，采样点的sdf值

### RenderingNetwork

![RenderingNetwork.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/RenderingNetwork.png)

input: rendering_input :`[batch_size * n_samples ,  3 + 27 + 3+ 256 = 289]`
`rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)`
- pts: batch_size * n_samples, 3
- gradients: batch_size * n_samples, 3
- dirs: batch_size * n_samples, 3
    - 位置编码 to view_dirs: batch_size * n_samples , 27
- feature_vector: batch_size * n_samples, 256

output: sampled_color采样点的RGB颜色 batch_size * n_samples , 3

### SingleVarianceNetwork

```
class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val):
        super(SingleVarianceNetwork, self).__init__()
        # variance 模型可以跟踪和优化这个参数，使其在训练过程中进行更新
        self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))

    def forward(self, x):
        # torch.zeros([1, 3])
        # 大小为 [len(x), 1] 的张量，每个元素都是 exp(variance * 10.0)
        return torch.ones([len(x), 1]) * torch.exp(self.variance * 10.0)

in Runner:
self.deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)
```

render中
`inv_s = deviation_network(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6) `

## dataset

- 相机内外参数矩阵
- 光线的生成以及坐标变换

BlendedMVS/bmvs_bear/cameras_sphere

```
"""
(4, 4) world_mats_np0
[[-1.0889766e+02  3.2340955e+02  6.2724188e+02 -1.6156446e+04] 
[-4.8021997e+02 -3.6971255e+02  2.8318774e+02 -8.9503633e+03]
[ 2.4123600e-01 -4.2752099e-01  8.7122399e-01 -2.1731400e+01]
[ 0.0000000e+00  0.0000000e+00  0.0000000e+00  1.0000000e+00]]
(4, 4) scale_mats_np0
[[ 1.6737139  0.         0.        -2.702419 ]
[ 0.         1.6737139  0.        -1.3968586]
[ 0.         0.         1.6737139 27.347609 ]
[ 0.         0.         0.         1.       ]]
"""

P = world_mat @ scale_mat
"""
[[-1.8226353e+02  5.4129504e+02  1.0498235e+03  8.3964941e+02]
 [-8.0375085e+02 -6.1879303e+02  4.7397528e+02  6.0833594e+02]
 [ 4.0376005e-01 -7.1554786e-01  1.4581797e+00  2.0397587e+00]
 [ 0.0000000e+00  0.0000000e+00  0.0000000e+00  1.0000000e+00]]

[[-1.8226353e+02  5.4129504e+02  1.0498235e+03  8.3964941e+02]
 [-8.0375085e+02 -6.1879303e+02  4.7397528e+02  6.0833594e+02]
 [ 4.0376005e-01 -7.1554786e-01  1.4581797e+00  2.0397587e+00]]
 """
P = P[:3, :4]
```

将P分解为相机内参和外参矩阵，in dataset.py

```
out = cv.decomposeProjectionMatrix(P)
K = out[0] # 3x3
[[1.00980786e+03 1.61999036e-04 6.39247803e+02]
 [0.00000000e+00 1.00980774e+03 4.83591949e+02]
 [0.00000000e+00 0.00000000e+00 1.67371416e+00]]
 
R = out[1] # 3x3
[[-0.33320493  0.8066752   0.48810825]
 [-0.9114712  -0.40804535  0.05214698]
 [ 0.24123597 -0.42752096  0.87122387]]

t = out[2] # 4x1
[[-0.16280915]
 [ 0.30441687]
 [-0.69216055]
 [ 0.6338275 ]]
 
K = K / K[2, 2]
[[6.0333350e+02 9.6790143e-05 3.8193369e+02]
 [0.0000000e+00 6.0333344e+02 2.8893341e+02]
 [0.0000000e+00 0.0000000e+00 1.0000000e+00]]

intrinsics = np.eye(4)
intrinsics[:3, :3] = K # intrinsics: 4x4 为相机内参矩阵
[[6.03333496e+02 9.67901433e-05 3.81933685e+02 0.00000000e+00]
 [0.00000000e+00 6.03333435e+02 2.88933411e+02 0.00000000e+00]
 [0.00000000e+00 0.00000000e+00 1.00000000e+00 0.00000000e+00]
 [0.00000000e+00 0.00000000e+00 0.00000000e+00 1.00000000e+00]]

pose = np.eye(4, dtype=np.float32)
pose[:3, :3] = R.transpose()
pose[:3, 3] = (t[:3] / t[3])[:, 0] # pose: 4x4 为相机外参矩阵
[[-0.33320493 -0.9114712   0.24123597 -0.25686666]
 [ 0.8066752  -0.40804535 -0.42752096  0.48028347]
 [ 0.48810825  0.05214698  0.87122387 -1.092033  ]
 [ 0.          0.          0.          1.        ]]

世界坐标系下，光线的原点：
[[-0.25686666]
 [ 0.48028347]
 [-1.092033  ]
 [ 1.        ]]
```

### 光线生成(随机)
然后生成光线，in `dataset.py/gen_random_rays_at` by img_idx ，batch_size, 并将rays的像素坐标转换到世界坐标系下

p_pixel --> p_camera --> p_world
`intrinsics @ p_pixel`:  `3x3 @ 3x1`
`pose @ p_camera`:  `3x3 @ 3x1`

```
def gen_random_rays_at(self, img_idx, batch_size):
    """
    Generate random rays at world space from one camera.
    """
    pixels_x = torch.randint(low=0, high=self.W, size=[batch_size]) 
    pixels_y = torch.randint(low=0, high=self.H, size=[batch_size])
    color = self.images[img_idx][(pixels_y, pixels_x)]    # batch_size, 3
    mask = self.masks[img_idx][(pixels_y, pixels_x)]      # batch_size, 3
    # p : 像素坐标系下的坐标
    p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float()  # batch_size, 3
    # 将p转换到相机坐标系下
    # matmul : [1, 3, 3] x [batch_size, 3, 1] -> [batch_size, 3, 1] -> [batch_size, 3]
    p = torch.matmul(self.intrinsics_all_inv[img_idx, None, :3, :3], p[:, :, None]).squeeze() # batch_size, 3
    # rays_v ：将p归一化
    rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)    # batch_size, 3
    # rays_v ：将p转换到世界坐标系下
    # matmul : [1, 3, 3] x [batch_size, 3, 1] -> [batch_size, 3, 1] -> [batch_size, 3]
    rays_v = torch.matmul(self.pose_all[img_idx, None, :3, :3], rays_v[:, :, None]).squeeze()  # batch_size, 3
    # [1,3].expand([batch_size, 3])
    rays_o = self.pose_all[img_idx, None, :3, 3].expand(rays_v.shape) # batch_size, 3
    return torch.cat([rays_o.cpu(), rays_v.cpu(), color, mask[:, :1]], dim=-1).cuda()    # batch_size, 10
```

### 计算near和far(from o,d)
根据rays_o 和rays_d 计算出near和far两个平面
```
def near_far_from_sphere(self, rays_o, rays_d):
    a = torch.sum(rays_d**2, dim=-1, keepdim=True)
    b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
    mid = 0.5 * (-b) / a
    # rays_o 在 rays_d 方向上的投影 / rays_d 在 rays_d 方向上的投影
    near = mid - 1.0
    far = mid + 1.0
    return near, far
```

### box的min和max(to生成mesh模型)

```
'''
(4, 4) scale_mats_np0
[[ 1.6737139  0.         0.        -2.702419 ]
[ 0.         1.6737139  0.        -1.3968586]
[ 0.         0.         1.6737139 27.347609 ]
[ 0.         0.         0.         1.       ]]
'''
object_bbox_min = np.array([-1.01, -1.01, -1.01, 1.0])
object_bbox_max = np.array([ 1.01,  1.01,  1.01, 1.0])
# Object scale mat: region of interest to **extract mesh**
object_scale_mat = np.load(os.path.join(self.data_dir, self.object_cameras_name))['scale_mat_0'] # 4x4

# object_bbox_? > object_scale_mat缩放+平移 > scale_mat缩放+平移
object_bbox_min = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_min[:, None] # 4x1
object_bbox_max = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_max[:, None] # 4x1
self.object_bbox_min = object_bbox_min[:3, 0] # 3
self.object_bbox_max = object_bbox_max[:3, 0] # 3
```

## render

### validate_mesh
根据一个$resolution^3$ 的sdf场，生成mesh的ply文件

#### extract_geometry

**extract_fields**
input:
- bound_min : 3 ; bound_max : 3 ; resolution : 64 
- query_func : pts -> sdf

output: u  
u : resolution x resolution x resolution, 为box 中每个点的sdf值

**extract_geometry**

根据体积数据和阈值重建出表面

>[pmneila/PyMCubes: Marching cubes (and related tools) for Python (github.com)](https://github.com/pmneila/PyMCubes)


input:
- bound_min, bound_max, resolution, 
- threshold, 用于`vertices, triangles = mcubes.marching_cubes(u, threshold)`，在等threshold面上，生成mesh的v和t
- query_func，根据位置pts利用network计算出sdf
    - query_func=lambda pts: -self.sdf_network.sdf(pts)
output:
- vertices：三角形网格点
    - N_v , 3: 3为点的三维坐标
- triangles：三角形网格
    -  N_t , 3: 3为三角形网格顶点的索引index

根据v和t，`mesh = trimesh.Trimesh(vertices, triangles)`生成mesh，并导出ply：
```
mesh.export(os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}.ply'.format(self.iter_step)))
```

## 数据集自定义

### imgs2poses.py

是否使用过colmap：
- 如果已经使用colmap生成了`sparse/0/`下的` ['cameras', 'images', 'points3D']`文件，将获得sparse_points.ply
- 若没有，则使用`run_colmap()`，即可生成sparse/0/下文件

#### run_colmap()

```
def run_colmap(basedir, match_type):
    
    logfile_name = os.path.join(basedir, 'colmap_output.txt')
    logfile = open(logfile_name, 'w')
    
    feature_extractor_args = [
        'colmap', 'feature_extractor', 
            '--database_path', os.path.join(basedir, 'database.db'), 
            '--image_path', os.path.join(basedir, 'images'),
            '--ImageReader.single_camera', '1',
            # '--SiftExtraction.use_gpu', '0',
    ]
    # subprocess.check_output: 运行命令行程序，等待程序运行完成，然后返回输出结果
    feat_output = ( subprocess.check_output(feature_extractor_args, universal_newlines=True) )
    logfile.write(feat_output)
    print('Features extracted')

    exhaustive_matcher_args = [
        'colmap', match_type, 
            '--database_path', os.path.join(basedir, 'database.db'), 
    ]

    match_output = ( subprocess.check_output(exhaustive_matcher_args, universal_newlines=True) )
    logfile.write(match_output)
    print('Features matched')
    
    p = os.path.join(basedir, 'sparse')
    if not os.path.exists(p):
        os.makedirs(p)

    # mapper_args = [
    #     'colmap', 'mapper', 
    #         '--database_path', os.path.join(basedir, 'database.db'), 
    #         '--image_path', os.path.join(basedir, 'images'),
    #         '--output_path', os.path.join(basedir, 'sparse'),
    #         '--Mapper.num_threads', '16',
    #         '--Mapper.init_min_tri_angle', '4',
    # ]
    mapper_args = [
        'colmap', 'mapper',
            '--database_path', os.path.join(basedir, 'database.db'),
            '--image_path', os.path.join(basedir, 'images'),
            '--output_path', os.path.join(basedir, 'sparse'), # --export_path changed to --output_path in colmap 3.6
            '--Mapper.num_threads', '16',
            '--Mapper.init_min_tri_angle', '4',
            '--Mapper.multiple_models', '0',
            '--Mapper.extract_colors', '0',
    ]

    map_output = ( subprocess.check_output(mapper_args, universal_newlines=True) )
    logfile.write(map_output)
    logfile.close()
    print('Sparse map created')
    
    print( 'Finished running COLMAP, see {} for logs'.format(logfile_name) )
```

上述代码相当于分别运行:
```
colmap feature_extractor --database_path os.path.join(basedir, 'database.db') --image_path os.path.join(basedir, 'images') --ImageReader.single_camera 1
colmap match_type --database_path os.path.join(basedir, 'database.db')
match_type : exhaustive_matcher Or sequential_matcher
colmap mapper --database_path os.path.join(basedir, 'database.db') --image_path os.path.join(basedir, 'images') --output_path os.path.join(basedir, 'sparse') --Mapper.num_threads 16 --Mapper.init_min_tri_angle 4 --Mapper.multiple_models 0 --Mapper.extract_colors 0
```

- feature_extractor: Perform **feature extraction or import features** for a set of images.
- exhaustive_matcher: Perform **feature matching** after performing feature extraction.
- mapper: **Sparse 3D reconstruction / mapping of the dataset** using SfM after performing feature extraction and matching.

然后将命令行的输出结果保存到logfile即`basedir/colmap_output.txt`中

> colmap命令行：[Command-line Interface — COLMAP 3.8-dev documentation](https://colmap.github.io/cli.html)
> dense中深度图转换：[COLMAP简明教程 导入指定参数 命令行 导出深度图 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/584386128)

### load_colmap_data() to colmap_read_model.py
`python .\colmap_read_model.py E:\BaiduSyncdisk\NeRF_Proj\NeuS\video2bmvs\M590\sparse\0 .bin`

读取`['cameras', 'images', 'points3D']`文件的数据

input:
- basedir
output: 
- poses, shape: 3 x 5 x num_images, include c2w: 3x4xn and hwf: 3x1xn
- pts3d, 一个长度为num_points字典，key为point3D_id，value为Point3D对象
- perm, # 按照name排序，返回排序后的索引的列表：`[from 0 to num_images-1]`

#### cameras images and pts3d be like: 

| var     | example |
| ------- | ------- |
| cameras |    `{1: Camera(id=1, model='SIMPLE_RADIAL', width=960, height=544, params=array([ 5.07683492e+02,  4.80000000e+02,  2.72000000e+02, -5.37403479e-03])), ...}`     |
| images  |     `{1: Image(id=1, qvec=array([ 0.8999159 , -0.29030237,  0.07162026,  0.31740581]), tvec=array([ 0.29762954, -2.81576928,  1.41888716]), camera_id=1, name='000.png', xys=xys, point3D_ids=point3D_ids, ...}`    |
| pts3D        |   `{1054: Point3D(id=1054, xyz=array([1.03491375, 1.65809594, 3.83718124]), rgb=array([147, 146, 137]), error=array(0.57352093), image_ids=array([115, 116, 117, 114, 113, 112]), point2D_idxs=array([998, 822, 912, 977, 889, 817])), ...}`      |

xys and point3D_ids in images be like:

```
xys=array([[ 83.70032501,   2.57579875],
       [ 83.70032501,   2.57579875],
       [469.29092407,   2.57086968],
       ...,
       [759.08764648, 164.65560913],
       [533.28503418, 297.13980103],
       [837.11437988, 342.07727051]]), 
point3D_ids=array([  -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,       
 -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
 -1,   -1,   -1,   -1, 9109,   -1,   -1,   -1,   -1,   -1,   -1,
 -1,   -1,   -1,   -1,   -1, 8781,   -1,   -1, 8628,   -1,   -1,
 -1, 2059,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
 -1,   -1,   -1,   -1, 8791,   -1,   -1, 8683,   -1, 8387,   -1,
 -1,   -1,   -1,   -1,   -1, 9008, 9007,   -1, 9161, 8786,   -1,
 -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
 -1,   -1,   -1,   -1,   -1,   -1,   -1, 9175,   -1,   -1,   -1,
9053,   -1,   -1,   -1,   -1, 8756,   -1,   -1,   -1,   -1,   -1,
 -1, 9024,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1, 9111,
 -1,   -1, 9018,   -1, 9004,   -1,   -1,   -1,   -1,   -1,   -1,
 -1,   -1,   -1,   -1,   -1, 8992,   -1,   -1,   -1,   -1,   -1,
4701,   -1, 9067,   -1, 9166, 3880,   -1,   -1,   -1,   -1,   -1,
 -1,   -1,   -1,   -1,   -1,   -1,   -1, 8725,   -1, 9112,   -1,
 -1,   -1,   -1, 8990,   -1, 8793, 9118, 8847, 9009, 9140, 9012,
 -1,   -1,   -1, 7743, 9065, 8604, 3935,   -1,   -1,   -1,   -1,
9075,   -1,   -1, 8966,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
 -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
 -1,   -1,   -1,   -1,   -1,   19,   -1,   -1,   -1,   -1,   -1,
9017,   -1,   -1,   -1, 9020,   -1, 9005,   -1,   -1,   -1,   -1,
 -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1, 8696,
 -1,   -1, 8930,   -1,   -1, 8970,   -1,   -1,   -1,   -1, 9076,
 -1, 9114, 8925,   -1, 8915,   -1, 9077, 8851, 8655, 5885, 4073,
 -1, 3839,   -1,   -1,   -1,   -1, 9165, 9078,   -1,   -1,   -1,
 -1,   -1,   -1,   -1, 9055,   -1,   -1,   -1,   -1,   -1,   -1,
 -1,   -1,   -1,   -1, 9017,   -1,   -1,   -1,   -1,   -1,   -1,
 -1, 8682,   -1,   -1, 9170,   -1, 7562, 7556,   -1,   -1,   -1,
 -1,   -1,   -1,   -1,   -1, 8962, 9079,   -1,   -1,   -1, 8586,
8224,   -1,   -1,   -1,   -1, 1399, 9168, 6439, 9121, 8255, 9169,
 -1, 9151, 8971, 4698, 9171, 9172,   -1,   -1, 8898, 3916,   -1,
 -1,   -1, 1788,   -1,   -1,   -1, 9080,   -1,   -1,   -1,   -1,
 -1,   -1, 2097,   -1, 4103,   -1,   -1,   -1,   -1, 2073,   -1,
 -1, 1771,   -1,   -1,   -1,   -1,   -1,   -1, 8813,   -1, 9030,
 -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1, 8841, 9081,
 -1,   -1,   -1, 8977,   -1, 8372, 9057, 6807, 9082, 5941, 4181,
1675,   -1, 1683,   -1,   -1, 1503, 9083, 1973, 9071, 2679, 2412,
3238,   -1, 9164, 1796, 9174,   -1,   -1,   -1,   -1,   -1,   -1,
9042, 9084,   -1,   -1,   -1,   -1,   -1, 9051, 9050,   -1, 9085,
 -1, 9158, 9086,  853, 7671, 9128,   -1,   -1, 9058,   -1, 9087,
 -1, 8502, 9102,   -1, 9106,   -1, 9039,   -1,   -1,   -1, 9069,
 -1, 2261,   -1, 1793, 2643,   -1,   -1, 8810, 8945,   -1,   -1,
 -1,   -1,   -1, 9043,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
9142,   -1,   -1, 9122, 9089, 9090, 8863, 9103, 2161, 2446,   -1,
 -1,   -1,   -1,   -1, 9104,   -1, 9060, 9131,   -1,   -1,   -1,
 -1, 8980, 8706,   -1, 9105, 9091, 9173,   -1,   -1, 2996,   -1,
 -1, 9092,   -1,   -1,   -1,   -1, 9094, 9095, 9096, 9097, 9156,
 -1,   -1,   -1,   -1, 8772, 8818,   -1,   -1, 9162, 9062, 9098,
 -1,   -1, 8907, 9099, 8985, 4624,   -1, 3746, 8951,   -1,   -1,
8908,   -1, 9135, 8986, 9101,   -1,   -1,   -1, 9137,   -1]))}
```

#### cameras文件

input:
- path_to_model_file, `camerasfile = os.path.join(realdir, 'sparse/0/cameras.bin')`
output:
- cameras，一个长度为num_cameras字典，key为camera_id，value为Camera对象

使用:
```
camerasfile = os.path.join(realdir, 'sparse/0/cameras.bin')
camdata = read_model.read_cameras_binary(camerasfile)

list_of_keys = list(camdata.keys()) # list from 1 to num_cameras
cam = camdata[list_of_keys[0]] # Camera(id=1, model='SIMPLE_RADIAL', width=960, height=544, params=array([ 5.07683492e+02,  4.80000000e+02,  2.72000000e+02, -5.37403479e-03]))
print( 'Cameras', len(cam)) # Cameras 5

h, w, f = cam.height, cam.width, cam.params[0]
hwf = np.array([h,w,f]).reshape([3,1])
```

#### images文件

input:
- path_to_model_file,`imagesfile = os.path.join(realdir, 'sparse/0/images.bin')`
output:
- images，一个长度为num_reg_images字典，key为image_id，value为Image对象

使用:
```
imagesfile = os.path.join(realdir, 'sparse/0/images.bin')
imdata = read_model.read_images_binary(imagesfile)

w2c_mats = []
bottom = np.array([0,0,0,1.]).reshape([1,4])

names = [imdata[k].name for k in imdata] # 一个长度为num_images的list，每个元素为图片的名字
print( 'Images #', len(names)) 
perm = np.argsort(names) # 按照name排序，返回排序后的索引的列表：[from 0 to num_images-1]
for k in imdata:
    im = imdata[k]
    R = im.qvec2rotmat() # 将旋转向量转换成旋转矩阵 3x3
    t = im.tvec.reshape([3,1]) # 平移向量 3x1
    m = np.concatenate([np.concatenate([R, t], 1), bottom], 0) # 4x4
    w2c_mats.append(m) # 一个长度为num_images的list，每个元素为4x4的矩阵

w2c_mats = np.stack(w2c_mats, 0) # num_images x 4 x 4
c2w_mats = np.linalg.inv(w2c_mats) # num_images x 4 x 4

poses = c2w_mats[:, :3, :4].transpose([1,2,0]) # 3 x 4 x num_images
poses = np.concatenate([poses, np.tile(hwf[..., np.newaxis], [1,1,poses.shape[-1]])], 1)
# tile : 将hwf扩展成3 x 1 x 1 ，然后tile成3 x 1 x num_images，tile表示在某个维度上重复多少次
# poses : 3 x 5 x num_images ，c2w：3 x 4 x num_images and hwf: 3 x 1 x num_images

# must switch to [-u, r, -t] from [r, -u, t], NOT [r, u, -t]
poses = np.concatenate([poses[:, 1:2, :], poses[:, 0:1, :], -poses[:, 2:3, :], poses[:, 3:4, :], poses[:, 4:5, :]], 1)
```

其中`R = im.qvec2rotmat()`将旋转向量转换成旋转矩阵:

如果给定旋转向量为 [qw, qx, qy, qz]，其中 qw 是标量部分，qx, qy, qz 是向量部分，可以通过以下步骤将旋转向量转换为旋转矩阵：

构造单位四元数 q：
```css
q = qw + qx * i + qy * j + qz * k  其中 i, j, k 是虚部的基本单位向量。
```

计算旋转矩阵 R(w2c)：
``` perl
R = | 1 - 2*(qy^2 + qz^2)   2*(qx*qy - qw*qz)   2*(qx*qz + qw*qy) |
    | 2*(qx*qy + qw*qz)     1 - 2*(qx^2 + qz^2) 2*(qy*qz - qw*qx) |
    | 2*(qx*qz - qw*qy)     2*(qy*qz + qw*qx)   1 - 2*(qx^2 + qy^2) |
```

```python
def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])
```

#### points3D文件

input:
- path_to_model_file: `points3dfile = os.path.join(realdir, 'sparse/0/points3D.bin')`
output:
- pts3D, 一个长度为num_points字典，key为point3D_id，value为Point3D对象

```
points3dfile = os.path.join(realdir, 'sparse/0/points3D.bin')
pts3d = read_model.read_points3d_binary(points3dfile)
```

# 实验

## Dataset：DTU&BlendedMVS

DTU: 15个场景、1600 × 1200像素
BlendedMVS：7个场景、768 × 576像素

## Baseline

表面重建方法：
- IDR：高质量表面重建，但需要前景mask监督
- DVR，没有比较
体积渲染方法：
- NeRF：我们使用 25 的threshold 从学习的密度场中提取网格（在补充材料中验证了为什么用25做阈值）
广泛使用的经典MVS方法：（MVS：Multi-View Stereo Reconstruction多视点立体重建）
- colmap：从colmap的输出点云中，使用Screened Poisson Surface Reconstruction重建mesh

UNISURF：将表面渲染与以占用场做场景表示的体积渲染统一起来

## 实现细节

设感兴趣的物体在单位球体内，每批采样512条光线，单个RTX2080Ti：14h（with mask），16h（without mask）
对于w/o mask ，使用NeRF++对背景进行建模，网络架构和初始化方案与IDR相似

## 环境配置：
autodl镜像：
    Miniconda  conda3
    Python  3.8(ubuntu20.04)
    Cuda  11.8
`conda create -n neus python=3.8`
`pip install -r requirements.txt`
`pip --default-timeout=1000 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html`

## 命令行运行程序：
`python exp_runner.py --mode train --conf ./confs/wmask.conf --case <case_name>` # 带mask监督
`python exp_runner.py --mode train --conf ./confs/womask.conf --case <case_name>` # 不带mask监督
case_name是所选物体数据集文件夹的名称
如果训练过程中中断，可以设置`--is_continue` 来进行加载最后的ckpt继续训练：
`python exp_runner.py --mode train --conf ./confs/wmask.conf --case <case_name> --is_continue`

训练生成的结果会保存在根目录下exp文件夹下，可以看到meshes文件夹中保存了训练过程中的mesh模型，但面片较少。需要比较精细的mesh模型需要运行
`python exp_runner.py --mode validate_mesh --conf <config_file> --case <case_name> --is_continue # use latest checkpoint`
多视角渲染
`python exp_runner.py --mode interpolate_<img_idx_0>_<img_idx_1> --conf <config_file> --case <case_name> --is_continue # use latest checkpoint`


### eg1: clock_wmask
```
python exp_runner.py --mode train --conf ./confs/wmask.conf --case bmvs_clock #带mask
中断后继续训练
python exp_runner.py --mode train --conf ./confs/wmask.conf --case bmvs_clock --is_continue
将mesh模型精细化
python exp_runner.py --mode validate_mesh --conf ./confs/wmask.conf --case bmvs_clock --is_continue
插值多视角渲染成mp4(0~1之间新视角生成)
python exp_runner.py --mode interpolate_000_001 --conf ./confs/wmask.conf --case bmvs_clock --is_continue
```
![Pasted image 20230601130943.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/Pasted%20image%2020230601130943.png)


除了主体模型外还有一些噪音
![Pasted image 20230601131053.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/Pasted%20image%2020230601131053.png)


Neus： (结果与resolution无关)
- 使用Neus时，精细化模型参数设置为 `resolution=512` 可能与此有关
- 改为`resolution=1024` 运行一下validate_mesh
![Pasted image 20230602165940.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/Pasted%20image%2020230602165940.png)



虽然运行后，面数更多了，但是细节处依然不清楚，可能需要继续增加训练的步数
- 看代码后：resolution可以增加object的分辨率，再缩放到物体实际大小
    - 可以看到vertices和faces都增加了，但由于sdf生成的相同，因此表面变得更精细，但细节处不清楚的地方依然不清楚

### eg2: bear_womask
```
python exp_runner.py --mode train --conf ./confs/womask.conf --case bmvs_bear  #没有mask
python exp_runner.py --mode train --conf ./confs/womask.conf --case bmvs_bear  --is_continue
```

![Pasted image 20230601211543.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/Pasted%20image%2020230601211543.png)

![Pasted image 20230601211823.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/Pasted%20image%2020230601211823.png)


运行后，mesh模型的面很少
```
将mesh模型精细化
python exp_runner.py --mode validate_mesh --conf ./confs/womask.conf --case bmvs_bear --is_continue
```
![Pasted image 20230601211901.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/Pasted%20image%2020230601211901.png)

![Pasted image 20230601211955.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/Pasted%20image%2020230601211955.png)



### Neus如何给模型加纹理：
>[How to reconstruct texture after generating mesh ? · Issue #48 · Totoro97/NeuS (github.com)](https://github.com/Totoro97/NeuS/issues/48)


## 渲染自定义视角的视频

根据中间两个img进行插值，中间插入生成的新视图图片

```
python exp_runner.py --mode interpolate_<img_idx_0>_<img_idx_1> --conf <config_file> --case <case_name> --is_continue # use latest checkpoint

eg:
python exp_runner.py --mode train --conf ./confs/womask.conf --case bmvs_bear
python exp_runner.py --mode interpolate_1_2 --conf ./confs/womask.conf --case bmvs_bear --is_continue # use latest checkpoint
```

eg: 1 to 2
![00300000_1_2.gif](https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/00300000_1_2.gif)

<div style="display: flex; justify-content: center;"> <img src="https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/001.png" alt="Image 1" style="width: 50%; height: auto; margin: 10px;"> to <img src="https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/002.png" alt="Image 2" style="width: 50%; height: auto; margin: 10px;"> </div>



# 自定义数据集
自己拍一组照片: **手机或者相机 绕 物体拍一周，每张的角度不要超过30°（保证有overlap区域）**
>[colmap简介及入门级使用 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/397760339)

![Pasted image 20230531192515.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/Pasted%20image%2020230531192515.png)


打开colmap.bat进入gui界面，点击**Reconstruction**，再点击**Automatic reconstruction**

**workspace folder：选择workspace文件夹，注意不支持中文路径**
**Image folder：选择存放多视角图像的数据文件夹，注意不支持中文路径**
**Data type：选择 Individual images**
**Quality：看需要选择，选择High重建花费的时间最长，重建的质量不一定最好；**

在COLMAP中看生成的稀疏点云
![Pasted image 20230531191938.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/Pasted%20image%2020230531191938.png)


在 workspace folder 文件夹->dense->0文件夹下找到 fused.ply数据，用 meshlab中打开可以看到稠密的三维重建的结果。
在Meshlab中查看生成的稠密点云

![Pasted image 20230531192204.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/Pasted%20image%2020230531192204.png)



# Neus使用自制数据集

>[(18条消息) 基于Nerf的三维重建算法Neus初探_Alpha狗蛋的博客-CSDN博客](https://blog.csdn.net/mockbird123/article/details/129934066)

两个文件夹：image和mask，一个文件
image文件夹就是rgb图片数据，算法默认支持png格式。
mask文件夹包含的是模型的前景图像，前景和后景以黑色和白色区分，如果配置文件选择withou mask，其实这个文件夹的数据是没有意义的。但必须有文件，且名称、图像像素要和image的图像一一对应。
最后是cameras_sphere.npz文件，它包括了相机的属性和图像的位姿信息等，这个是需要我们自己计算的。官方给出了两种计算方案，第二种是用colmap计算npz文件。

## 使用Colmap生成npz文件

```
cd colmap_preprocess
python img2poses.py ${data_dir}
```

将会生成：`${data_dir}/sparse_points.ply`，在meshlab中选择多余部分的Vertices，并删除，然后保存为`${data_dir}/sparse_points_interest.ply`.

然后
```
python gen_cameras.py ${data_dir}
```

就会在 ${data_dir}下生成preprocessed，包括image、mask和cameras_sphere.npz
