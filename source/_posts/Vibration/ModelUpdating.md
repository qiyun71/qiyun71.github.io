---
title: Model Updating Review
date: 2023-12-22 15:03:03
tags: 
categories: Vibration
---

**模型修正Model Updating**
*卫星(结构<-->振动)利用频率响应(FR)等数据进行有限元模型结构参数的更新*

<!-- more -->

# 基础知识

基本概念：
- [什么是固有频率？ - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/23320350)
  - 通常一个结构有很多个**固有频率**。固有频率与外界激励没有关系，是结构的一种固有属性。**只与材料的刚度k和质量m有关**：$\omega_{n} =  \sqrt{\frac{k}{m}}$
  - [2.固有频率介绍_固有频率越大越好还是越小越好-CSDN博客](https://blog.csdn.net/qq_39200110/article/details/106057561)
- 模态分析是一种处理过程：根据结构的固有特性，包括固有频率、阻尼比和模态振型，这些动力学属性去描述结构。[结构动力学中的模态分析(1) —— 线性系统和频响函数 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/69229246)
  - 每一个模态都对应一个**固定的振动频率，阻尼比**及**振动形式**。而**固定的振动形式**也被称为**模态振型**[模态是什么](https://www.zhihu.com/question/24578439/answer/837484786)
  - 一个自由度为N的系统，包含N阶模态，通常将固有频率由小到大排列
  - 模态分析方法是通过坐标变换将多自由度系统解耦成为模态空间中的一系列单自由度系统，通过坐标变换可将模态空间中的坐标变换为实际物理坐标得到多自由度系统各个坐标的时域位移解。
  - 我们认为系统某点的响应及频响函数都是全部模态的叠加，即我们所采用的是完整的模态集。但实际上并非所有模态对响应的贡献都是相同的。对低频响应来说，高阶模态的影响较小。对实际结构而言，我们感兴趣的往往是它的前几阶或十几阶模态，更高阶的模态常常被抛弃。[机械振动理论(4)-实验实模态分析 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/82442784)
  - 模态参数识别:是指对振动系统进行激振（即输入），**通过测量获得系统的输入、输出**（或仅仅是输出）信号数据，经过对他们进行处理和分析，依据不同的识别模型和方法，**识别出系统的结构模态参数**（如频率、阻尼比、振型、模态刚度、模态质量等）。这类问题为结构动力学第一类逆问题[模态参数识别及有限元模型修正(seu.edu.cn)](https://seugs.seu.edu.cn/_upload/article/files/e0/39/83ad9ffd4cc098c6a6a3f439177e/7cd241da-840e-437a-9bbd-3a0794d1584c.pdf)
    - 模态参数识别是结构动力学中的反问题，它建立在实验的基础上，基于理论与实验相结合的原则，辨识出系统的模态参数，最终实现对系统的改进。通 过获取结构的动力学特性，对结构性能进行评价，从而判断结构的可靠性及安 全性是否符合要求。
- 模型修正

频谱分析：[从傅里叶变换，到短时傅里叶变换，再到小波分析（CWT），看这一篇就够了（附MATLAB傻瓜式实现代码） - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/589651368)

# 论文

## CNN是什么

[7大类深度CNN架构创新综述 | 机器之心 (jiqizhixin.com)](https://www.jiqizhixin.com/articles/2019-01-25-6)

[你应该知道的几种CNN网络与实现 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/176987177)


other：
[深度学习全连接与cnn区别 (volcengine.com)](https://www.volcengine.com/theme/1216339-S-7-1)
## A robust stochastic model updating method with resampling processing

为了更好地估计参数的不确定性，提出了一种鲁棒随机模型更新框架。在该框架中，为了提高鲁棒性，重采样过程主要设计用于处理不良样本点，特别是有限样本量问题。其次，提出了基于巴塔查里亚距离和欧几里得距离的平均距离不确定度度量，以充分利用测量的可用信息。随后采用粒子群优化算法对结构的输入参数进行更新。最后以质量-弹簧系统和钢板结构为例说明了该方法的有效性和优越性。通过将测量样品加入不良样品，讨论了重采样过程的作用。

- 有限元模型具有一定的不确定性，主要是由于模型的简化、近似以及模型参数的不确定性，如弹性模量、几何尺寸、边界条件、静、动载荷条件等。
- 实验系统的测量都具有一定的不确定度。这种不确定性与难以控制的随机实验效应有关，如制造公差引入的偏差，随后信号处理期间的测量噪声或有限的测量数据。

## FR model based on UCNN

[A frequency response model updating method based on unidirectional convolutional neural network (readpaper.com)](https://readpaper.com/pdf-annotate/note?pdfId=2110304823896008192&noteId=2110305055136380672)
蓝色：Review | 红色：之前方法缺陷 | 绿色：理论知识/改进

本文提出了一种基于单向卷积神经网络 (UCNN) 的方法，以利用频率响应 (FR) 数据进行有限元模型更新。UCNN 旨在在没有任何人工特征提取的情况下从 FR 数据获取高精度逆映射到更新参数。单向卷积分别应用于FR数据的频率和位置维度，以避免数据耦合。UCNN 在卫星模型更新实验中优于基于残差的模型更新方法和二维卷积神经网络。它在训练集中内外都实现了高精度的结果。

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20231226203922.png)

评价指标：
- FR 保证准则(assurance criterion) $FRAC=\frac{|H_e^TH_s|^2}{(H_e^TH_e)(H_s^TH_s)}$ 用于描述试验数据与模拟FR数据的相似度

# 数据集

[Download a Data File | Case School of Engineering | Case Western Reserve University](https://engineering.case.edu/bearingdatacenter/download-data-file)

# 实验

[模态-力锤法,激振器法 (FRF频率响应函数,传递函数,共振频率及阻尼系数,猝发随机激励,力窗/指数窗) - 北京美科环试MeK](https://www.mek.net.cn/DataPhysics_Software_FRF.html)

## 软件

[模态分析软件|模态测试系统|实验模态分析|模态测试软件|模态软件 (hzrad.com)](http://www.hzrad.com/edm-modal-analysis)

# 传统方法Convention

- 有限元方法获取数据集耗费时间长，使用一个代理模型来代替有限元计算模型。
- 根据随机样本$X_{s}$通过代理模型得到模拟响应$Y_{s}$，动力学实验得到实验响应$Y_{e}(f_{1} ... f_{6})$
- 用子区间相似度计算模拟样本$Y_{s}$和试验样本$Y_{e}$之间的值，并作为目标函数。
- 用麻雀搜索算法将目标函数迭代寻优，得到修正后的均值$\mu$和标准差$\sigma$


# 深度学习方法

## plate_UCNN(no use)

```markdown
\data\train\fre1000
- sita1.xlsx #1000x1 弹性模量
- sita2.xlsx #1000x1 钢板厚度
- xiangying.xlsx #198000x10 全是0 # 每99×2行一条

\data\test\fre1000
- sita1.xlsx #1000x1
- sita2.xlsx #1000x1
- xiangying.xlsx #198000x10 全是0
```

## satellite_UCNN(code and images not by author)

### 数据集制作

主体结构6个参数为：只修正三个参数：$\theta_1$、$\theta_3$、$\theta_5$
- **主弹性模量**$\theta_1$ 70Gpa，
- 主密度$\theta_2$  ，密度2.7x$10^{3} kg/m^{3}$
- **中心筒厚度**$\theta_3$ 2mm
- 底板厚度$\theta_4$ 1mm
- **剪切板厚度**$\theta_5$ 2mm
- 顶板厚度$\theta_6$ 2mm
  - 其中$\theta_{4}、\theta_6$为常数，其他为待修正参数，$\theta_2$训练效果很差，直接去掉了，可能造成影响
  - 其他参数：模型高1250mm，中心筒直径400mm，顶板边长600mm，剪切板长1000mm，宽400mm

```markdown
/data/train 训练数据分为25组，每一组400条，共10000次仿真数据
- fre-400-1
  - sita1.xlsx #400x1
  - sita2.xlsx
  - ...
  - sita6.xlsx
  - xiangying.xlsx #80800x11 频响数据，每101×2行为一条，对应0-50Hz 101个频率点x、y两个方向的数据
- fre-400-2
- fre-400-3
- fre-400-4
- ...
- fre-400-25

测量的卫星模型加速度频响数据，将有限元模型的加速度频响数据经预处理转化为对应的频响图像，然后输入进UCNN进行训练
将选取的3个有限元模型参数sita1/3/5的值组成向量作为训练输出结果的GT对照标签

/data/test
- fre-400
  - sita1.xlsx #400x1
  - sita2.xlsx 
  - ...
  - sita6.xlsx
  - xiangying.xlsx #80800x11
```

选取了 11 个节点，并测量了水平X和竖直Y两个方向的加速度频响数据，频率范围为 ==0-30Hz==?，频率间隔为 0.5。得到最终输入网络的频响图像尺寸为 2×11×61，对应标签形状为 3×1

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20231225211854.png)

**数据集生成**
- 先在一定范围内生成大量随机数作为有限元模型的结构参数
  - 弹性模量50.0~90.0GPa
  - 中心筒厚度1.00~3.00mm
  - 剪切板厚度1.00~3.00mm
- 将随机生成的结构参数输入有限元模型，得到水平X和竖直Y方向、11 个节点在61个频率下的加速度响应

FR数据转图：
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20231225212852.png)

### 网络结构

UCNN单向卷积

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20231227133859.png)

### 评价指标

测试集错误率：
$错误率 = \frac{预测值-标签GT}{标签GT}$

## Mine

### 网络结构：

| 网络 | 图片 |
| :--: | ---- |
| UCNN | ![image.png\|333](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20231227133859.png) |
| MLP | ![image.png\|333](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20240111150150.png)<br> |
| MLP_S | ![image.png\|333](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20240111150239.png)<br> |
| VGG16 | ![image.png\|333](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20240111145858.png) |


### 实验记录

```python
20231227-111217: 10, 0.9 UCNN
20231227-111738: 25, 0.99 UCNN
20231227-160116: 25, 0.99 MLP 4x128
20231227-160651: 25, 0.99 UCNN 3.5min
20231227-162720: 25, 0.99 MLP 8x256
20231227-163116: 25, 0.99 MLP 8x256 **400** epoch 4.0min
20240102-164844: 25, 0.99 MLP_S 14个4x256的MLP 200 epoch 9min
20240102-170301: 25, 0.99 MLP_S 14个2x256的MLP 200 epoch 5.6min
20240102-203115: 25, 0.99 MLP_S 14个2x256的MLP **400** epoch 15.1min
20240102-205253: 25, 0.99 MLP_S 14个4x256的MLP **400** epoch 23.27min
20240102-211805: 25, 0.99 UCNN **400** epoch 7min
20240111-160514: 25, 0.99 VGG 200 epoch 
20240111-162059: 25, 0.99 VGG 400 epoch 26.34 min
20240113-190506: 25, 0.99 ResNet50 400 epoch 36.1 min
```

不论MLP还是UCNN，中间sita3预测的误差都很大

```powershell
(satellite) PS D:\0Proj\ModelUpdating\satellite_UCNN> python run.py --test --resume outputs\@20231227-163116_mlp\400_mlp.pth --net mlp   
error_rate=:0.05787282592434471=(0.19222232587635518/3.321460855007172)
=====================
error_rate_each0=:0.04022682424495547=(0.28112122416496277/6.9884021282196045)
error_rate_each1=:0.12927590329299216=(0.25523426607251165/1.9743375182151794)
error_rate_each2=:0.04024536597068831=(0.040311464481055735/1.0016423881053924)

(satellite) PS D:\0Proj\ModelUpdating\SatelliteModelUpdating> python run.py --test --resume  outputs\@20240102-203115_mlp_s\400_mlp_s.pth --net mlp_s
error_rate=:0.03934166444889552=(0.13067179843783377/3.321460855007172)
=====================
error_rate_each0=:0.028448767469817206=(0.1988114271312952/6.9884021282196045)
error_rate_each1=:0.0800526909024898=(0.15805103108286858/1.9743375182151794)
error_rate_each2=:0.03509528007871003=(0.03515292014926672/1.0016423881053924)

(satellite) PS D:\0Proj\ModelUpdating\SatelliteModelUpdating> python run.py --test --resume  outputs\@20240102-205253_mlp_s\400_mlp_s.pth --net mlp_s # 过拟合了，训练集上的损失比较小，但是测试集上的loss比较大
error_rate=:0.03124857323569491=(0.10379091277718544/3.321460855007172)
=====================
error_rate_each0=:0.01964181444371989=(0.13726489786058665/6.9884021282196045)
error_rate_each1=:0.07125288617244709=(0.14067724645137786/1.9743375182151794)
error_rate_each2=:0.03337576539351412=(0.03343058135360479/1.0016423881053924)

(satellite) PS D:\0Proj\ModelUpdating\SatelliteModelUpdating> python .\run.py --test --resume outputs\@20240102-211805_ucnn\400_ucnn.pth
error_rate=:0.02647251309147154=(0.08792741596698761/3.321460855007172)
=====================
error_rate_each0=:0.022187629130279763=(0.15505607463419438/6.9884021282196045)
error_rate_each1=:0.03919995023153533=(0.07739393245428801/1.9743375182151794)
error_rate_each2=:0.031280856565034834=(0.03133223187178373/1.0016423881053924)

(satellite) PS D:\0Proj\ModelUpdating\SatelliteModelUpdating> python .\run.py --net vgg --test --resume outputs\202401\@20240111-162059_vgg\400_vgg.pth
error_rate=:0.02620232199857664=(0.0870299868285656/3.321460855007172)
=====================
error_rate_each0=:0.016490224581949674=(0.11524032056331635/6.9884021282196045)
error_rate_each1=:0.05937809189634055=(0.11723239459097386/1.9743375182151794)
error_rate_each2=:0.02857031368553574=(0.028617237228900194/1.0016423881053924)

(satellite) PS D:\0Proj\ModelUpdating\SatelliteModelUpdating> python .\run.py --test --resume outputs\202401\@20240113-190506_resnet\400_resnet.pth --net resnet
error_rate=:0.12867085358744207=(0.42737520337104795/3.321460855007172)
=====================
error_rate_each0=:0.1063032449177=(0.7428898230195046/6.9884021282196045)
error_rate_each1=:0.22810707873746822=(0.4503603637218475/1.9743375182151794)
error_rate_each2=:0.08872962392607951=(0.08887535240501165/1.0016423881053924)
```

UCNN略好于MLP_S，改进的点在于如何用少量样本即可完成训练