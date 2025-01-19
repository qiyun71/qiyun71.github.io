---
title: Reliability
date: 2024-12-25 19:01:55
tags:
  - Reliability
categories: Other Interest
top: true
---

Reliability：指产品在**规定的条件**下和**规定的时间**内，无差错地完成**规定任务**的概率。

针对产品，依靠数字孪生/CAX技术，使用概率开展研究：
- 实物：实际生产出来的产品，必须了解实验对象（机械/电子/生化...产品），对不同类型产品有哪些故障失效形式
  - 失效概率曲线...
- 设备：实验/测量设备，学会如何开展测试实验（Modal Testing）
- 仿真：建模仿真技术，学会软件操作+脚本编写，懂原理更好 (CAX)
- 数学：概率论<贝叶斯>，机器学习/深度学习，可靠性分析基础理论/方法
- 可靠性分析论文(前沿理论) + 可靠性分析项目(实践操作)

<!-- more -->

>  [专著-北航可靠性与系统工程学院](https://rse.buaa.edu.cn/kxyj1/kycg/zz.htm) Book list


>  [crafe.net/files/可靠性科学方法论-康锐.pdf](http://www.crafe.net/files/%E5%8F%AF%E9%9D%A0%E6%80%A7%E7%A7%91%E5%AD%A6%E6%96%B9%E6%B3%95%E8%AE%BA-%E5%BA%B7%E9%94%90.pdf) 大致了解以下
>  [电子产品可靠性云评估](http://crafe.net/prod.html)  CRAFE 2.0 软件平台简介 电子产品可靠性综合评估软件

可靠性数学方法
- 故障数据的统计分析
- 可靠性的理论话语
- 可靠性统计试验
可靠性物理方法
- 故障率——协变量模型
- 故障时间模型
- 性能裕量模型
可靠性逻辑方法
- 功能逻辑方法——可靠性框图
- 故障逻辑方法——故障树分析
可靠性设计方法
- 电子产品降额设计法
- 热设计
- 电磁兼容设计
- 防振动设计
- 余度设计法


## 系统可靠性设计分析基础

可靠度：产品在规定条件下和规定的时间内，完成规定功能的概率
$R(t)=P(\xi>t)$
- $R(t)$可靠度函数
- $\xi$ 产品故障前的工作时间(h)
- t规定的时间

$R(t) =\frac{N_{0}-r(t)}{N_{0}}$
- $N_{0}$ 在t=0时刻，规定条件下正常工作的产品数
- $r(t)$ 在0~t时刻产品的累计故障数(假设产品故障后不予修复)

累积故障概率(累积故障分布函数)：产品在规定条件下和规定时间内，丧失规定功能的概率
$F(t)=P(\xi\leq t) = \frac{r(t)}{N_{0}}$
显然：$F(t)+R(t) =1$




>  [失效率 - 维基百科，自由的百科全书](https://zh.wikipedia.org/wiki/%E5%A4%B1%E6%95%88%E7%8E%87)


# Others

>  [系统可靠性设计与评估——从理论到应用](https://www.skler.cn/communication/meeting/202011/W020201105666593156082.pdf)

>  [北航可靠性PPT讲义 – 可靠性网](https://www.kekaoxing.com/3282.html)


# Code

>  [MatthewReid854/reliability: Reliability engineering toolkit for Python - https://reliability.readthedocs.io/en/latest/](https://github.com/MatthewReid854/reliability)

