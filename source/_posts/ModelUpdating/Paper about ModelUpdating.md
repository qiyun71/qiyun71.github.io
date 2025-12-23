---
title: Paper about ModelUpdating
date: 2024-02-27 17:06:54
tags: 
categories: ModelUpdating
---

**Uncertainty**
- Epistemic Uncertainty
  - 源自认知信息不足，可以减少乃至消除
- Aleatory Uncertainty
  - 源自系统/结构固有的随机性

**Model Updating**
- 确定性 MU
- 随机 MU(包括 BayesMU)
- 区间 MU

- Certainty
- Uncertainty
  - Numerical
    - Perturbation method (Taylor expansion)
    - Karhunen-Loeve Expansion
    - Polynomial Chaos Expansion
  - Statistics
    - Monte Carlo Simulation 
      - FE surrogate mode
      - Design of Experiment (Sampling method)
  - Fuzzy
    - Fuzzy Polynomial Expansion


[2]. Sifeng Bi (申请人), K. He, Y. Zhao, D. Moens, M. Beer, J. Zhang, “Towards the NASA  UQ challenge 2019: Systematically forward and inverse approaches for uncertainty  propagation and quantification,” Mechanical Systems and Signal Processing, vol. 165,  p.108387, 2022. [IF:7.9,唯一一作,唯一通讯,中科院 Top]  
[3]. Sifeng Bi (申请人), M. Broggi, M. Beer, “The role of the Bhattacharyya distance in  stochastic model updating,” Mechanical Systems and Signal Processing, vol. 117, pp.  437-452, 2019. [IF:7.9,唯一一作,唯一通讯,中科院 Top]  [4]. Sifeng Bi (申请人), M. Broggi, P. Wei, M. Beer, “The Bhattacharyya distance: Enriching  the P-box in stochastic sensitivity analysis,” Mechanical Systems and Signal Processing,  vol. 129, pp. 265-281, 2019. [IF:7.9,唯一一作,唯一通讯,中科院 Top] 
[5]. Sifeng Bi (申请人), S. Prabhu, S. Cogan, S. Atamturktur, “Uncertainty quantification  metrics with varying statistical information in model calibration and validation,” AIAA  Journal, vol. 55 (10), pp. 3570-3583, 2017. [IF:2.1,唯一一作,唯一通讯,航空航天 领域知名期刊]  
[8]. Y. Zhao, B. Sun, Sifeng Bi (申请人), M. Beer, D. Moens, “A sub-convex similarity-based  model updating method considering multivariate uncertainties,” Engineering Structures,  vol. 318, p. 118752, 2024. [IF:5.6,唯一通讯,中科院 Top,获期刊 Featured Paper  Award]  
[9]. Y. Zhao, J. Yang, M. Faes, Sifeng Bi (申请人), Y. Wang, “The sub-interval similarity: A  general uncertainty quantification metric for both stochastic and interval model updating,”  Mechanical Systems and Signal Processing, vol. 178, p.109319, 2022. [IF:7.9,唯一通 讯,中科院 Top] 

[10]. X. Wei, J. Liu, Sifeng Bi (申请人), “Uncertainty quantification and propagation of crowd  behaviour effects on pedestrian-induced vibrations of footbridges,” Mechanical Systems  and Signal Processing, vol. 167, p. 108557, 2022. [IF:7.9,唯一通讯,中科院 Top]

<!-- more -->

"web of science" search results:
https://www.webofscience.com/wos/alldb/summary/2476080f-19c8-476d-a43c-136a99535f14-01672444ff/times-cited-descending/1


# Review

| Year     | Paper                                                                                                                                                                                       | Summarize&*Case*                                                                                                                 |
| -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| 1993     | [Model Updating In Structural Dynamics: A Survey - ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0022460X83713404)                                                  | 从确定性的角度对结构动力学中的模型修正进行了全面的回顾                                                                                                      |
| 2006     | [结构动力学有限元模型修正的发展------模型确认](结构动力学有限元模型修正的发展------模型确认.md)                                                                                                                                   |                                                                                                                                  |
| 2011     | [The sensitivity method in finite element model updating: A tutorial - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0888327010003316)                                  | 有限元模型的误差来源：<br>模型结构误差(对物理结构力学行为特征的假设、数值方法引入的离散化误差)<br>典型误差(模型参数的错误假设，如 E、G、T 等)                                                    |
| 2013     | [Imprecise probabilities in engineering analyses - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0888327013000812)                                                      | 不精确概率模型，如 evidence theory, probability bounds analysis with p-boxes, and fuzzy probabilities，info-gap theory                      |
| 2016     | [Structural Dynamic Model Updating Techniques: A State of the Art Review \| Archives of Computational Methods in Engineering](https://link.springer.com/article/10.1007/S11831-015-9150-3)  | FE 动态响应的影响因素<br>![image.png\|333](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240411111847.png) |
|          | [Structural Dynamic Model Updating Techniques: A State of the Art Review \| Archives of Computational Methods in Engineering](https://link.springer.com/article/10.1007/S11831-015-9150-3)  | FEMU 应用<br>![image.png\|333](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240411112415.png)<br>  |
| 2019     | [有限元模型修正研究进展(从线性到非线性)](有限元模型修正研究进展(从线性到非线性).md)                                                                                                                                             | 对线性 FEMU 的总结，以及对非线性的展望                                                                                                             |
| 2021     | [Overview of Stochastic Model Updating in Aerospace Application Under Uncertainty Treatment \| SpringerLink](https://link.springer.com/chapter/10.1007/978-3-030-83640-5_8)                 |                                                                                                                                  |
| 2022     | [Review of finite element model updating methods for structural applications - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2352012422004039)                          |                                                                                                                                  |
| **2023** | [Stochastic Model Updating with Uncertainty Quantification_An Overview and Tutorial](Review/Stochastic%20Model%20Updating%20with%20Uncertainty%20Quantification_An%20Overview%20and%20Tutorial.md) | 不确定性来源：参数不确定性、模型不确定性、实验不确定性<br>*NASA UQ 挑战 2014、飞机模型*                                                                              |

# Deterministic Model Updating

| Year | Paper                                                                                                                                                                                              | Summarize&*Case*                                                                                                                                                                                     |
| ---- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1974 | [Statistical Identification of Structures \| AIAA Journal](https://arc.aiaa.org/doi/10.2514/3.49190)                                                                                               | the inverse eigensensitivity method                                                                                                                                                                  |
| 2010 | [Finite Element Model Updating Using Computational Intelligence Techniques: Applications to Structural Dynamics \| SpringerLink](https://link.springer.com/book/10.1007/978-1-84996-323-7)         | the maximum likelihood approach and Bayesian approaches<br>multi-layer perceptron neural networks<br>particle swarm and GA-based optimization methods; simulated annealing; response surface methods |
| 2010 | [Finite element model updating using bees algorithm \| Structural and Multidisciplinary Optimization](https://link.springer.com/article/10.1007/s00158-010-0492-z)                                 | accelerometer FRF & (**BA**)bees algorithm & genetic algorithm (GA) & (**PSO**)particle swarm optimization  & the inverse eigensensitivity method                                                    |
| 2016 | [Sequential surrogate modeling for efficient finite element model updating - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S004579491630027X)                                   |                                                                                                                                                                                                      |
| 2016 | [Finite element model updating using simulated annealing hybridized with unscented Kalman filter - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0045794916303935)             | 模拟退火 SA 算法                                                                                                                                                                                           |
| 2017 | [Sensitivity-based finite element model updating of a pontoon bridge - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0141029616307830)                                         | 基于敏感度                                                                                                                                                                                                |
| 2019 | [Finite element model updating using deterministic optimisation: A global pattern search approach - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0141029618340665?via%3Dihub) | global pattern search(deterministic optimisation), 同时也讨论了一些其他的优化算法                                                                                                                                   |
| 2023 | [Deterministic and probabilistic-based model updating of aging steel bridges - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2352012423006239)                                 | probabilistic-based                                                                                                                                                                                  |

# Stochastic Model Updating

| Year | Paper                                                                                                                                                                                                                                                                      | Summarize&*Case* |
| ---- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------- |
| 2005 | [Uncertainty identification by the maximum likelihood method - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0022460X05004529)                                                                                                                         |                  |
| 2015 | [A Monte Carlo simulation based inverse propagation method for stochastic model updating - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0888327015000138)                                                                                             |                  |
| 2022 | [Active learning structural model updating of a multisensory system based on Kriging method and Bayesian inference - Yuan - 2023 - Computer-Aided Civil and Infrastructure Engineering - Wiley Online Library](https://onlinelibrary.wiley.com/doi/abs/10.1111/mice.12822) | Active learning  |
| 2023 | [Active learning aided Bayesian nonparametric general regression for model updating using modal data - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0888327023007380)                                                                                 | Active learning  |
| 2023 | [(PDF) Stochastic model updating based on sub-interval similarity and BP neural network](https://www.researchgate.net/publication/367239633_Stochastic_model_updating_based_on_sub-interval_similarity_and_BP_neural_network)                                              |                  |
| 2024 | [Efficient Bayesian inference for finite element model updating with surrogate modeling techniques \| Journal of Civil Structural Health Monitoring](https://link.springer.com/article/10.1007/s13349-024-00768-y)                                                         |                  |
| 2024 | [Latent Space-based Stochastic Model Updating \| alphaXiv](https://www.alphaxiv.org/abs/2410.03150)                                                                                                                                                                        |                  |

# Interval Model Updating

| Year | Paper                                                                                                                                                                                                                                                                     | Summarize&*Case*                                                                                                                                         |
| ---- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2010 | [Interval model updating: method and application](https://past.isma-isaac.be/downloads/isma2010/papers/isma2010_0140.pdf)                                                                                                                                                 |                                                                                                                                                          |
| 2011 | [Interval model updating with irreducible uncertainty using the Kriging predictor](Interval%20model%20updating%20with%20irreducible%20uncertainty%20using%20the%20Kriging%20predictor.md)                                                                                 | Kriging predictor *质量弹簧、梁*                                                                                                                               |
| 2015 | [An interval model updating strategy using interval response surface models](An%20interval%20model%20updating%20strategy%20using%20interval%20response%20surface%20models.md)                                                                                             | IRSM*质量弹簧、钢板*                                                                                                                                            |
| 2017 | [Interval model updating using perturbation method and Radial Basis Function neural networks](Interval%20model%20updating%20using%20perturbation%20method%20and%20Radial%20Basis%20Function%20neural%20networks.md)                                                       | PM 和 RBF-NN*钢板、卫星*                                                                                                                                         |
| 2018 | [Interval identification of structural parameters using interval overlap ratio and Monte Carlo simulation](Interval%20identification%20of%20structural%20parameters%20using%20interval%20overlap%20ratio%20and%20Monte%20Carlo%20simulation.md)                           | IOR 指标和 MC 传播*质量弹簧系统、钢板*                                                                                                                                    |
| 2018 | [Full article: Structural dynamics model updating with interval uncertainty based on response surface model and sensitivity analysis](https://www.tandfonline.com/doi/full/10.1080/17415977.2018.1554656)                                                                 | response surface model and sensitivity analysis                                                                                                          |
| 2019 | [Interval Identification of Structural Parameters Using Interval Deviation Degree and Monte Carlo Simulation](Interval%20Identification%20of%20Structural%20Parameters%20Using%20Interval%20Deviation%20Degree%20and%20Monte%20Carlo%20Simulation.md)                     | IDD 指标和 MC 传播*质量弹簧系统、钢板、卫星*                                                                                                                                 |
| 2019 | [A multivariate interval approach for inverse uncertainty quantification with limited experimental data - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0888327018305946?dgcid=raven_sd_recommender_email)                                            | *DLR-AIRMOD*                                                                                                                                             |
| 2020 | [Interval model updating using universal grey mathematics and Gaussian process regression model](Interval%20model%20updating%20using%20universal%20grey%20mathematics%20and%20Gaussian%20process%20regression%20model.md)                                                 | *质量弹簧系统、铝合金板*                                                                                                                                            |
| 2022 | [MULTILEVEL QUASI-MONTE CARLO FOR INTERVAL ANALYSIS](https://www.scopus.com/record/display.uri?eid=2-s2.0-85131119871&origin=inward&txGid=67f5737367330e0d7788bae32149f241)                                                                                               | 基于 MC 的区间不确定性传播                                                                                                                                            |
| 2022 | [Bayesian inversion for imprecise probabilistic models using a novel entropy-based uncertainty quantification metric](Bayesian%20inversion%20for%20imprecise%20probabilistic%20models%20using%20a%20novel%20entropy-based%20uncertainty%20quantification%20metric.md)<br> | *简支梁、NASA UQ 挑战 2014*                                                                                                                                      |
| 2022 | [A novel interval model updating framework based on correlation propagation and matrix-similarity method](A%20novel%20interval%20model%20updating%20framework%20based%20on%20correlation%20propagation%20and%20matrix-similarity%20method.md)                             | *质量弹簧系统、真实滑动质量梁*                                                                                                                                         |
| 2023 | [Efficient inner-outer decoupling scheme for non-probabilistic model updating with high dimensional model representation and Chebyshev approximation - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0888327022011086)                                | *质量弹簧系统、广州塔、a free-free plate*                                                                                                                           |
| 2024 | [An improved interval model updating method via adaptive Kriging models](An%20improved%20interval%20model%20updating%20method%20via%20adaptive%20Kriging%20models.md)                                                                                                     | *质量弹簧系统、对接圆柱壳体*                                                                                                                                          |
| 2024 | [Certified interval model updating using scenario optimization \| AIAA SciTech Forum](https://arc.aiaa.org/doi/10.2514/6.2024-0172)                                                                                                                                       | Scenario optimization                                                                                                                                    |
|      |                                                                                                                                                                                                                                                                           |                                                                                                                                                          |
|      |                                                                                                                                                                                                                                                                           |                                                                                                                                                          |
| 2024 | [An Interval Neural Network Method for Identifying Static Concentrated Loads in a Population of Structures](https://www.mdpi.com/2226-4310/11/9/770)                                                                                                                      | [Interval Neural Networks 2020](https://arxiv.org/pdf/2003.11566) 通过物理参数和 measured response 来辨识集中负载。没有用 Gradients-based 的算法，而是使用了 GA(Genetic Algorithm)进行优化 |

# 通用 Method

| Year | Paper                                                                                                                                                                                                                                                                         | Summarize&*Case*                      |
| ---- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------- |
| 2007 | [Transitional Markov Chain Monte Carlo Method for Bayesian Model Updating, Model Class Selection, and Model Averaging](Transitional%20Markov%20Chain%20Monte%20Carlo%20Method%20for%20Bayesian%20Model%20Updating,%20Model%20Class%20Selection,%20and%20Model%20Averaging.md) | TMCMC 采样方法                             |
| 2014 | [Modified perturbation method for eigenvalues of structure with interval parameters \| Science China Physics, Mechanics & Astronomy](https://link.springer.com/article/10.1007/s11433-013-5328-6)                                                                             | Perturbation, interval parameters     |
| 2020 | [Bayesian probabilistic damage characterization based on a perturbation model using responses at vibration nodes - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S088832701930665X?via%3Dihub#s0025)                                                       | Perturbation-based FE surrogate Model |
| 2023 | [Distribution-free stochastic model updating with staircase den](https://www.taylorfrancis.com/chapters/oa-edit/10.1201/9781003323020-81/distribution-free-stochastic-model-updating-staircase-density-functions-kitahara-kitahara-bi-broggi-beer)                            | 基于贝叶斯的无分布参数估计                         |
|      |                                                                                                                                                                                                                                                                               |                                       |

# Inverse Surrogate Model

| Year | Journal                                               | Paper                                                                                                                                                                                                                                                         | Summarize&*Case*                                                                                                                                                                        |
| ---- | ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2004 | Engineering Applications of Artificial Intelligence   | [Direct identification of structural parameters from dynamic responses with neural networks - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0952197604000922)                                                                             |                                                                                                                                                                                         |
| 2018 | Computational Mechanics                               | [Fast model updating coupling Bayesian inference and PGD model reduction \| Computational Mechanics](https://link.springer.com/article/10.1007/s00466-018-1575-8)                                                                                             |                                                                                                                                                                                         |
| 2019 | Mechanics of Advanced Materials and Structures **Q2** | [A frequency response model updating method based on unidirectional convolutional neural network: Mechanics of Advanced Materials and Structures: Vol 28 , No 14 - Get Access](https://www.tandfonline.com/doi/full/10.1080/15376494.2019.1681037)            | convert FRF to feature map with 2 channel<br>UCNN<br>                                                                                                                                   |
| 2023 | Mechanical Systems and Signal Processing              | [A feature map of frequency response functions based model updating method using the Bayesian convolutional neural network - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0888327023007264?ref=pdf_download&fr=RR-2&rr=87df480b9ca904d1) | A feature map of frequency response functions(FMFRF)<br>[Bayesian convolutional neural network(**BCNN**)](https://arxiv.org/pdf/1901.02731)<br>a benchmark multi-plate structure        |
| 2023 | Mechanical Systems and Signal Processing              | [Dynamic load identification based on deep convolution neural network - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0888327022008251)                                                                                                   | Deep Dilated Convolution neural network (DCNN)<br>vibration response(time series) --> excitation(time series), 在序列数据的处理上 DCNN 和 CNN 方法由于 MLP                                                 |
| 2024 | Mechanical Systems and Signal Processing              | [Inverse surrogate model for deterministic structural model updating based on random forest regression - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0888327024003145)                                                                  | Deterministic Model Updating based on random forest regression<br>![image.png\|333](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240709170910.png)<br> |


## [Inverse surrogate model for deterministic structural model updating based on random forest regression - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0888327024003145)

- ？This occurs since optimization methods tends to focus only on minimizing an objective function based on the disparity between experimental and model-derived modal parameters, typically **neglecting the mutual relations between the latter**, therefore increasing the risk to convergence to local minima.  NN 可以考虑 exp 和 sim 的 dynamic response(mode frequency 前后几阶)各自之间的相关性
- NN-based 对结构的几何和机械特性的预测要比传统方法得出的结果准确得多
- NN-based 方法不受初始化的影响

## [Dynamic load identification based on deep convolution neural network - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0888327022008251#s0090)
- CNN：strong anti-noise ability 由于他的卷积层，可以看作过滤器

## [A feature map of frequency response functions based model updating method using the Bayesian convolutional neural network - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0888327023007264?ref=pdf_download&fr=RR-2&rr=87df480b9ca904d1#s0060)
- 多测量点的 FRFs 可以反映大多数结构信息，如 natural frequency, mode shape, damping。训练集由具有不同噪声水平的 FRF 训练样本组成
- 使用 BCNN 可以为 CNN 引入权重的概率分布，使得 CNN 可以在预测中添加适度的不确定性和正则化，能够在小数据集上避免 overfitting
- *另外，需要指出的是，频响函数的特征图中没有考虑测量点的空间信息，原则上测量点的空间位置也包含重要的实际意义。空间位置信息将来可以扩展到 FRF 的特征图。* **没有在输入中融合空间的位置信息**


> [Bayesian Neural Networks: An Introduction and Survey](https://arxiv.org/pdf/2006.12024)


# Other

global sensitivity analysis (GSA) [parameter-dependence · GitHub Topics](https://github.com/topics/parameter-dependence)