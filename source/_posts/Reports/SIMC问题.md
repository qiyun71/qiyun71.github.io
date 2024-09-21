我现在写的Self-supervised Interval Model Calibration方法，假定把里面的神经网络叫做**自监督区间校准模型SICM**，它可以有两种训练的方法： (**目前我现在论文使用的是第一种，之前我在introduction有个创新点想错了——加快训练速度/减少训练次数，实际上SICM并没有这个能力**)
1. 如果用生成的(大量)仿真区间数据$\boldsymbol{Y}^{\mathbf{I}}_{sim}$进行训练，那训练结束后可以做到快速的校准/辨识，就是训练速度很慢(数据集大)。这种方法意义不是很大，只能说相较于传统Model Updating方法有速度提升。实际上与Naive Interval Model Calibration (NICM)相比没有改进，硬凑的一点是能将model validation的精度提升，也就是减小校准后$\boldsymbol{Y}^{\mathbf{I}}_{cal}$与实验的$\boldsymbol{Y}^{\mathbf{I}}_{\exp}$之间的平均误差，但是这里有一个大前提是SICM校准后model validation使用的区间传播 (Interval propagation) 方法必须与SICM结构中用的区间传播方法相同。 **!!!!!!** 如果所有(即训练集生成、SICM结构、model validation)的区间传播用同一种方法，那么SICM与NICM相比，没有任何(精度/效率)提升，这是由于训练集生成时通过区间传播生成，训练集中的$\boldsymbol{X}^{\mathbf{I}}$ 与 $\boldsymbol{Y}^{\mathbf{I}}$ 是相对应的，因此SICM监督$\boldsymbol{X}^{\mathbf{I}}$ 与 监督 $\boldsymbol{Y}^{\mathbf{I}}$ 之间的效果是一样的。**| SICM方法创新点：1. 与传统方法相比可以快速辨识区间参数  2. (硬凑的点)当训练集生成与model validation的区间传播方法不同，且当SICM结构中区间传播方法与model validation中方法相同时，SICM相较于NICM来说model validation时有精度提升。*也就是说创新点为SICM可以消除数据集生成时由于区间传播方法所带来的误差，提升model validation的精度(相较于之前的方法BCNN/DCNN都没有考虑区间模型修正中区间传播带来的误差)* |** (实际上，所有区间传播使用同一种方法，精度会更高一点)
2. 如果用(一组)实验数据$\boldsymbol{Y}^{\mathbf{I}}_{\exp}$进行训练，神经网络训练很快而且网络收敛后也可以做到快速的校准/辨识。**这是一种在线训练的方式，有点类似Updating的过程，但又有所差异**：
    - 传统的Model Updating：SSA等优化算法不断寻找最优结构参数$\boldsymbol{X}^{\mathbf{I}}_{optimal}$，让$\boldsymbol{Y}^{\mathbf{I}}_{sim}$与$\boldsymbol{Y}^{\mathbf{I}}_{\exp}$之间的差异越来越小
    - SICM进行 Updating：通过loss函数(网络预测的$\boldsymbol{Y}^{\mathbf{I}}_{predict}$与$\boldsymbol{Y}^{\mathbf{I}}_{\exp}$之间的差异)，反向传播不断调整神经网络的参数(权重参数和偏差参数)，最终可以让网络输出符合要求的辨识参数$\boldsymbol{X}^{\mathbf{I}}_{predict}$。**训练后的MLP就拥有了Model Calibration的能力，但是理论上只能准确地Calibrate 这一组实验数据**

---

目前论文引言中的这一段可以修改成这样吗：
Recent advancements have seen researchers propose inverse surrogate models based on neural networks to identify structural deterministic parameters from measured dynamic response features. The inverse surrogate model based on neural networks facilitates the direct and rapid calibration of structural parameters, eliminating the need for extensive time and computational resources typically required for parameter optimization. Yang et al. [42] have recently proposed using an inverse model based on a dilated convolutional neural network (DCNN) to identify dynamic loads from vibration responses. The network nonlinearity enables a good fit between sequential data. Similarly, Wang et al. [43] employ a Bayesian convolutional neural network (BCNN) to simultaneously update structural and damping parameters from frequency response feature maps. ~~However, these methods require retraining the neural network if the updated model fails to achieve the desired accuracy after validation, leading to a substantial computational burden. In contrast, a self-supervised [44] method based on MLP neural network can effectively eliminate the need for multiple training sessions for model updating.~~ **这些方法通过反向辨识模型可以准确地校准有限元结构参数。然而由于区间传播会产生一定的误差，这些方法直接用于区间参数校准会导致校准后参数在model validation 时精度很低。相反，自监督方法由于其从数据本身提取潜在特征因此可以消除这一误差。**

---

方法1(目前这篇论文方法) 与 方法2 的精度对比：

| $\boldsymbol{Y}^{\mathbf{I}}_{predict}$于$\boldsymbol{Y}^{\mathbf{I}}_{\exp}$的平均误差 | 1. SICM训练 (Identification) | 2. SICM 在线训练 (Updating) |
| --------------------------------------------------------------------------------- | -------------------------- | ----------------------- |
| 区间下界                                                                              | 0.326                      | 0.420                   |
| 区间上界                                                                              | 0.307                      | 0.295                   |
| 训练时间                                                                              | 60min以上                    | 不到1min                  |
| 推理时间                                                                              | 1s                         | 1s                      |

目前的对这两种方法的想法：
- 方法一网络是通过大量的有限元仿真数据训练死的，只是一个有限元参数 逆辨识模型/校准模型。(更适合学术)。意义不大，除非生成数据集的时候不进行区间传播(不先生成结构参数区间$\boldsymbol{X}^{\mathbf{I}}$，然后区间传播得到$\boldsymbol{Y}^{\mathbf{I}}$ )，而是只随机生成一些特征频率区间$\boldsymbol{Y}^{\mathbf{I}}$(优点就是可以减少数据集生成的困难)。但这样由于这些随机生成的特征频率区间没法通过SICM神经网络预测+区间传播得到，可能会导致网络无法收敛。所以实际上不如方法二——直接使用实验数据来进行训练。
- 方法二网络是活的，通过测量实验数据和在线训练的方式可以不断增强网络的能力。(更适合工程)。但是意义也不大，因为一组/一次实验的数据只需要校准/辨识一次，除非多次测量的实验区间参数非常相近，这样网络收敛的速度会很快。

---

(区间模型修正中) 实验测量特征频率的区间，应该会有多次测量的数据，而不是只有这样一组实验数据：

| 特征频率  | 不确定性区间参数(测量)     |
| ----- | ---------------- |
| $f_1$ | [42.66, 43.64]   |
| $f_2$ | [118.29, 121.03] |
| $f_3$ | [133.24, 136.54] |
| $f_4$ | [234.07, 239.20] |
| $f_5$ | [274.29, 280.64] |

***但是实验测量数据是很难获得的，所以方法2在线训练可以通过以下流程让网络逐渐增强：***
- 先在第一组实验区间数据上进行训练，训练快速收敛后停止训练，网络参数固定，此时网络可以准确校准这一组实验的区间
- 然后当第二组实验被测得时，与一组实验区间数据组合，对网络进行继续训练，同样收敛后停止，此时理论上网络可以准确校准这两组实验的区间
- 依次类推，随着实验次数增多，训练时间会不断加长，但是推理也就是校准/辨识的时间还是很快

**实验数据越多，网络训练越多，神经网络校准能力越强**
