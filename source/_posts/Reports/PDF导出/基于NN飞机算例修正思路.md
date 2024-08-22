## 基于NN方法

### 数据集生成

$\mu_a,\sigma_a,\mu_b,\sigma_b$ 4个参数 --> $f_1,f_2,f_3,f_4,f_5$ 前5阶固有频率
- $\mu_{a}\in [290,310]$ , $\sigma_{a} \in [0,5]$ (mm)
- $\mu_{b}\in [20,30]$ , $\sigma_{b} \in [0,5]$ (mm)

$\mu_a,\sigma_a,\mu_b,\sigma_b$ 4个参数在各自的区间内均匀分布，随机抽取m=500组

针对抽取的m组中的每一组$\mu_a,\sigma_a,\mu_b,\sigma_b$ ：需要生成n=50组$a,b,E_1,E_2$数据
- 根据$\mu_a,\sigma_a,\mu_b,\sigma_b$ ，得到a和b一般正态分布$X_{a} \sim N(\mu_{a},\sigma^{2}_{a}),X_{b} \sim N(\mu_b,\sigma^{2}_b)$ n = 50
- 根据50组$a,b$，使用Nastran生成100组 $f_1,f_2,f_3,f_4,f_5$

总共需要使用Nastran进行$500 \times 50 = 25,000$次计算
按照每次计算花费10s计算，共需要25万 s = 69.4 h = 2.89 day

### 网络结构

神经网络训练：
- 输入：5x100大小的数组（100组 $f_1,f_2,f_3,f_4,f_5$）
- 标签：4x1的向量（$\mu_a,\sigma_a,\mu_b,\sigma_b$ ）

训练思路：
1. 输入5x100大小的数组通过FC(全连接层)计算得到中间向量，然后reshape成3通道图片，使用CNN处理图片，提取特征并解码为4x1的向量

