---
title: Data Analysis
date: 2024-09-21 14:32:41
tags: 
categories: Learn
---

数据降维/

sklearn | numpy | pandas
seaborn | matplotlib

<!-- more -->


# 数据降维(Dimension Reduction)

>  [常见的PCA、tSNE、UMAP降维及聚类基本原理及代码实例_umap聚类-CSDN博客](https://blog.csdn.net/qq_43337249/article/details/116612811)

What：
降维是指通过保留一些比较重要的特征，去除一些冗余的特征，减少数据特征的维度。而特征的重要性取决于该特征能够表达多少数据集的信息，也取决于使用什么方法进行降维。一般情况会先使用线性的降维方法再使用非线性的降维方法，通过结果去判断哪种方法比较合适。
降维算法一般分为两类: 1.寻求在数据中保存距离结构的:PCA、MDS等算法 2倾向于保存局部距离而不是全局距离的。`t-SNE` diffusion maps(UMAP)

Why：
- 数据的多重共线性：特征属性之间存在着相互关联关系。多重共线性会导致解的空间不稳定， 从而导致模型的泛化能力弱；
- 高纬空间样本具有稀疏性，导致模型比较难找到数据特征；过多的变量会妨碍模型查找规律；
- 仅仅考虑单个变量对于目标属性的影响可能忽略变量之间的潜在关系。
- ~~当数据的特征太多，无法很好地展示时，需要降维~~


How：

## Method about Dimension Reduction

### PCA

PCA是将数据的最主要成分提取出来代替原始数据，也就是将n维特征映射到新的维度中，由k维正交特征组成的特征空间就是主成分，使用的降维方法就是投影。


### t-SNE

t-distributed Stochastic Neighbor Embedding


### UMAP

> [UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction](https://arxiv.org/pdf/1802.03426)
> [下一次数学突破会在哪里？](https://www.zhihu.com/question/21550185/answer/3363179389) UMAP



## Code about t-SNE and PCA：

```python
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# n_samples, seq_length, feature_dim
# ori_data 3000,24,6
# generate_data 3000,24,6

# compare设置的比ori_data.shape[0]小一点方便计算
anal_sample_no = min([compare, ori_data.shape[0]])

idx = np.random.permutation(ori_data.shape[0])[:anal_sample_no]
ori_data = ori_data[idx]
generated_data = generated_data[idx]

# 对 feature_dim 取平均 3000,24,6 --> 3000,24
for i in range(anal_sample_no):
  if (i == 0):
    prep_data = np.reshape(np.mean(ori_data[0, :, :], 1), [1, seq_len])
    prep_data_hat = np.reshape(np.mean(generated_data[0, :, :], 1), [1, seq_len])
  else:
    prep_data = np.concatenate((prep_data,np.reshape(np.mean(ori_data[i, :, :], 1), [1, seq_len])))
    prep_data_hat = np.concatenate((prep_data_hat,np.reshape(np.mean(generated_data[i, :, :], 1), [1, seq_len])))

# 数据降维
# # prep_data 3000,24
# # prep_data_hat 3000,24

## PCA
pca = PCA(n_components=2)
pca.fit(prep_data) # 对`prep_data`进行训练，计算主成分（方向向量）
### 将原始数据投影到2个主成分的空间上，`pca_results`是降维后的数据
pca_results = pca.transform(prep_data) # (3000, 2) ori
pca_hat_results = pca.transform(prep_data_hat) # (3000, 2) gen

## t-SNE

### Do t-SNE Analysis together
prep_data_final = np.concatenate((prep_data, prep_data_hat), axis=0) # 6000, 24

### TSNE anlaysis
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(prep_data_final) # 6000,2
### 降维后的数据
tsne_results = tsne_results[:anal_sample_no, :] # 3000,2
pca_hat_results = tsne_results[anal_sample_no:, :] # 3000,2
```

