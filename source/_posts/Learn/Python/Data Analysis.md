---
title: Data Analysis
date: 2024-09-21 14:32:41
tags:
  - 
categories: Learn/Python
---

sklearn | numpy | pandas
seaborn | matplotlib

<!-- more -->


# 数据降维

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