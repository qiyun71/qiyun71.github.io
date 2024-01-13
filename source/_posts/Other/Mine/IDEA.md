
# NeuS重建三维模型最好使用多少视图

[How Many Views Are Needed to Reconstruct an Unknown Object Using NeRF?](https://arxiv.org/pdf/2310.00684.pdf)

- 三维重建模型质量评价指标CD倒角距离
- NeuS系列方法需要最好地视图数量，在评价指标上进行对比

# NeRF方法重建一个场景需要手调超参数

每个场景单独训练，如果超参数不同，得到的模型质量结果也不同
- 每个场景都有一个最优的学习率