---
title: Case about ModelUpdating
date: 2024-03-07 21:26:57
tags:
  - ModelUpdating
categories: ModelUpdating
---

Model Updating算例

Stochastic model calibration with image encoding：
- [qiyun71/SatelliteModelUpdating: Satellite FE Model Updating](https://github.com/qiyun71/SatelliteModelUpdating)
- resnet： NASA 2019 case

Traditional model updating——Python复现：[qiyun71/TraditionalModelUpdating: Traditional Model Updating by optim algo and loss function(or metrics)](https://github.com/qiyun71/TraditionalModelUpdating)

- [x] Sparrow Search Algorithm(SSA)
- [ ] Particle Swarm Optimization(PSO)
- [ ] Simulated Annealing(SA)

- [X] L1 L2 or MAE MSE or Euclidian distance(ED)
- [x] Bhattacharyya distance(BD)
- [x] Mahalanobis distance(MD)
- [ ] IOR
- [ ] IDD
- [x] Interval Similarity(IS)
- [x] Sub-Interval Similarity(SIS)
- [x] Ellipse Similarity(ES)
- [ ] Sub-Ellipse Similarity(SES)
- [ ] Interval probability box(I-pbox)

Straightforward parameter-based MLP(EI)：
- [qiyun71/MU_MassSpring](https://github.com/qiyun71/MU_MassSpring)
- MU_Airplane
- MU_SteelPlate

Response-consistent MLP for interval model calibration：[qiyun71/NN-basedModelUpdating: NN-based Model Updating](https://github.com/qiyun71/NN-basedModelUpdating)
实验数据训练：[qiyun71/SSMU: Self-supervised Model Updating (SSMU) based on time-domain response data.](https://github.com/qiyun71/SSMU)

<!-- more -->

# Simple

## [NASA challenge 2019](Cases/NASA%20challenge%202019.md)

## [IMAC2023](Cases/IMAC2023.md)

## [Mass Spring System](Cases/Mass%20Spring%20System.md)

# Complex

## [Steel Plate Structures](Cases/Steel%20Plate%20Structures.md)

## [Airplane Model](Cases/Airplane%20Model.md)

## [Satellite Model](Cases/Satellite%20Model.md)

## [Folding Fin](Cases/Folding%20Fin.md)