---
title: J4_Virtual sampling generation
date: 2025-12-26 14:19:58
tags: 
categories: Blog&Book&Paper/Write/Write Paper/Model Updating
Year: 
Journal:
---

| Title        | J4_Virtual sampling generation |
| ------------ | ------------------- |
| Author       |                     |
| Organization |                     |
| Paper        |                     |
| Project      |                     |

<!-- more -->

# Code

```python
# Data generation
# Dataset for training Dual-VAE
# Data augmentation
# DatasetISM for training inverse surrogate model
# Model updating
# Model validation
```

# Title

A generative data augmentation framework for model updating via Dual-VAE and Metropolis-Hastings sampling

All parameters (strcutrual parameters, measured response, pointcloud) is derived from object essence (latent space).

# Abstract

Model updating is a critical technique in ensuring the accuracy and reliability of numerical simulations. However, traditional data-driven model updating techniques often struggle with limited experimental or high-fidelity simulation data due to cost and time constraints. To bridge the gap between limited high-fidelity data and the requirements of deep learning, this paper proposes a novel generative data augmentation framework based on a Dual Variational Autoencoder (Dual-VAE) architecture. Unlike standard generative models, the proposed framework employs two parallel encoders to map structural parameters and system responses into an unified latent space, constrained by latent space consistency and reconstruction losses to capture their intrinsic correlations. To guarantee the validity of the generated data, a rigorous sample screening strategy integrating prior physical constraint and the Metropolis-Hastings (MH) algorithm is introduced to reject outliers and ensure manifold consistency. Subsequently, the augmented dataset is subsequently used to construct a robust inverse surrogate model that accurately maps system responses to model parameters. Numerical case studies demonstrate that the proposed method significantly improves model updating accuracy in small-sample conditions.

# Keywords

Model updating, Data augmentation, Variational autoencoder, Metropolis-Hastings sampling


# Introduction

