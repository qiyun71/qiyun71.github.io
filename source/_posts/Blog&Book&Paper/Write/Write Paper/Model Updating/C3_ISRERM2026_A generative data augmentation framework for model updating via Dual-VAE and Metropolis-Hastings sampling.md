---
title: C3 ISRERM2026
date: 2025-12-23 20:58:15
tags: 
categories: Blog&Book&Paper/Write/Write Paper/Model Updating
Year: 
Journal:
---

| Title        | C3 ISRERM2026 |
| ------------ | ------------------- |
| Author       |                     |
| Organization |                     |
| Paper        |                     |
| Project      |                     |

<!-- more -->

#### Abstract Guidelines
1. Number of characters entered into the form will be checked against the character limit configured and give error when you exceed it. The character limit is 5,000 characters (not words).
2. You can enter entity references/numeric character references, although the number of characters within the references will be counted against the character limit.
3. Copying and pasting text from MS-Word will be accompanied by line breaks and character formatting (bold, italic, underline, subscript and superscript) with the text.  
    Please refer to "[available symbols](https://confit-sfs.atlas.jp/customer/common/List_of_symbols_2024.4.15_en.html)" for symbols you can use in the form.
4. When you copy&paste Greek characters such as "α""β" presented in MS-Word using "Symbol" font, they will be changed to English characters such as "a" "b" in the form. Please use "available symbols" instead.

# Mini-Symposia Proposal

Bayesian Model Inference for Risk and Reliability
organized by Pengfei Wei, Sifeng Bi, Masaru Kitahara, Marcos Valdebenito, Jingwen Song, Alice Cicirello, Matthias Faes, and Michael Beer
Description:
Reliability engineering and risk management increasingly rely on exploring diverse models (e.g., data-driven, physics-based, and hybrid frameworks) to enable robust analysis and decision-making under limited information and uncertainty. The predictive credibility of these models—critical for trustworthy design and risk assessment—hinges on rigorous inference frameworks. Bayesian model inference offers a principled paradigm for model validation, calibration, selection, fusion, and credibility-enhancing experimental design, bridging theoretical foundations with engineering practice. This Mini-Symposia aims to convene researchers and practitioners to explore cutting-edge advances in Bayesian model inference, addressing the following interconnected themes (but not limited to):

(1) Theoretical Foundations, including novel Bayesian nonparametric methods, hierarchical modeling for multi-scale systems, Bayesian model calibration/selection/averaging for uncertainty quantification, credibility metrics integrating physical constraints, structural health monitoring, high-precision learning from small informative datasets or large, heterogeneous and (spatially and temporally) correlated dataset, handling the statistical nature of the measurement uncertainty, etc.
(2) Numerical Innovations, focusing on scalable MCMC algorithms, deep learning-driven variational inference, approximate Bayesian computation for complex physics models, and computational efficiency strategies for high-dimensional parameter spaces.
(3) Engineering Applications, spanning reliability assessment in aerospace propulsion systems, risk-informed design for civil infrastructure under climate uncertainty, fault diagnosis in energy grids, and safety-critical systems in healthcare.
(4) Emerging Frontiers, welcoming interdisciplinary ideas at the intersection of Bayesian inference with AI-driven automated model construction, quantum-informed inference, and real-time data assimilation for dynamic systems.
By fostering cross-disciplinary dialogue, the symposium seeks to advance methodological rigor, promote practical implementations, and identify new frontiers in leveraging Bayesian inference to enhance model credibility and their applications in risk & reliability. This platform will facilitate knowledge exchange, collaborative problem-solving, and the translation of theoretical insights into actionable engineering solutions.

Organizers:
Pengfei Wei (Northwestern Polytechnical University) E-mail: pengfeiwei@nwpu.edu.cn
Sifeng Bi (Beihang University, Beijing) E-mail: sifeng.bi@buaa.edu.cn
Masaru Kitahara (Hokkaido University) E-mail: kitahara@eng.hokudai.ac.jp
Marcos Valdebenito (Technical University of Dortmund) E-mail: marcos.valdebenito@tu-dortmund.de
Jingwen Song (Northwestern Polytechnical University) E-mail: jingwensong@nwpu.edu.cn
Alice Cicirello (University of Cambridge) E-mail: ac685@cam.ac.uk
Matthias Faes (Technical University of Dortmund) E-mail: matthias.faes@tu-dortmund.de
Michael Beer (Leibniz Universität Hannover) E-mail: beer@irz.uni-hannover.de

# Title

A generative data augmentation framework for model updating via Dual-VAE and Metropolis-Hastings sampling

# Abstract

Model updating is a critical technique in ensuring the accuracy and reliability of numerical simulations. However, traditional data-driven model updating techniques often struggle with limited experimental or high-fidelity simulation data due to cost and time constraints. To bridge the gap between limited high-fidelity data and the requirements of deep learning, this paper proposes a novel generative data augmentation framework based on a Dual Variational Autoencoder (Dual-VAE) architecture. Unlike standard generative models, the proposed framework employs two parallel encoders to map structural parameters and system responses into an unified latent space, constrained by latent space consistency and reconstruction losses to capture their intrinsic correlations. To guarantee the validity of the generated data, a rigorous sample screening strategy integrating prior physical constraint and the Metropolis-Hastings (MH) algorithm is introduced to reject outliers and ensure manifold consistency. Subsequently, the augmented dataset is subsequently used to construct a robust inverse surrogate model that accurately maps system responses to model parameters. Numerical case studies demonstrate that the proposed method significantly improves model updating accuracy in small-sample conditions.

# Keywords

Model updating, Data augmentation, Variational autoencoder, Metropolis-Hastings sampling