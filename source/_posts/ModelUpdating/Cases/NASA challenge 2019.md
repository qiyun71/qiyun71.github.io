---
title: NASA challenge 2019
date: 2025-08-13 13:10:33
tags: 
categories: ModelUpdating/Cases
Year: 
Journal:
---

> [NASA Langley UQ Challenge on Optimization Under Uncertainty](https://uqtools.larc.nasa.gov/nasa-uq-challenge-problem-2020/)
> [Towards the NASA UQ Challenge 2019: Systematically forward and inverse approaches for uncertainty propagation and quantification - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0888327021007408?via%3Dihub)

- **Model Calibration** & Uncertainty Quantification of Subsystems
- Uncertainty Model Reduction
- Reliability Analysis
- Reliability-Based Design
- Model Updating and Design Parameter Tuning
- Risk-based Design

**Model Calibration:**

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20240307224720.png)

如何根据目标的y(a,e,t) 校准不确定性参数 a & e

### Equation

The NASA Langley Research Center introduced the Optimization under Uncertainty Challenge [23], primarily aimed at the development of new aerospace vehicles and systems designed to function within harsh environments under a range of operational conditions. Its subproblem A, which pertains to model calibration and UQ of Subsystems, is illustrated in Fig. 6. This subproblem specifically addresses the methodology for deriving uncertainty models for aleatory and epistemic parameters based on time-domain response data. This complex case serves to illustrate the accuracy and feasibility of the proposed inverse neural network calibration approach. 

In this problem, the physical system is represented as a black-box analytical model without losing any fidelity. The system response is a time-domain sequential data defined as y(t) = y(a, e, t), where a and e are the aleatory and epistemic variables, respectively. The uncertainty model for a is defined as ai ∼ fa,i = 1,⋯,5, where fa is a unknown Probability Density Function (PDF), presenting aleatory uncertainty. The uncertainty model for e is defined as ej ∈ E, j = 1, ⋯4, where E is an interval representing epistemic uncertainty. 

100 sets of time-domain response data are provided by the Challenge host as the observation set D1, which will be employed to identify the uncertainty characteristics of parameters a and e based on the proposed neural network model.

### Applications

#### Sensitivity analysis

|     |     |
| --- | --- |
|     |     |


#### Stochastic model calibration with image encoding