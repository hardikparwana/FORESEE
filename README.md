# Adversary-CBF


This repository implements our ICRA 2023 submission on 

**FORESEE: Model-based Reinforcement Learning using Unscented Transform with application to Tuning of Control Barrier Functions**

Authors: Hardik Parwana and Dimitra Panagou, University of Michigan

Note: this repo is under development. While all the relevant code is present, we will work on making it more readable and customizable soon! Stay Tuned!


## Description

We aim to solve the following constrained model-based Reinforcement Learning(RL) problem.

Our approach involves following three steps:
- Future state and reward prediction using uncertain dyanmics model
- Compute Policy Gradient
- Peform Constrained Gradient Descent to update policy parameters

## CartPole Swingup

## CBF tuning for Leader-Follower

|  | No Adaptation | With adaptation |
| --------------| -------------------| -----------------|
| No input bound | ![1](https://github.com/hardikparwana/FORESEE/blob/main/no_adapt_no_bound.gif) | ![2](https://github.com/hardikparwana/FORESEE/blob/main/adapt_no_bound.gif) |

| With input bounds | ![3](![1](https://github.com/hardikparwana/FORESEE/blob/main/no_adapt_with_bound.gif)) | ![4](![1](https://github.com/hardikparwana/FORESEE/blob/main/adapt_with_bound.gif))



## Experiments
