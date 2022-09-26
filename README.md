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
![1] 

|  | No Adaptation | With adaptation |
| --------------| -------------------| -----------------|
|  

![PAPER_with_trust](https://user-images.githubusercontent.com/19849515/162593597-f028c61d-7a9d-4ff9-88b4-5851aeae1806.gif) | ![PAPER_NO_TRUST](https://user-images.githubusercontent.com/19849515/162593600-273fd93a-c82c-4655-b232-a03181672b15.gif) | ![PAPER_NO_TRUST_large_alpha](https://user-images.githubusercontent.com/19849515/162593605-af184d72-0d08-4c7e-bcdf-f88d18b42a5d.gif)



## Experiments
