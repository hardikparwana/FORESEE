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

The first two steps are known to be computationally intractable. A popular method, introduced in PILCO, computes analytical formulas for mean and covariance poropagation when the prior distribution is given by a Gaussian and the transition dynamics is given by a gaussian process with a gaussian kernel. We instead use Unscented Transform to propagate states to the future. Depending on number of soigma points employed, we can maintain mean and covariances or even higher order moments of the distribution. Propagting finite number of particles (sigma points) through state-dependent uncertainty model though requires increase in number of sigam points to be able to represent the distributions and this leads to an explosion that is undesirable. Therefore, we introduce differentiable sigma point expansion and compression layer based on moment matching that allows us to keep the algorithm scalable.

## CartPole Swingup



## CBF tuning for Leader-Follower

|  | No Adaptation | With adaptation |
| --------------| -------------------| -----------------|
| No input bound | ![1](https://github.com/hardikparwana/FORESEE/blob/main/no_adapt_no_bound.gif) | ![2](https://github.com/hardikparwana/FORESEE/blob/main/adapt_no_bound.gif) |
| With input bounds | ![3](https://github.com/hardikparwana/FORESEE/blob/main/no_adapt_with_bound.gif) | ![4](https://github.com/hardikparwana/FORESEE/blob/main/adapt_with_bound.gif)



## Experiments
