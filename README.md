# Adversary-CBF


This repository implements our ICRA 2023 submission on 

**FORESEE: Model-based Reinforcement Learning using Unscented Transform with application to Tuning of Control Barrier Functions**

Authors: Hardik Parwana and Dimitra Panagou, University of Michigan

Note: this repo is under development. While all the relevant code is present, we will work on making it more readable and customizable soon! Stay Tuned!


## Description

We aim to solve the following constrained model-based Reinforcement Learning(RL) problem.

Our approach involves following three steps:
1. Future state and reward prediction using uncertain dyanmics model
2. Compute Policy Gradient
3. Peform Constrained Gradient Descent to update policy parameters

**Step 1 and 2:** The first two steps are known to be analytically intractable. A popular method, introduced in PILCO, computes analytical formulas for mean and covariance poropagation when the prior distribution is given by a Gaussian and the transition dynamics is given by a gaussian process with a gaussian kernel. We instead use Unscented Transform to propagate states to the future. Depending on number of soigma points employed, we can maintain mean and covariances or even higher order moments of the distribution. Propagting finite number of particles (sigma points) through state-dependent uncertainty model though requires increase in number of sigam points to be able to represent the distributions and this leads to an explosion that is undesirable. Therefore, we introduce differentiable sigma point expansion and compression layer based on moment matching that allows us to keep the algorithm scalable.

**Step 3:** We use Seqential Quadratic Programming type of update to use policy gradients in a way that help maintain constraints that were already satisfied by current policy. If current policy is unable to satisfy a constraint, then reward is designed to reduce the infeasibility margin of this unsatisfiable constraint.  

## CartPole Swingup
In our first example, we randomly initialize the parameters of policy and then try to learn parameters online (in receding horizon fashion) that stabilizes the pole in upright position. Only a horizontal force on the cart can be applied. Only an uncertain dynamics model is available to the system. We run our algorithm for unconstrained as well as constrained cart position.

Unconstrained
[https://user-images.githubusercontent.com/19849515/192346260-4f0c70e6-17d6-4ad0-a211-bd56c90e54b2.mp4](hello)

## CBF tuning for Leader-Follower

|  | No Adaptation | With adaptation |
| --------------| -------------------| -----------------|
| No input bound | ![1](https://github.com/hardikparwana/FORESEE/blob/main/no_adapt_no_bound.gif) | ![2](https://github.com/hardikparwana/FORESEE/blob/main/adapt_no_bound.gif) |
| With input bounds | ![3](https://github.com/hardikparwana/FORESEE/blob/main/no_adapt_with_bound.gif) | ![4](https://github.com/hardikparwana/FORESEE/blob/main/adapt_with_bound.gif)



## Experiments for CBF policy based Leader Follower
We also perform experiments with 2 AION R1 UGV rovers. The leader robot is moved manually by the user. Gaussian Process(GP) is used to learn the motion of leader as a function of time based on past observations. The GP dynamics model is then passed on to the follower that uses this model to predict future and apply our proposed algorithm.
