# FORESEE(4C): Foresee for Certified Constrained Control

This repository implements our TRO 2023 submission on 
**FORESEE: Prediction with Expansion-Compression Unscented Transform for Online Policy Optimization**

Authors: Hardik Parwana and Dimitra Panagou, University of Michigan

Note: This repo is under development. While all the relevant code is present, we will work on making it more readable and customizable soon! Stay Tuned! Please raise an issue or send me an email if you run into issues before this documentation is ready. I am happy to help adapt this algorithm to suit your needs!

## State Prediction with Stochastic (Uncertain) Dynamics

We propose a new method for numerical prediction of future states under generic stochastic dynamical systems, i.e, nonlinear dynamical systems with state-dependent disturbance. We use a sampling-based approach, namely Unscented Transform to do so. Previous approaches with UT only had state-independent uncertainty. The presence of state-dependent uncertainty necessiates to increase the number of samples, aka sigma points in UT, to grow with time. This leads to an unscalable approach. Therefore we propose Expansion-Contraction layers where
- Expansion Layer: maps each sigma point to multiple sigma points according to the disturbance level at that point. This leads to an increase in the total number of points.
- Compression Layer: uses moment matching to find a smaller number of sigma points that have the same moments as the expanded points
A sequence of Expansion-Compression layer is used for multi-step prediction. Our layers are completely differentiable and hence can be used for policy optimization.  Finally, we also propose an online gradient descent scheme for policy optimization.

![uncertainty_propagation_tro](https://github.com/hardikparwana/FORESEE/assets/19849515/264e06ee-5edf-4393-b71e-35ad73617086)


## Trajectory Optimization for Stochastic (Uncertain) Dynamics

| Constraint Satisfaction in Expectation | Constraint Satisfactyion with Confidence Interval | 
| -------------------| -----------------|
| ![mpc_ss_mean_obj1](https://github.com/hardikparwana/FORESEE/assets/19849515/ca992d95-78e7-42d7-9b59-d441f5dae56a) | ![mpc_ss_ci_obj1_v2](https://github.com/hardikparwana/FORESEE/assets/19849515/7c850479-8406-4a60-9913-c5d471cfc534) |


## CBF tuning for Leader-Follower
Th objective for the follower is to keep leader inside the field-of-view and, preferably, at the center. Adaptation is needed as depending on the pattern of leader's movement, different policy parameters perform better. The policy here is a CBF-CLF-QP that is to be satisfied in expectation when dynamics is uncertain. The first sim shows the performance of default parameters. The second one shows improvemwnt with our adaptation running online. Results change significantly when control input bounds are imposed. The QP does not even exhibit a solution after some time when default parameters are used and the simulation ends. The proposed algorithm is able toadapt parameters online to continuously satisfy input bounds. The prediction horizon is taken to be 20 time steps.

|  | No Adaptation | With adaptation |
| --------------| -------------------| -----------------|
| No input bound | ![no_adapt_no_bound](https://user-images.githubusercontent.com/19849515/192348004-6dcbf70f-2db5-49dd-9f4f-04370dc028e4.gif) | ![adapt_no_bound](https://user-images.githubusercontent.com/19849515/192348165-5f6fbaf4-81e1-4cd6-893f-d5f763ea9cbc.gif) |
| With input bounds | ![no_adapt_with_bound](https://user-images.githubusercontent.com/19849515/192348231-a921fa36-6198-45b5-94c2-80ae87ab8b39.gif) | ![adapt_with_bound](https://user-images.githubusercontent.com/19849515/192348335-448600b8-042b-4bb5-8c9f-17e654584336.gif)




## Dependencies

For Pytorch code, the following dependencies are required:
- Python version 3.8
- numpy==1.22.3 
- gym==0.26.0 
- gym-notices==0.0.8 
- gym-recording==0.0.1 
- gpytorch==1.8.1 
- torch==1.12.1 ( PyTorch's JIT feature was used to speed up computations wherever possible.)
- pygame==2.1.2 
- gurobipy==9.5.1 
- cvxpy==1.2.0 
- cvxpylayers==0.1.5 
- cartpole==0.0.1

For JAX code, the following dependencies are required
- Python 3.11
- numpy==1.22.3 matplotlib sympy argparse scipy==1.10.1
- cvxpy==1.2.0 cvxpylayers==0.1.5 gym==0.26.0 gym-notices==0.0.8 gym-recording==0.0.1 moviepy==1.0.3 cyipopt==1.2.0 jax==0.4.13 jaxlib==0.4.11 gpjax==0.5.9 optax==0.1.4 jaxopt
- diffrax==0.3.0
- pygame==2.3.0

  We also provide a Dockefile in the `docker_files` and 311_requirements.txt file for Python3.11 dependencies that can be used to run JAX examples


Note that you will also have to add a source directory to PYTHONPATH as there is no setup.py file provided yet. Note that the relevant gym environment for cartpole simulation is already part of this repo. This was done to change the discreet action space to a continuous action space and to change the physical properties of the objects.

## Running the Code
We will be adding interactive jupyter notebooks soon! In the meantime, try out our scripts (comments to be addded soon!)
To run the leader-follower example, run
```
python leader_follower/UT_RL_2agent_jit_simple.py
```
For cartpole, run
```
python cartpole/cartpole_UTRL_simple_offline_constrained.py
```

## Description

We aim to solve the following constrained model-based Reinforcement Learning(RL) problem.

Our approach involves following three steps:
1. Future state and reward prediction using uncertain dyanmics model
2. Compute Policy Gradient
3. Peform Constrained Gradient Descent to update policy parameters

**Step 1 and 2:** The first two steps are known to be analytically intractable. A popular method, introduced in PILCO, computes analytical formulas for mean and covariance poropagation when the prior distribution is given by a Gaussian and the transition dynamics is given by a gaussian process with a gaussian kernel. We instead use Unscented Transform to propagate states to the future. Depending on number of soigma points employed, we can maintain mean and covariances or even higher order moments of the distribution. Propagting finite number of particles (sigma points) through state-dependent uncertainty model though requires increase in number of sigam points to be able to represent the distributions and this leads to an explosion that is undesirable. Therefore, we introduce differentiable sigma point expansion and compression layer based on moment matching that allows us to keep the algorithm scalable.

**Step 3:** We use Seqential Quadratic Programming type of update to use policy gradients in a way that help maintain constraints that were already satisfied by current policy. If current policy is unable to satisfy a constraint, then reward is designed to reduce the infeasibility margin of this unsatisfiable constraint.  

## CartPole Swingup
In our first example, we randomly initialize the parameters of policy and then try to learn parameters online (in receding horizon fashion) that stabilizes the pole in upright position. Only a horizontal force on the cart can be applied. Only an uncertain dynamics model is available to the system. We run our algorithm for unconstrained as well as constrained cart position. The prediction horizon is taken to be 30 time steps.

- Unconstrained: X axis range (0,12) in animation

[https://user-images.githubusercontent.com/19849515/192346260-4f0c70e6-17d6-4ad0-a211-bd56c90e54b2.mp4](https://user-images.githubusercontent.com/19849515/192346260-4f0c70e6-17d6-4ad0-a211-bd56c90e54b2.mp4)

- Constrained: X axis range (-1.5,1.5) in animation

[https://user-images.githubusercontent.com/19849515/192346448-6c2d450f-03a1-4d46-9f1b-ab653c9f1902.mp4](https://user-images.githubusercontent.com/19849515/192346448-6c2d450f-03a1-4d46-9f1b-ab653c9f1902.mp4)




## Experiments for CBF policy based Leader Follower
We also perform experiments with 2 AION R1 UGV rovers. The leader robot is moved manually by the user. Gaussian Process(GP) is used to learn the motion of leader as a function of time based on past observations. The GP dynamics model is then passed on to the follower that uses this model to predict future and apply our proposed algorithm.

Each AION UGV is equipped with pixhawk board running custom px4 firmware for unicycle type of velocity control.
