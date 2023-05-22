Copyright (c) 2020 Mitsubishi Electric Research Laboratories (MERL).

All rights reserved. The software, documentationand/or data in this file is provided on an "as is" basis, 
and MERL has no obligations to provide maintenance, support, updates, enhancements or modifications.  

MERL specifically disclaims any warranties, including, but not limited to, the implied warranties of merchantability 
and fitness for anyparticular purpose.In no event shall MERL be liable to any party for direct, indirect, special, 
incidental, or consequential damages, including lost profits, arising out of the use of this software and its documentation, 
even if MERL has been advised of the possibility of such damages.

As more fully described in the license agreement that was required in order to download this software, documentation 
and/or data, permission to use, copy and modify this software without fee is granted, but only for educational, 
research and non-commercial purposes.

# MC-PILCO

This package implements the Model-based Reinforcement Learning algorithm called Monte Carlo Probabilistic Inference for Learning and COntrol (MC-PILCO), for modeling and control of dynamical system. The algorithm relies on Gaussian Processes (GPs) to model the system dynamics and on a Monte Carlo approach to estimate the policy gradient during optimization. The Monte Carlo approach is shown to be effective for policy optimization thanks to a proper cost function shaping and use of dropout. The possibility of using a Monte Carlo approach allows a more flexible framework for Gaussian Process Regression that leads to more structured and more data efficient kernels.
The algorithm is also extended to work for Partially Measurable Systems and takes the name of MC-PILCO-4PMS.
Please see [1] for a detailed description of the algorithm.
The code is implemented in python3 and reproduces all the simulation examples in the related publication, namely, multiple ablation studies and the solution of a cart-pole system swing-up (available both in a python simulated environment and in the physic engine MuJoCo), a trajectory controller for a UR5 (implemented in Mujoco). The results can be reproduced with statistical value via Monte Carlo simulations.
The user has the possibility to add his own python system or Mujoco Environment and solve it with MC-PILCO or MC-PILCO-4PMS.

Please refer to the guide for a more detailed explanation of the code base.

## Dependencies
- [PyTorch 1.4 or superior] (https://pytorch.org/)
- [NumPy] (https://numpy.org/)
- [Matplotlib] (https://matplotlib.org/)
- [Pickle] (https://docs.python.org/3/library/pickle.html)
- [Argparse] (https://docs.python.org/3/library/argparse.html)
- [gpr_lib] is provided courtesy of Alberto Dalla Libera with permission to redistribute as part of this software package.

# Optional
- [MuJoCo 2.00](http://www.mujoco.org/)
- [MuJoCo-Py] (https://github.com/openai/mujoco-py) 
- [Gym] (http://gym.openai.com/)

## Installation
1. Download the source code.
2. Create a python environment with the following packages: PyTorch, NumPy, Matplotlib, Pickle, Argparse.
3. If you want to test the code on MuJoCo environments, make sure to have also MuJoCo_py and Gym libraries.

## Usage
Please refer to the guide 'MC_PILCO_Software_Package.pdf'


## Examples
Inside 'mc_pilco' folder:

- Run '$ python test_mcpilco_cartpole.py' to test MC-PILCO in the cartpole swing-up task (GP model with squared-exponential+polynomial kernel).
- Run '$ python test_mcpilco_cartpole_rbf_ker.py' to test MC-PILCO in the cartpole swing-up task (GP model with squared-exponential kernel).
- Run '$ python test_mcpilco_cartpole_multi_init.py' to test MC-PILCO in the cartpole swing-up task stating from two separate possible initial cart positions.
- Run '$ python test_mcpilco4pms_cartpole.py' to test MC-PILCO4PMS in the cartpole swing-up task when considering the presence of sensors and state estimation.
- Run '$ python test_mcpilco_cartpole_mujoco.py' to test MC-PILCO in the cartpole swing-up task in MuJoCo.
- Run '$ python test_mcpilco_ur5_mujoco.py' to use MC-PILCO to learn a joint-space controller for a UR5 robot arm in MuJoCo.


## Citing
If you use this package, please cite the following paper: Amadio, F.et al. Model-Based Policy Search Using Monte Carlo Gradient Estimation with RealSystems Application. arXiv preprint (2021). 2101.12115.
