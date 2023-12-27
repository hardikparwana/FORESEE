import scipy
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

from getMPC_vars import *
from getModelParams import *
from borderAdjustment import *
from robot_models.car2D import *
import jax_cosmo
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline as splinify

# track_raw = loadmat('track2.mat')
track_raw = loadmat('/home/hardik/Desktop/Research/rover_groundstation_ros2_jumpstart/colcon_ws/src/FORESEE/utils/track2.mat')["track2"]

# CarModel = 'ORCA';
CarModel = 'FullSize';

MPC_vars = getMPC_vars(CarModel);
# print(MPC_vars)
ModelParams = getModelParams(MPC_vars["ModelNo"]);

# print(ModelParams)
print(track_raw)
track2 = {
    "center": track_raw["center"][0,0],
    "inner": track_raw["inner"][0,0],
    "outer": track_raw["outer"][0,0]
}

# shrink track by half of the car widht plus safety margin
safteyScaling = 1.5
track, track2 = borderAdjustment(track2,ModelParams,safteyScaling)

fig, ax = plt.subplots()
plt.plot(track["center"][0,:], track["center"][1,:], 'b--')
plt.plot(track["inner"][0,:], track["inner"][1,:], 'k')
plt.plot(track["outer"][0,:], track["outer"][1,:], 'k')

# Compute function of borders
# border_center = splinify( track["center"][0,:], track["center"][1,:], k=3 )
# border_top = splinify( track["center"][0,:], track["center"][1,:], k=3 )
# border_bottom = splinify( track["center"][0,:], track["center"][1,:], k=3 )

# plt.plot(track2["center"][0,:], track2["center"][1,:], 'k--')
# plt.plot(track2["inner"][0,:], track2["inner"][1,:], 'r')
# plt.plot(track2["outer"][0,:], track2["outer"][1,:], 'r')

# Initialize dynamics function
car_state_dot = lambda x, u: car2D_dynamics( x, u, CarModel )

x0 = track["center"][0,0]
y0 = track["center"][1,0]
x1 = track["center"][0,1]
y1 = track["center"][1,1]
heading = np.arctan2( y1 - y0, x1 - x0 )


dt = 0.05
robot = Car2D( np.array([x0, y0, heading, 0, 0, 0]) ,dt, ax, CarModel )

plt.show()