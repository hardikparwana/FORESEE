# Run this on terminal once
# export PYTHONPATH=/root/px4_ros_com_ros2/src/robot-framework-py/:${PYTHONPATH}

from dasc_robots.robot import Robot
from dasc_robots.ros_functions import *
import rclpy
from rclpy.node import Node
import numpy as np
import threading
import time

import torch
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
from utils.unicycle import *

from std_msgs.msg import String

####  CBF Controller ###################

u1_max = 2.5#2.0# 3.0
u2_max = 4.0
u1 = cp.Variable((2,1))
delta = cp.Variable(1)
delta_u = cp.Variable((2,1))

u1_ref = cp.Parameter((2,1),value = np.zeros((2,1)) )
num_constraints1 = 1 + 3
A1 = cp.Parameter((num_constraints1,2),value=np.zeros((num_constraints1,2)))
b1 = cp.Parameter((num_constraints1,1),value=np.zeros((num_constraints1,1)))
const1 = [A1 @ u1 + b1 + delta * np.array([1,0,0,0]).reshape(-1,1) >= 0]
const1 += [ cp.abs( u1[0,0] )  <= u1_max + delta_u[0,0] ]
const1 += [ cp.abs( u1[1,0] )  <= u2_max + delta_u[1,0] ]
const1 += [ delta_u[0,0] >= 0 ]
const1 += [ delta_u[1,0] >= 0 ]
objective1 = cp.Minimize( cp.sum_squares( u1 - u1_ref  ) + 100*cp.sum_squares(delta) + 10000 * cp.sum_squares( delta_u ) )
cbf_controller = cp.Problem( objective1, const1 )
assert cbf_controller.is_dpp()
solver_args = {
            'verbose': False
        }
cbf_controller_layer = CvxpyLayer( cbf_controller, parameters=[ u1_ref, A1, b1 ], variables = [u1, delta_u] )

########################################

class FORESEE(Node):

    def __init__(self, robots):
        super().__init__('minimal_publisher')
        self.robots = robots
        
        self.publisher_ = self.create_publisher(String, 'topic', 10)

        timer_period_control = 0.05  # seconds
        self.timer_control = self.create_timer(timer_period_control, self.control_callback)
        
        self.i = 0

        self.controller_k_torch = torch.tensor(1.0, dtype=torch.float)
        self.controller_alpha_torch = torch.tensor(np.array([0.8,0.8,0.8]), dtype=torch.float)
        

    def control_callback(self):
        msg = String()
        msg.data = 'Hello World: %d' % self.i
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1

        pos = np.array([0,0,0]) #self.robots[0].get_world_position()
        quat = np.array([1,0,0,0]) #self.robots[0].get_body_quaternion()
        yaw = 2.0 * np.arctan2( quat[3],quat[0] )
        # print(f" quat0:{quat[0]}, quat1:{quat[3]}, yaw:{yaw*180/np.pi} ")
        pose = np.array( [ pos[0], pos[1], yaw ] )

        leader_pos = np.array([1,0,0]) #self.robots[1].get_world_position()
        leader_quat = np.array([1,0,0,0]) #self.robots[1].get_body_quaternion()
        leader_yaw = 2.0 * np.arctan2( leader_quat[0], leader_quat[3] )
        leader_pose = leader_pos[0:2] #np.array( [ leader_pos[0], leader_pos[1], leader_yaw ] )

        leader_pose_dot = torch.tensor( [0,0], dtype=torch.float ).reshape(-1,1)

        A, b = unicycle_SI2D_clf_cbf_fov_evaluator(torch.tensor(pose, dtype=torch.float).reshape(-1,1), torch.tensor( leader_pose, dtype=torch.float).reshape(-1,1), leader_pose_dot, self.controller_k_torch, self.controller_alpha_torch)
        u_ref = unicycle_nominal_input_tensor_jit( torch.tensor(pose, dtype=torch.float).reshape(-1,1), torch.tensor( leader_pose, dtype=torch.float ).reshape(-1,1) )

        solution, deltas = cbf_controller_layer( u_ref, A, b )
        solution = solution.detach().numpy()
        vx = solution[0,0]
        wz = solution[1,0]

        print(f"Commanding velocity u:{vx}, omega:{wz}: yaw:{yaw}")
        self.robots[0].command_velocity( np.array([0,vx,0,0,wz]) )


def main(args=None):
    rclpy.init(args=args)
    # Spin in a separate thread
    # node = rclpy.create_node('minimal_publisher')
    # thread = threading.Thread(target=rclpy.spin, args=(node, ), daemon=True)
    # thread.start()
    # rate = node.create_rate(30)

    ## Robot Initialization ####
    ros_init("fov_following")

    robot7 = Robot("rover7", 7)
    robot2 = Robot("rover2", 2)
    print("Robot Initialized")

    robot7.init()
    robot2.init()

    robots = [robot7, robot2]
    threads = start_ros_nodes(robots)

    robot7.set_command_mode( 'velocity' )

    vx = 0.0
    wz = 2.0
    for i in range(100):
        robot7.command_velocity( np.array([0,vx,0,0,wz]) )
        time.sleep(0.05)#rate.sleep()

    robot7.cmd_offboard_mode()
    robot7.arm()
    
    #############################

    control_node = FORESEE(robots)
    rclpy.spin(control_node)

    control_node.destroy_node()
    rclpy.shutdown()

    print("End of Run")