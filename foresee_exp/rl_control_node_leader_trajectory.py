# Run this on terminal once
# export PYTHONPATH=/root/px4_ros_com_ros2/src/robot-framework-py/:${PYTHONPATH}

import queue
from dasc_robots.robot import Robot
from dasc_robots.ros_functions import *
import rclpy
from rclpy.node import Node
import numpy as np
import threading
import time
from rclpy.executors import MultiThreadedExecutor

import torch
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
from utils.unicycle import *
from utils.gp_utils import *
from utils.ut_utils import *
from utils.utils_generic import *

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

def wrap_angle_numpy(angle):
    if angle > np.pi:
        angle = angle - 2 * np.pi
    if angle < -np.pi:
        angle = angle + 2 * np.pi
    return angle

###############################################

class FORESEE(Node):

    def __init__(self, robots):
        super().__init__('minimal_publisher')
        self.robots = robots
        
        self.publisher_ = self.create_publisher(String, 'topic', 10)

        # Observer
        self.leader_pose = np.array([0,0,0]).reshape(-1,1)
        self.leader_pose_previous = np.array([0,0,0]).reshape(-1,1)
        self.observer_init = False

        # Control Parameters
        self.i = 0
        self.controller_k_torch = torch.tensor(1.0, dtype=torch.float)
        self.controller_alpha_torch = torch.tensor(np.array([0.3,0.3,0.3]), dtype=torch.float)
        self.t0 = robots[1].get_current_timestamp_us() / 1000000.0

        # Estimator
        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
        # self.train_x = np.array( [ 0, 0, 1.0, 0.0 ] ).reshape(1,-1)
        self.train_x = np.array( [ 0 ] ).reshape(1,-1)
        self.train_y = np.array( [ 0, 0 ] ).reshape(1, -1)
        self.gp = MultitaskGPModel(torch.tensor(self.train_x, dtype=torch.float), torch.tensor(self.train_y, dtype=torch.float), self.likelihood, num_tasks=2)
        self.gp.eval()
        self.likelihood.eval()
        self.estimator_init = False
        # self.queue = queue()     

        # RL parameter
        self.lr_rate = 0.05#0.05

        ###### Callbacks ############

        # Find and send Control for rover 7
        # self.timer_period_control = 0.05  # seconds
        # self.timer_control = self.create_timer(self.timer_period_control, self.control_callback)

        # Move the leader
        self.timer_period_leader_control = 0.05  # seconds
        self.timer_leader_control = self.create_timer(self.timer_period_leader_control, self.leader_control_callback)

        # Store Observed Data
        self.timer_period_leader_observer = 0.05
        self.timer_observer = self.create_timer( self.timer_period_leader_observer, self.observer_callback )

        # GP Fit
        self.timer_period_leader_estimator = 0.2
        self.timer_estimator = self.create_timer( self.timer_period_leader_estimator, self.estimator_callback )

        # Update Controller
        # self.timer_period_rl = 0.4  # seconds
        # self.timer_rl = self.create_timer(self.timer_period_rl, self.rl_callback)
        # self.dt_outer = torch.tensor(0.1, dtype=torch.float)
        # self.H = 10 

    def position_controller(self, leader_pos, leader_yaw, waypoint):
        diff = waypoint - leader_pos
        desired_heading = np.arctan2( diff[1]/np.linalg.norm(diff), diff[0]/np.linalg.norm(diff) )
        print(f"desired heading:{desired_heading}, yaw:{leader_yaw}")
        actual_heading = leader_yaw
        heading_error = wrap_angle_numpy( desired_heading - actual_heading )
        omega = 3.0 * heading_error
        vx = 4.0 * np.cos(heading_error) * np.linalg.norm(diff)
        return vx, omega

    def leader_control_callback(self):

        waypoint1 = np.array([ -1.45, 1.250 ])
        t1 = 3.0
        waypoint2 = np.array([ -2.25, -1.169 ])
        t2 = 3.0
        waypoint3 = np.array([ 0.7, -1.74 ])
        t3 = 3.0
        waypoint4 = np.array([ 2.056, 0.534 ])
        t4 = 3.0

        leader_pos = self.robots[1].get_world_position()
        leader_pos = leader_pos[0:2]
        leader_quat = self.robots[1].get_body_quaternion()
        leader_yaw = 2.0 * np.arctan2( leader_quat[3], leader_quat[0] )
        t = np.float32( self.robots[1].get_current_timestamp_us() / 1000000.0 - self.t0 )

        if t < t1:
            vx, omega = self.position_controller( leader_pos, leader_yaw, waypoint1 )
        elif t < t2:
            vx, omega = self.position_controller( leader_pos, leader_yaw, waypoint2 )
        elif t < t3:
            vx, omega = self.position_controller( leader_pos, leader_yaw, waypoint3 )
        elif t < t4:
            vx, omega = self.position_controller( leader_pos, leader_yaw, waypoint4 )
        else:
            omega = np.pi/6
            R = 0.8
            vx = omega * R

        # vx = 0
        # omega = 0

        vx = np.clip( vx, -1.3, 1.3 )
        omega = np.clip( omega, -3.0, 3.0 )

        
        # R = 0.8
        # omega = np.pi/6
        # x = R * np.cos( omega * t ) - 1.0
        # y = R * np.sin( omega * t )
        # vx = omega * R 

        print(f"t: {t}, x:{vx}, omega:{omega}")
        # self.robots[1].command_position( np.array([x, y, 0, 0, 0]) )
        # vx = 0
        # omega = 0
        self.robots[1].command_velocity(np.array([0, vx, 0, 0, omega]) )

    def observer_callback(self):
        dt = 0.05

        self.leader_pose_previous = np.copy( self.leader_pose )

        leader_pos = self.robots[1].get_world_position()
        leader_quat = self.robots[1].get_body_quaternion()
        leader_yaw = 2.0 * np.arctan2( leader_quat[3], leader_quat[0] )
        self.leader_pose = np.append( leader_pos[0:2], leader_yaw  ).reshape(-1,1)

        if not self.observer_init:
            self.leader_pose_previous = np.copy( self.leader_pose )
            self.observer_init = True

        diff = self.leader_pose - self.leader_pose_previous
        # print(f"diff:{diff}")
        diff[2,0] = wrap_angle_numpy( diff[2,0] )
        new_y = diff / self.timer_period_leader_observer

        # new_x = np.array([ leader_pos[0], leader_pos[1], np.cos( leader_yaw ), np.sin( leader_yaw ) ])
        new_x = np.float32( self.robots[1].get_current_timestamp_us() / 1000000.0 - self.t0 )

        self.train_x = np.append( self.train_x,  new_x.reshape(1,-1), axis = 0 )
        self.train_y = np.append( self.train_y,  new_y[0:2].reshape(1,-1), axis = 0 )

        # print(f"train x:{ self.train_x }, y:{ self.train_y }")

    def estimator_callback(self):
 
        data_horizon = 3
        num_data = int( data_horizon / self.timer_period_leader_observer )
        
        train_x = np.copy( self.train_x[-num_data:, :] )
        train_y = np.copy( self.train_y[-num_data:, :] )

        idxs = np.random.choice(np.shape( train_x )[0], min( np.shape(train_x)[0], 100 ), replace=False)
        # self.train_x = train_x[idxs, :]
        # self.train_y = train_y[idxs, :]

        # print(f"num_data: {num_data}, train_x size: {np.shape(train_x)[0]} ")

        # print(f" size:{np.shape( train_x )[0]}, train_x:{train_x} , selected:{idxs}, selected: {train_x[idxs, :]}")
        # self.train_x = np.array( [ 0, 0, 1.0, 0.0 ] ).reshape(1,-1)

        # Keep
        # self.gp.set_train_data( torch.tensor( self.train_x, dtype = torch.float ), torch.tensor( self.train_y, dtype=torch.float ), strict = False)
        #  self.gp.set_train_data( train_x_torch, train_y_torch, strict = False)
        # train_gp(self.gp,self.likelihood, train_x_torch, train_y_torch, training_iterations = 10)

        train_x_torch = torch.tensor(train_x[idxs, :], dtype=torch.float)
        train_y_torch = torch.tensor(train_y[idxs, :], dtype=torch.float)

        likelihood_temp = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)    
        gp_temp = MultitaskGPModel(train_x_torch, train_y_torch, likelihood_temp, num_tasks=2)
        for param_name, param in self.gp.named_parameters():
            gp_temp.initialize( **{ param_name:torch.clone( param ) } )

        train_gp(gp_temp,likelihood_temp, train_x_torch, train_y_torch, training_iterations = 10)
        
        # Copy GP back from temp to original
        self.gp.set_train_data( train_x_torch, train_y_torch, strict = False)
        for param_name, param in gp_temp.named_parameters():
            self.gp.initialize( **{ param_name:torch.clone( param ) } )

        self.estimator_init = True

        # self.get_logger().info('GP Training done')
    
    def rl_callback(self):

        if not self.estimator_init:
            return
        # print("hello0")
        alpha_torch_temp = torch.clone( self.controller_alpha_torch )
        # print("hello01")
        alpha_torch_numpy = alpha_torch_temp.detach().numpy()
        # print("hello02")
        alpha_torch = torch.tensor( alpha_torch_numpy, dtype=torch.float, requires_grad=True )
        # print("hello03")
        k_torch_temp = torch.clone( self.controller_k_torch )
        k_torch_numpy = k_torch_temp.detach().numpy()
        k_torch = torch.tensor( k_torch_numpy, dtype=torch.float, requires_grad=True )
        # print("hello")
        # print(f"alpha:{alpha_torch}, k_torch:{}")

        # Initialize sigma points
        f_pos = self.robots[0].get_world_position()
        f_quat = self.robots[0].get_body_quaternion()
        f_yaw = 2.0 * np.arctan2( f_quat[3],f_quat[0] )
        f_pose = torch.tensor( np.array( [ f_pos[0], f_pos[1], f_yaw ] ).reshape(-1,1), dtype=torch.float )
        follower_states = [ f_pose ]
        # print("hello1")
        l_pos = self.robots[1].get_world_position()
        l_quat = self.robots[1].get_body_quaternion()
        l_yaw = 2.0 * np.arctan2( l_quat[3],l_quat[0] )
        l_pose = torch.tensor( np.array([ l_pos[0], l_pos[1] ]).reshape(-1,1), dtype=torch.float )
        # l_pose = torch.tensor( np.array( [ l_pos[0], l_pos[1], l_yaw ] ).reshape(-1,1), dtype=torch.float )
        # print("hell2")
        prior_leader_states, prior_leader_weights = initialize_sigma_points2_JIT(l_pose)
        leader_states = [prior_leader_states]
        leader_weights = [prior_leader_weights]

        reward = torch.tensor([0],dtype=torch.float)

        tp = torch.tensor( np.float64( self.robots[1].get_current_timestamp_us() / 1000000.0 - self.t0 ), dtype = torch.float )
        
        for i in range(self.H):       
            # print(f"leader size:{leader_states[i]}")
            
            leader_xdot_states, leader_xdot_weights = traced_sigma_point_expand_JIT( follower_states[i], leader_states[i], leader_weights[i], tp, self.gp )
            
            leader_states_expanded, leader_weights_expanded = traced_sigma_point_scale_up5_JIT( leader_states[i], leader_weights[i])#leader_xdot_weights )
            # print(f"i:{i}, tp:{tp}: l shape:{leader_states[i].shape}, xdot:{leader_xdot_states.shape}, ex:{leader_states_expanded.shape}")
            A, B = traced_unicycle_SI2D_UT_Mean_Evaluator( follower_states[i], leader_states_expanded, leader_xdot_states, leader_weights_expanded, k_torch, alpha_torch ) 
                
            leader_mean_position = traced_get_mean_JIT( leader_states[i], leader_weights[i] )  
                
            u_ref = traced_unicycle_nominal_input_tensor_jit( follower_states[i], leader_mean_position )
            
            solution, deltas = cbf_controller_layer( u_ref, A, B )
            # print("solution", solution)

            follower_states.append( traced_unicycle_step_torch( follower_states[i], solution, self.dt_outer ) )        
            leader_next_state_expanded = leader_states_expanded + leader_xdot_states * self.dt_outer
            
            leader_next_states, leader_next_weights = traced_sigma_point_compress_JIT( leader_next_state_expanded, leader_xdot_weights )        
            leader_states.append( leader_next_states ); leader_weights.append( leader_next_weights )
            
            # Get reward for this state and control input choice = Expected reward in general settings
            reward = reward + traced_unicycle_reward_UT_Mean_Evaluator_basic( follower_states[i+1], leader_states[i+1], leader_weights[i+1])
            
            tp = tp + self.dt_outer
        reward.sum().backward(retain_graph=True)

        alpha_grad = getGrad( alpha_torch, l_bound = -2, u_bound = 2 )
        k_grad = getGrad( k_torch, l_bound = -2, u_bound = 2 )

        # print(f"reward:{reward}, alpha_grad:{alpha_grad}, k_grad:{k_grad} ") 

        self.controller_alpha_torch = torch.clip(self.controller_alpha_torch - self.lr_rate * alpha_grad, min = 0, max = None )
        self.controller_k_torch = torch.clip( self.controller_k_torch - self.lr_rate * k_grad, min = 0, max = None)    
        alpha_cur = self.controller_alpha_torch.detach().numpy()
        # self.get_logger().info( 'Reward %f k %f alpha %f %f %f' % (reward.item(), self.controller_k_torch.item(), self.controller_alpha_torch[0].item(), self.controller_alpha_torch[1].item(), self.controller_alpha_torch[3].item() ) ) 
        self.get_logger().info( 'Reward and params %f %f %f %f %f' % (reward, self.controller_k_torch.item(), alpha_cur[0], alpha_cur[1], alpha_cur[2]) )
        # print( f" reward:{reward}, alpha_grad:{alpha_grad}, k_grad:{k_grad}" )

    def control_callback(self):
        msg = String()
        msg.data = 'Hello World: %d' % self.i
        self.publisher_.publish(msg)
        # self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1

        pos = self.robots[0].get_world_position()
        quat = self.robots[0].get_body_quaternion()
        yaw = 2.0 * np.arctan2( quat[3],quat[0] )
        # print(f" quat0:{quat[0]}, quat1:{quat[3]}, yaw:{yaw*180/np.pi} ")
        pose = np.array( [ pos[0], pos[1], yaw ] )

        leader_pos = self.robots[1].get_world_position()
        leader_quat = self.robots[1].get_body_quaternion()
        leader_yaw = 2.0 * np.arctan2( leader_quat[0], leader_quat[3] )
        leader_pose = leader_pos[0:2] #np.array( [ leader_pos[0], leader_pos[1], leader_yaw ] )
        # print(f"leader pos: { leader_pos }")

        alpha_torch = torch.clone( self.controller_alpha_torch )
        k_torch = torch.clone( self.controller_k_torch )

        if not self.estimator_init:
            leader_pose_dot = torch.tensor( [0,0], dtype=torch.float ).reshape(-1,1)
        else:
            # print(f"using GP")
            cur_t = torch.tensor( np.float64( self.robots[1].get_current_timestamp_us() / 1000000.0 - self.t0 ), dtype = torch.float )
            # print(f"from controller input_x:{cur_t}, gp x:{self.gp.train_inputs}")
            prediction = self.gp( cur_t.reshape(1,-1) )
            leader_pose_dot = prediction.mean.reshape(-1,1)

        A, b = unicycle_SI2D_clf_cbf_fov_evaluator(torch.tensor(pose, dtype=torch.float).reshape(-1,1), torch.tensor( leader_pose, dtype=torch.float).reshape(-1,1), leader_pose_dot, k_torch, alpha_torch)
        u_ref = unicycle_nominal_input_tensor_jit( torch.tensor(pose, dtype=torch.float).reshape(-1,1), torch.tensor( leader_pose, dtype=torch.float ).reshape(-1,1) )
        # u_ref = torch.tensor([0,0], dtype=torch.float).reshape(-1,1)

        # print(f"A:{A}")
        # print(f"Au_refb:{ A @ u_ref + b }")

        solution, deltas = cbf_controller_layer( u_ref, A, b )
        solution = solution.detach().numpy()
        vx = solution[0,0]
        wz = solution[1,0]

        vx = np.clip( vx, -0.6, 0.6 )
        wz = np.clip( wz, -2.0, 2.0 )

        # print(f"Commanding velocity u:{vx}, omega:{wz}: yaw:{yaw}, u_ref:{ u_ref.T }, alpha:{self.controller_alpha_torch}, k:{self.controller_k_torch}")
        self.robots[0].command_velocity( np.array([0,vx,0,0,wz]) )

        # vx = 0.0
        # wz = 0.0
        # self.robots[0].command_velocity( np.array([0,vx,0,0,wz]) )
        # self.robots[1].command_velocity( np.array([0,vx,0,0,wz]) )


def main(args=None):
    rclpy.init(args=args)
    # Spin in a separate thread
    # node = rclpy.create_node('minimal_publisher')
    # thread = threading.Thread(target=rclpy.spin, args=(node, ), daemon=True)
    # thread.start()
    # rate = node.create_rate(30)

    ## Robot Initialization ####
    ros_init("rl_fov_following")

    robot7 = Robot("rover7", 7)
    robot2 = Robot("rover2", 2)
    print("Robot Initialized")

    robot7.init()
    robot2.init()

    robots = [robot7, robot2]
    threads = start_ros_nodes(robots)

    time.sleep(2.0)

    robot7.set_command_mode( 'velocity' )
    robot2.set_command_mode( 'velocity' )

    vx = 0.0
    wz = 0.0
    for i in range(100):
        robot7.command_velocity( np.array([0,vx,0,0,wz]) )
        robot2.command_velocity( np.array([0,vx,0,0,wz]) )
        time.sleep(0.05)#rate.sleep()

    robot7.cmd_offboard_mode()
    robot7.arm()
    robot2.cmd_offboard_mode()
    robot2.arm()
    
    #############################

    control_node = FORESEE(robots)

    # executor = MultiThreadedExecutor(num_threads=5)
    # executor.add_node(control_node)
    # executor.spin()

    rclpy.spin(control_node)

    # executor.shutdown()
    control_node.destroy_node()
    rclpy.shutdown()

    print("End of Run")