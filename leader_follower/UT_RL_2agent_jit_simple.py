import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
plt.rcParams.update({'font.size': 10})
plt.rcParams['text.usetex'] = True
from tempfile import TemporaryFile

import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

from robot_models.SingleIntegrator2D import *
from robot_models.Unicycle import *
from robot_models.UnicycleJIT import *

from utils.utils import *
from ut_utils.ut_utilsJIT import *

torch.autograd.set_detect_anomaly(True)

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=torch.jit.TracerWarning) 

outfile = TemporaryFile()
#############################################################
# CBF Controller: centralized
u1_max = 2.5#2.0# 3.0
u2_max = 4.0
u1 = cp.Variable((2,1))
# delta = cp.Variable((4,1))
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

# @torch.jit.script
def cbf_controller_layer_jit( u1_ref, A1, b1 ):
    print(f"u1:{u1_ref}, A:{A1}, b:{b1}")
    u  = cbf_controller_layer(u1_ref, A1, b1)
    print(f"u soln:{u}")
    return u[0]

# u1r = torch.tensor([1.0,1.0]).reshape(-1,1)
# A1 = torch.ones((num_constraints1,2))
# b1 = torch.zeros((num_constraints1,1))
# cbf_controller_layer_jit(u1r, A1, b1)

###############################################################
def initialize_tensors(follower, leader):
    follower.X_torch = torch.tensor( follower.X, requires_grad = True, dtype=torch.float )
    leader.X_torch = torch.tensor( leader.X, requires_grad = True, dtype=torch.float )
    follower.alpha_torch = torch.tensor(follower.alpha, dtype=torch.float, requires_grad=True)
    follower.k_torch = torch.tensor( follower.k, dtype=torch.float, requires_grad = True )
    
def compute_A1_b1_tensor(robotsJ, robotsK, robotsJ_state, robotsK_state, t, noise, control_leader_predict_function = []):
    
    x_dot_k_mean, x_dot_k_cov = control_leader_predict_function( t, noise ) #traced_leader_predict_jit
    # print(f"gp mean: { x_dot_k_mean }, actual_last_xdot: {robotsK.Xdots[:,-1]}")
        
    x_dot_k = x_dot_k_mean.T.reshape(-1,1) #+ cov terms??     
    A1, b1 = unicycle_SI2D_clf_cbf_fov_evaluator(robotsJ_state, robotsK_state, x_dot_k, robotsJ.k_torch, robotsJ.alpha_torch)
   
    return A1, b1

first_run = True
first_generate_sigma_run = True
    
def get_future_reward( follower, leader, t = 0, noise = torch.tensor(0), enforce_input_constraints = False, leader_predict_function = traced_sigma_point_expand_JIT  ):
    # Initialize sigma points for other robots
    follower_states = [torch.clone(follower.X_torch)]        
    prior_leader_states, prior_leader_weights = initialize_sigma_points2_JIT(leader.X_torch)
    leader_states = [prior_leader_states]
    leader_weights = [prior_leader_weights]

    reward = torch.tensor([0],dtype=torch.float)
    tp = t
    
    maintain_constraints = []
    improve_constraints = []    
    
    for i in range(H):       
        
        # t0 = time.time()leader_predict_function
        leader_xdot_states, leader_xdot_weights = leader_predict_function( follower_states[i], leader_states[i], leader_weights[i], torch.tensor(tp), noise ) #traced_sigma_point_expand_JIT
        # print(f"Time 1: {time.time()-t0}")
        
        # t0 = time.time()
        leader_states_expanded, leader_weights_expanded = traced_sigma_point_scale_up5_JIT( leader_states[i], leader_weights[i])#leader_xdot_weights )
        # print(f"Time 2: {time.time()-t0}")
        
        # t0 = time.time()
        A, B = traced_unicycle_SI2D_UT_Mean_Evaluator( follower_states[i], leader_states_expanded, leader_xdot_states, leader_weights_expanded, follower.k_torch, follower.alpha_torch )
        # print(f"Time 3: {time.time()-t0}")    
              
        # t0 = time.time()
        leader_mean_position = traced_get_mean_JIT( leader_states[i], leader_weights[i] )  
        # print(f"Time 4: {time.time()-t0}")
              
        # print(f"leader_mean:{leader_mean_position.T}, follower:{ follower_states[-1].T }")
        u_ref = traced_unicycle_nominal_input_tensor_jit( follower_states[i], leader_mean_position )
        
        # t0 = time.time()
        solution, deltas = cbf_controller_layer( u_ref, A, B )
        # print("solution", solution)
        # print(f"Time 5: {time.time()-t0}")
        if enforce_input_constraints:
            if np.any( deltas.detach().numpy() > 0.01 ):
                # cannot satisfy with input constraints
                improve_constraints.append( -B[0] ) # increase B. reduce -B
                improve_constraints.append( -B[1] ) # increase B. reduce -B
                improve_constraints.append( -B[2] ) # increase B. reduce -B
                improve_constraints.append( -B[3] ) # increase B. reduce -B
                if deltas[0,0] > 0.01:
                    improve_constraints.append( deltas[0,0] )
                if deltas[1,0] > 0.01:
                    improve_constraints.append( deltas[1,0] )
                print(f"Infeasible solution found at :{i}. Will improve first")
                # exit()
                return maintain_constraints, improve_constraints, False, reward
            else:
                temp = A @ solution + B
                # maintain_constraints.append(temp[0] + 0.01)
                maintain_constraints.append(temp[1] + 0.01)
                maintain_constraints.append(temp[2] + 0.01)
                maintain_constraints.append(temp[3] + 0.01)
                # if np.any( temp[1:].detach().numpy() < 0 ):
                #     print("Issue here")
            
        follower_states.append( follower.step_torch( follower_states[i], solution, dt_outer ) )        
        leader_next_state_expanded = leader_states_expanded + leader_xdot_states * dt_outer
        
        # t0 = time.time()
        leader_next_states, leader_next_weights = traced_sigma_point_compress_JIT( leader_next_state_expanded, leader_xdot_weights )        
        # print(f"Time 6: {time.time()-t0}")
        leader_states.append( leader_next_states ); leader_weights.append( leader_next_weights )
        
        # Get reward for this state and control input choice = Expected reward in general settings
        # t0 = time.time()
        reward = reward + traced_unicycle_reward_UT_Mean_Evaluator_basic( follower_states[i+1], leader_states[i+1], leader_weights[i+1])
        # print(f"Time 7: {time.time()-t0}")
        
        tp = tp + dt_outer

    return maintain_constraints, improve_constraints, True, reward
        
################################################################

# Sim Parameters
num_steps = 300#100#50#20#100#50 #100 #200 #200
learn_period = 1#2
gp_training_iter_init = 30
train_gp = False
outer_loop = 2
H = 30# 5
gp_training_iter = 10
d_min = 0.3
d_max = 2.0
angle_max = np.pi/3
num_points = 5
dt_inner = 0.01
dt_outer = 0.05 #0.1
alpha_cbf = 0.3 #1.0#0.1 # 0.5   # Initial CBF
k_clf = 1
num_robots = 1
lr_alpha = 0.05 #0.05
max_history = 100
print_status = False

follower_init_pose = np.array([0,0,np.pi*0.0])
leader_init_pose = np.array([0.4,0])


plot_x_lim = (0,6)  #(0,10)
plot_y_lim = (-3,3) #(-4,4)



def constrained_update( objective, maintain_constraints, improve_constraints, params ) :
    
    
    num_params = 4
    d = cp.Variable((num_params,1))
    
    # Get Performance optimal direction
    try:
        objective.sum().backward(retain_graph = True) 
        k_grad = getGrad(params[0], l_bound = -20.0, u_bound = 20.0 )
        alpha_grad = getGrad(params[1], l_bound = -20.0, u_bound = 20.0 )
        objective_grad = np.append( k_grad.reshape(1,-1), alpha_grad.reshape(1,-1), axis = 1 )
    except:
        objective_grad = np.array([[0,0,0,0]])
    
    # Get constraint improve direction # assume one at a time
    improve_constraint_direction = np.array([0,0,0,0]).reshape(1,-1)
    for i, constraint in enumerate( improve_constraints):
        constraint.sum().backward(retain_graph=True)
        k_grad = getGrad(params[0], l_bound = -20.0, u_bound = 20.0 )
        alpha_grad = getGrad(params[1], l_bound = -20.0, u_bound = 20.0 )
        improve_constraint_direction = improve_constraint_direction +  np.append( k_grad.reshape(1,-1), alpha_grad.reshape(1,-1), axis = 1 )
    
    # Get allowed directions
    N = len(maintain_constraints)
    if N>0:
        d_maintain = np.zeros((N,num_params))#cp.Variable( (N, num_params) )
        constraints = []
        for i, constraint in enumerate(maintain_constraints):
            constraint.sum().backward(retain_graph=True)
            k_grad = getGrad(params[0], l_bound = -20.0, u_bound = 20.0 )
            alpha_grad = getGrad(params[1], l_bound = -20.0, u_bound = 20.0 )
            d_maintain[i,:] = np.append( k_grad.reshape(1,-1), alpha_grad.reshape(1,-1), axis = 1 )[0]
            
            if constraints ==[]: 
                constraints = constraint.detach().numpy().reshape(-1,1)
            else:
                constraints = np.append( constraints, constraint.detach().numpy().reshape(-1,1), axis = 0 )       

        const = [ constraints + d_maintain @ d >= 0 ]
        const += [ cp.sum_squares( d ) <= 200 ]
        if len(improve_constraint_direction)>0:
            obj = cp.Minimize( improve_constraint_direction @ d )
        else:
            obj = cp.Minimize(  objective_grad @ d  )
        problem = cp.Problem( obj, const )    
        problem.solve( solver = cp.GUROBI )    
        if problem.status != 'optimal':
            print("Cannot Find feasible direction")
            exit()
        
        # print("update direction: ", d.value.T)
        
        return d.value
    
    else:
        if len( improve_constraints ) > 0:
            obj = cp.Maximize( improve_constraint_direction @ d )
            # print("update direction: ", -improve_constraint_direction.reshape(-1,1).T)
            return -improve_constraint_direction.reshape(-1,1)
        else:
            return -objective_grad.reshape(-1,1)
        
    


def simulate_scenario(movie_name = 'test.mp4', adapt = False, noise = 0.1, enforce_input_constraints = False, leader_predict_function = traced_sigma_point_expand_JIT, control_leader_predict_function = [] ):

    t = 0
    first_time = True
    plt.ion()
    fig = plt.figure()
    # Plotting             
    
    ax = plt.axes(xlim=plot_x_lim,ylim=plot_y_lim)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect(1)

    follower = Unicycle(follower_init_pose, dt_inner, ax, num_robots=num_robots, id = 0, min_D = d_min, max_D = d_max, FoV_angle = angle_max, color='g',palpha=1.0, alpha=alpha_cbf, k = k_clf, num_alpha = 3 )
    leader = SingleIntegrator2D(leader_init_pose, dt_inner, ax, color='r',palpha=1.0, target = 0 )

    metadata = dict(title='Movie Adapt 0', artist='Matplotlib',comment='Movie support!')
    writer = FFMpegWriter(fps=15, metadata=metadata)

    step_rewards_adapt = []
    follower.states_array = np.copy( follower.X )
    leader.states_array = np.copy( leader.X )

    with writer.saving(fig, movie_name, 100): 

        for i in range(num_steps):

            # High frequency
            if i % outer_loop != 0 or i<learn_period:
            
                uL, vL = leader_motion(t)
                u_leader = np.array([ uL, vL ]).reshape(-1,1)
                
                leader.step(u_leader, dt_inner)
                
                # implement controller
                initialize_tensors(follower, leader)
                u_ref = unicycle_nominal_input_tensor_jit( follower.X_torch, leader.X_torch )
                A, B = compute_A1_b1_tensor( follower, leader, follower.X_torch, leader.X_torch, torch.tensor(t), torch.tensor(noise), control_leader_predict_function = control_leader_predict_function )
                
                solution, deltas = cbf_controller_layer( u_ref, A, B )
                if np.any( deltas.detach().numpy() > 0.01 ) and enforce_input_constraints:
                    print("Solution failed")
                    return fig, ax, follower, leader, step_rewards_adapt
                    solution = solution.detach().numpy()
                    solution[0,0] = np.clip( solution[0,0], -u1_max, u1_max )
                    solution[1,0] = np.clip( solution[1,0], -u2_max, u2_max )
                else:
                    # print(f"u_follower:{solution}")
                    follower.step(solution.detach().numpy(), dt_inner)
                follower.ks_u = np.append( follower.ks_u, follower.k )
                follower.alphas_u = np.append( follower.alphas_u, follower.alpha, axis=1 )
                
                # print(f"reward computation: f:{ follower.X.T }, L:{leader.X.T}")
                step_rewards_adapt.append( unicycle_compute_reward_jit(torch.tensor(follower.X),torch.tensor(leader.X)).detach().numpy()[0,0] )
                
                t = t + dt_inner
                
            # Low Frequency tuning
            else: 
                
                if adapt:
                    initialize_tensors(follower, leader)
                    
                    success = False
                    while not success:
                        maintain_constraints, improve_constraints, success, reward = get_future_reward( follower, leader, t = t, noise = torch.tensor(noise), enforce_input_constraints = enforce_input_constraints, leader_predict_function = leader_predict_function, )                    
                        grads = constrained_update( reward, maintain_constraints, improve_constraints, [follower.k_torch, follower.alpha_torch] )
                        
                        grads = np.clip( grads, -2.0, 2.0 )
                        follower.k = np.clip(follower.k + lr_alpha * grads[0], 0.0, None )
                        follower.alpha = np.clip( follower.alpha + lr_alpha * grads[1:].reshape(-1,1), 0.0, None )
                        print(f"follower k:{follower.k}, alpha:{follower.alpha.T}")
                        follower.ks = np.append( follower.ks, follower.k )
                        follower.alphas = np.append( follower.alphas, follower.alpha, axis=1 )
                        initialize_tensors(follower, leader)
                    # print("Successfully made it feasible")      
                    # exit()     
                        
            fig.canvas.draw()
            fig.canvas.flush_events()
            writer.grab_frame()
           
    # return data for plotting 
    return fig, ax, follower, leader, step_rewards_adapt
  
## Without noise: perfect knowledge??

noise  = 1.0 #0.5
save_plot = True

fig1, ax1, follower1, leader1, rewards1 = simulate_scenario( movie_name = 'cs2_noise_no_bound_no_adapt.mp4', adapt = False, noise = 0.0, enforce_input_constraints=False, leader_predict_function = traced_sigma_point_expand_JIT, control_leader_predict_function = traced_leader_predict_jit )  
fig2, ax2, follower2, leader2, rewards2 = simulate_scenario( movie_name = 'cs2_noise_adapt_no_bound.mp4', adapt = True, noise = 0.0, enforce_input_constraints=False, leader_predict_function = traced_sigma_point_expand_JIT, control_leader_predict_function = traced_leader_predict_jit )  
fig3, ax3, follower3, leader3, rewards3 = simulate_scenario( movie_name = 'cs2_noise_no_adapt_with_bound.mp4', adapt = False, noise = 0.0, enforce_input_constraints=True, leader_predict_function = traced_sigma_point_expand_JIT, control_leader_predict_function = traced_leader_predict_jit )
fig4, ax4, follower4, leader4, rewards4 = simulate_scenario( movie_name = 'cs2_noise_adapt_with_bound.mp4', adapt = True, noise = 0.0, enforce_input_constraints=True, leader_predict_function = traced_sigma_point_expand_JIT, control_leader_predict_function = traced_leader_predict_jit )

fig5, ax5, follower5, leader5, rewards5 = simulate_scenario( movie_name = 'cs2_ideal_no_bound_no_adapt.mp4', adapt = False, noise = 0.0, enforce_input_constraints=False, leader_predict_function = traced_sigma_point_expand_ideal_JIT, control_leader_predict_function = traced_leader_predict_ideal_jit )  
fig6, ax6, follower6, leader6, rewards6 = simulate_scenario( movie_name = 'cs2_ideal_adapt_no_bound.mp4', adapt = True, noise = 0.0, enforce_input_constraints=False, leader_predict_function = traced_sigma_point_expand_ideal_JIT, control_leader_predict_function = traced_leader_predict_ideal_jit )  
fig7, ax7, follower7, leader7, rewards7 = simulate_scenario( movie_name = 'cs2_ideal_no_adapt_with_bound.mp4', adapt = False, noise = 0.0, enforce_input_constraints=True, leader_predict_function = traced_sigma_point_expand_ideal_JIT, control_leader_predict_function = traced_leader_predict_ideal_jit )
fig8, ax8, follower8, leader8, rewards8 = simulate_scenario( movie_name = 'cs2_ideal_adapt_with_bound.mp4', adapt = True, noise = 0.0, enforce_input_constraints=True, leader_predict_function = traced_sigma_point_expand_ideal_JIT, control_leader_predict_function = traced_leader_predict_ideal_jit )

fig9, ax9, follower9, leader9, rewards9 = simulate_scenario( movie_name = 'cs2_nominal_no_bound_no_adapt.mp4', adapt = False, noise = 0.0, enforce_input_constraints=False, leader_predict_function = traced_sigma_point_expand_nominal_JIT, control_leader_predict_function = traced_leader_predict_nominal_jit )  
fig10, ax10, follower10, leader10, rewards10 = simulate_scenario( movie_name = 'cs2_nominal_adapt_no_bound.mp4', adapt = True, noise = 0.0, enforce_input_constraints=False, leader_predict_function = traced_sigma_point_expand_nominal_JIT, control_leader_predict_function = traced_leader_predict_nominal_jit )  
fig11, ax11, follower11, leader11, rewards11 = simulate_scenario( movie_name = 'cs2_nominal_no_adapt_with_bound.mp4', adapt = False, noise = 0.0, enforce_input_constraints=True, leader_predict_function = traced_sigma_point_expand_nominal_JIT, control_leader_predict_function = traced_leader_predict_nominal_jit )
fig12, ax12, follower12, leader12, rewards12 = simulate_scenario( movie_name = 'cs2_nominal_adapt_with_bound.mp4', adapt = True, noise = 0.0, enforce_input_constraints=True, leader_predict_function = traced_sigma_point_expand_nominal_JIT, control_leader_predict_function = traced_leader_predict_nominal_jit )
  
  
plt.ioff()
# plt.show()
# exit()

h11 = []; h21 = []
h12 = []; h22 = []
h13 = []; h23 = []

def get_barriers( follower, leader ):
    h1s = []; h2s = []; h3s = []
    for i in range( follower.Xs.shape[1] ):
        h1, h2, h3 = unicycle_SI2D_fov_barrier( follower.Xs[:,i].reshape(-1,1), leader.Xs[:,i].reshape(-1,1) )
        h1s.append(h1); h2s.append(h2); h3s.append(h3)
    tp = np.linspace( 0, dt_inner * follower.Xs.shape[1], follower.Xs.shape[1]  )
    return tp, h1s, h2s, h3s
    
# tp1, h11, h12, h13 = get_barriers( follower1, leader1 )    
# tp2, h21, h22, h23 = get_barriers( follower2, leader2 )
# tp3, h31, h32, h33 = get_barriers( follower3, leader3 )
# tp4, h41, h42, h43 = get_barriers( follower4, leader4 )
# tp5, h51, h52, h53 = get_barriers( follower5, leader5 )    
# tp6, h61, h62, h63 = get_barriers( follower6, leader6 )
# tp7, h71, h72, h73 = get_barriers( follower7, leader7 )
# tp8, h81, h82, h83 = get_barriers( follower8, leader8 )
# tp9, h91, h92, h93 = get_barriers( follower9, leader9 )    
# tp10, h101, h102, h103 = get_barriers( follower10, leader10 )
# tp11, h111, h112, h113 = get_barriers( follower11, leader11 )
# tp12, h121, h122, h123 = get_barriers( follower12, leader12 )


# print(f"rewards1:{rewards12}")
   
# Plot

print("Plotting now")

def plot_results(follower1, follower2, follower3, follower4, leader1, leader2, leader3, leader4, rewards1, rewards2, rewards3, rewards4, namespace="noise"):
    
    tp1, h11, h12, h13 = get_barriers( follower1, leader1 )    
    tp2, h21, h22, h23 = get_barriers( follower2, leader2 )
    tp3, h31, h32, h33 = get_barriers( follower3, leader3 )
    tp4, h41, h42, h43 = get_barriers( follower4, leader4 )

    figure1, axis1 = plt.subplots( 1 , 1)
    axis1.plot(tp1,h11,'r',label='Unbounded - Fixed', alpha = 0.3)
    axis1.plot(tp2,h21,'r.',label='Unbounded - Adaptive')
    axis1.plot(tp3,h31,'r--',label='Bounded - Fixed')
    axis1.plot(tp4,h41,'r',label='Bounded - Adapt')

    axis1.plot(tp1,h12,'g',label='Unbounded - Fixed', alpha = 0.3)
    axis1.plot(tp2,h22,'g.',label='Unbounded - Adaptive')
    axis1.plot(tp3,h32,'g--',label='Bounded - Fixed')
    axis1.plot(tp4,h42,'g',label='Bounded - Adapt')

    axis1.plot(tp1,h13,'k',label='Unbounded - Fixed', alpha = 0.3)
    axis1.plot(tp2,h23,'k.',label='Unbounded - Adaptive')
    axis1.plot(tp3,h33,'k--',label='Bounded - Fixed')
    axis1.plot(tp4,h43,'k',label='Bounded - Adapt')
    
    axis1.set_ylim(-0.2,4.0)

    # axis1.set_title('Follower barriers 1 alphas')
    axis1.set_xlabel('time (s)')
    axis1.legend()
    axis1.grid()

    figure2, axis2 = plt.subplots( 2 , 1)
    axis2[0].plot(tp1,follower1.Us[0,:],'r',label='Unbounded - Fixed', alpha = 1.0)
    axis2[0].plot(tp2,follower2.Us[0,:],'b',label='Unbounded - Adaptive')
    axis2[0].plot(tp3,follower3.Us[0,:],'k',label='Bounded - Fixed')
    axis2[0].plot(tp4,follower4.Us[0,:],'g',label='Bounded - Adapt')
    axis2[0].axhline(y = u1_max, color = 'k', linestyle='--')
    axis2[0].set_ylabel('$u$')
    axis2[0].grid()
    axis2[0].set_ylim(0.0,3.5)

    axis2[1].plot(tp1,follower1.Us[1,:],'r',label='Unbounded - Fixed', alpha = 0.3)
    axis2[1].plot(tp2,follower2.Us[1,:],'b',label='Unbounded - Adaptive')
    axis2[1].plot(tp3,follower3.Us[1,:],'k',label='Bounded - Fixed')
    axis2[1].plot(tp4,follower4.Us[1,:],'g',label='Bounded - Adapt')
    axis2[1].set_ylabel(r'$\omega$')
    axis2[1].axhline(y = u2_max, color = 'k', linestyle='--')
    axis2[1].grid()
    axis2[1].set_ylim(-5.0,5.0)

    # axis1.set_title('Follower barriers 1 alphas')
    axis2[1].set_xlabel('time (s)')
    axis2[1].legend()

    figure3, axis3 = plt.subplots( 1 , 1)
    tp1 = np.linspace( 0, dt_inner * len(rewards1), len(rewards1)  )
    tp2 = np.linspace( 0, dt_inner * len(rewards2), len(rewards2)  )
    tp3 = np.linspace( 0, dt_inner * len(rewards3), len(rewards3)  )
    tp4 = np.linspace( 0, dt_inner * len(rewards4), len(rewards4)  )
    axis3.plot(tp1,-rewards1,'k',label='Unbounded - Fixed', alpha = 1.0)
    axis3.plot(tp2,-rewards2,'r',label='Unbounded - Adaptive', alpha = 1.0)
    axis3.plot(tp3,-rewards3,'g',label='Bounded - Fixed', alpha = 1.0)
    axis3.plot(tp4,-rewards4,'c',label='Bounded - Adapt', alpha = 1.0)
    axis3.set_xlabel('time (s)')
    axis3.legend()
    axis3.set_ylabel('Rewards')
    axis3.grid()
    axis3.set_ylim(0.9,2.2)

    figure4, axis4 = plt.subplots( 2 , 1)
    tp1 = np.linspace( 0, dt_inner * follower2.alphas_u.shape[1], follower2.alphas_u.shape[1]  )
    tp2 = np.linspace( 0, dt_inner * follower2.alphas_u.shape[1], follower2.alphas_u.shape[1]  )
    tp3 = np.linspace( 0, dt_inner * follower2.alphas_u.shape[1], follower2.alphas_u.shape[1]  )
    tp4 = np.linspace( 0, dt_inner * follower2.alphas_u.shape[1], follower2.alphas_u.shape[1]  )
    axis4[0].plot(tp4,follower2.ks_u,'c',label=r'$\alpha_0$', alpha = 1.0)
    axis4[0].plot(tp1,follower2.alphas_u[0,:],'k',label=r'$\alpha_1$', alpha = 1.0)
    axis4[0].plot(tp2,follower2.alphas_u[1,:],'r',label=r'$\alpha_2$', alpha = 1.0)
    axis4[0].plot(tp3,follower2.alphas_u[2,:],'g',label=r'$\alpha_3$', alpha = 1.0)
    # axis4[0].set_title('Unbounded Input with Adaptive')
    axis4[0].grid()
    axis4[0].legend()
    axis4[0].set_ylim(-0.1, 2.0)

    axis4[1].plot(tp4,follower4.ks_u,'c',label=r'$\alpha_0$', alpha = 1.0)
    axis4[1].plot(tp1,follower4.alphas_u[0,:],'k',label=r'$\alpha_1$', alpha = 1.0)
    axis4[1].plot(tp2,follower4.alphas_u[1,:],'r',label=r'$\alpha_2$', alpha = 1.0)
    axis4[1].plot(tp3,follower4.alphas_u[2,:],'g',label=r'$\alpha_3$', alpha = 1.0)
    # axis4[1].set_title('Bounded Adaptive')
    axis4[1].set_xlabel('time (s)')
    axis4[1].grid()
    axis4[1].legend()
    axis4[1].set_ylim(-0.1, 20)
    
    if save_plot:
        figure1.savefig(namespace+"barriers.eps")
        figure1.savefig(namespace+"barriers.png")
        figure2.savefig(namespace+"control.eps")
        figure2.savefig(namespace+"control.png")
        figure3.savefig(namespace+"rewards.eps")
        figure3.savefig(namespace+"rewards.png")
        figure4.savefig(namespace+"params.eps")
        figure4.savefig(namespace+"params.png")
   
plot_results( follower1, follower2, follower3, follower4, leader1, leader2, leader3, leader4, np.asarray(rewards1), np.asarray(rewards2), np.asarray(rewards3), np.asarray(rewards4),  namespace="cs2_noise" )
plot_results( follower5, follower6, follower7, follower8, leader5, leader6, leader7, leader8, np.asarray(rewards5), np.asarray(rewards6), np.asarray(rewards7), np.asarray(rewards8),  namespace="cs2_ideal" )
plot_results( follower9, follower10, follower11, follower12, leader9, leader10, leader11, leader12, np.asarray(rewards9), np.asarray(rewards10), np.asarray(rewards11), np.asarray(rewards12),  namespace="cs2_nominal" )
    
plt.show()
