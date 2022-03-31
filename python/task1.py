import numpy as np
import time
import cvxpy as cp
import matplotlib.pyplot as plt
from robot_models.SingleIntegrator2D import *
from utils.utils import *

# Sim Parameters                  
dt = 0.01
tf = 6.5#10
num_steps = int(tf/dt)
t = 0
d_min = 0.1
h_min = 0.5

min_dist = 0.05
alpha_cbf = 0.8
alpha_der_max = 0.05#0.5
update_param = True

# Plot                  
plt.ion()
fig = plt.figure()
ax = plt.axes(xlim=(0,30),ylim=(-10,10))
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_aspect(1)

# agents
robots = []
num_robots = 3
robots.append( SingleIntegrator2D(np.array([3,1]), dt, ax, num_robots=num_robots, id = 0, color='g',palpha=1.0, alpha=alpha_cbf ) )
robots.append( SingleIntegrator2D(np.array([2.5,0]), dt, ax, num_robots=num_robots, id = 1, color='g',palpha=1.0, alpha=alpha_cbf ) )
robots.append( SingleIntegrator2D(np.array([3.5,0]), dt, ax, num_robots=num_robots, id = 2, color='g',palpha=1.0, alpha=alpha_cbf ) )

# agent nominal version
robots_nominal = []
num_robots = 3
robots_nominal.append( SingleIntegrator2D(np.array([3,1]), dt, ax, num_robots=num_robots, id = 0, color='g',palpha=0.4) )
robots_nominal.append( SingleIntegrator2D(np.array([2.5,0]), dt, ax, num_robots=num_robots, id = 1, color='g',palpha=0.4 ) )
robots_nominal.append( SingleIntegrator2D(np.array([3.5,0]), dt, ax, num_robots=num_robots, id = 2, color='g',palpha=0.4 ) )
U_nominal = np.zeros((2,num_robots))

# Uncooperative
greedy = []
greedy.append( SingleIntegrator2D(np.array([0,4]), dt, ax, color='r',palpha=1.0) )

greedy_nominal = []
greedy_nominal.append( SingleIntegrator2D(np.array([0,4]), dt, ax, color='r',palpha=0.4) )

# Adversarial agents
# adversary = []
# adversary.append( SingleIntegrator2D(np.array([0,4]), dt, ax) )
num_adversaries = 1

############################## Optimization problems ######################################

###### 1: CBF Controller
u1 = cp.Variable((2,1))
u1_ref = cp.Parameter((2,1),value = np.zeros((2,1)) )
num_constraints1  = num_robots - 1 + num_adversaries
A1 = cp.Parameter((num_constraints1,2),value=np.zeros((num_constraints1,2)))
b1 = cp.Parameter((num_constraints1,1),value=np.zeros((num_constraints1,1)))
const1 = [A1 @ u1 <= b1]
objective1 = cp.Minimize( cp.sum_squares( u1 - u1_ref  ) )
cbf_controller = cp.Problem( objective1, const1 )


###### 2: Best case controller
u2 = cp.Variable( (2,1) )
Q2 = cp.Parameter( (1,2), value = np.zeros((1,2)) )
num_constraints2 = num_robots - 1 + num_adversaries
# minimze A u s.t to other constraints
A2 = cp.Parameter((num_constraints2,2),value=np.zeros((num_constraints2,2)))
b2 = cp.Parameter((num_constraints2,1),value=np.zeros((num_constraints2,1)))
const2 = [A2 @ u2 <= b2]
const2 += [cp.abs(u2[0,0])<=10.0]
const2 += [cp.abs(u2[1,0])<=10.0]
objective2 = cp.Minimize( Q2 @ u2 )
best_controller = cp.Problem( objective2, const2 )

##########################################################################################

tp = []

for i in range(num_steps):
    
    const_index = 0
    
    ## Greedy's nominal movement
    u_greedy_nominal = np.array([1.0, 0.0])
    greedy_nominal[0].step(u_greedy_nominal)
    
    ## Greedy's believed movement
    V_nominal, dV_dx_nominal = greedy[0].lyapunov( greedy_nominal[0].X  )
    u_greedy_nominal = -1.0 * dV_dx_nominal.T /np.linalg.norm( dV_dx_nominal )
    
    ## Greedy actual movement
    V, dV_dx = greedy[0].lyapunov( robots[0].X )
    u_greedy = -5.0 * dV_dx.T / np.linalg.norm( dV_dx )
    greedy[0].step(u_greedy)
    
    # Move nominal agents
    for j in range(num_robots):
        u_nominal = np.array([0,1.0])
        robots_nominal[j].step( u_nominal )
        V, dV_dx = robots[j].lyapunov(robots_nominal[j].X)
        robots[j].U_nominal = -3.0*dV_dx.T/np.linalg.norm(dV_dx)
        
    
    for j in range(num_robots):
        
        const_index = 0
                        
        # greedy
        for k in range(num_adversaries):
            h, dh_dxi, dh_dxk = robots[j].agent_barrier(greedy[k], d_min);  
            
            # Control QP constraint
            robots[j].A1[const_index,:] = dh_dxi @ robots[j].g()
            robots[j].b1[const_index] = -dh_dxi @ robots[j].f() - dh_dxk @ ( greedy[k].f() + greedy[k].g() @ greedy[k].U ) - robots[j].adv_alpha[0,k] * h
            const_index = const_index + 1

            # Best Case LP objective
            robots[j].adv_objective[k] = dh_dxi @ robots[j].g()
            
        for k in range(num_robots):
            
            if k==j:
                continue
            
            h, dh_dxi, dh_dxk = robots[j].agent_barrier(robots[k], d_min);
                
            # Control QP constraint
            robots[j].A1[const_index,:] = dh_dxi @ robots[j].g()
            robots[j].b1[const_index] = -dh_dxi @ robots[j].f() - dh_dxk @ ( robots[k].f() + robots[k].g() @ robots[k].U ) - robots[j].robot_alpha[0,k] * h
            const_index = const_index + 1
            
            # Best Case LP objective
            robots[j].robot_objective[k] = dh_dxi @ robots[j].g()
            
        
        
    for j in range(num_robots):
        
        const_index = 0      
        # Constraints in LP and QP are same      
        A1.value = robots[j].A1
        A2.value = robots[j].A1
        b1.value = robots[j].b1
        b2.value = robots[j].b1
        
        # Solve for trust factor
        if update_param:
            for k in range(num_adversaries):
                Q2 = robots[j].adv_objective[k]
                best_controller.solve()
                if best_controller.status!='optimal':
                    print(f"LP status:{best_controller.status}")
                            
                h, dh_dxi, dh_dxk = robots[j].agent_barrier(greedy[k], d_min);              
                A = dh_dxk
                b = -robots[j].adv_alpha[0,k] * h  - dh_dxi @ ( robots[j].f() + robots[j].g() @ u2.value ) #- dh_dxi @ robots[j].U
                
                robots[j].trust_adv[0,k] = compute_trust( A, b, u_greedy, u_greedy_nominal, h, min_dist, h_min )  
                if robots[j].trust_adv[0,k]<0:
                    print(f"{j}'s Trust of {k} adversary: {best_controller.status}: {robots[j].trust_adv[0,k]}, h:{h} ")    
                robots[j].adv_alpha[0,k] = robots[j].adv_alpha[0,k] + alpha_der_max * robots[j].trust_adv[0,k]/0.05*dt
                if (robots[j].adv_alpha[0,k]<0):
                    robots[j].adv_alpha[0,k] = 0.01
                    
            for k in range(num_robots):
                if k==j:
                    continue
            
                Q2 = robots[j].robot_objective[k]
                best_controller.solve()
                        
                h, dh_dxi, dh_dxk = robots[j].agent_barrier(robots[k], d_min);
                A = dh_dxk
                b = -robots[j].robot_alpha[0,k] * h - dh_dxi @ ( robots[j].f() + robots[j].g() @  u2.value)  #- dh_dxi @ robots[j].U  # need best case U here. not previous U
                
                robots[j].trust_robot[0,k] = compute_trust( A, b, robots[k].U, robots[k].U_nominal, h, min_dist, h_min )            
                if robots[j].trust_robot[0,k]<0:
                    print(f"{j}'s Trust of {k} robot: {best_controller.status}: {robots[j].trust_robot[0,k]}, h:{h}")
                robots[j].robot_alpha[0,k] = robots[j].robot_alpha[0,k] + alpha_der_max * robots[j].trust_robot[0,k]/0.05*dt
                if (robots[j].robot_alpha[0,k]<0):
                    robots[j].robot_alpha[0,k] = 0.01
                    
        # Plotting
        robots[j].adv_alphas = np.append( robots[j].adv_alphas, robots[j].adv_alpha, axis=0 )
        robots[j].trust_advs = np.append( robots[j].trust_advs, robots[j].trust_adv, axis=0 )
        robots[j].robot_alphas = np.append( robots[j].robot_alphas, robots[j].robot_alpha, axis=0 )
        robots[j].trust_robots = np.append( robots[j].trust_robots, robots[j].trust_robot, axis=0 )
        robots[j].robot_hs = np.append( robots[j].robot_hs, robots[j].robot_h, axis=0 )
        robots[j].adv_hs = np.append( robots[j].adv_hs, robots[j].adv_h, axis=0 )
        
        # Solve for control input
        u1_ref.value = robots[j].U_nominal
        cbf_controller.solve()
        if cbf_controller.status!='optimal':
            print(f"{j}'s input: {cbf_controller.status}")
        robots[j].nextU = u1.value        
        
    for j in range(num_robots):
        robots[j].step( robots[j].nextU )
        robots[j].render_plot()
        # print(f"{j} alphas: {robots[j].robot_alpha[0,j]}, adv_alhpa:{robots[j].adv_alpha}")
    
    t = t + dt
    tp.append(t)
    
    fig.canvas.draw()
    fig.canvas.flush_events()
    
   
plt.ioff()   
# Plot


####### Alphas #######
# Robot 0
figure1, axis1 = plt.subplots(2, 2)
axis1[0,0].plot(tp,robots[0].adv_alphas[1:,0],'r',label='Adversary')
axis1[0,0].plot(tp,robots[0].robot_alphas[1:,1],'g',label='Robot 2')
axis1[0,0].plot(tp,robots[0].robot_alphas[1:,2],'k',label='Robot 3')
axis1[0,0].set_title('Robot 1 alphas')
axis1[0,0].set_xlabel('time (s)')
axis1[0,0].legend()

# Robot 1
axis1[1,0].plot(tp,robots[1].adv_alphas[1:,0],'r',label='Adversary')
axis1[1,0].plot(tp,robots[1].robot_alphas[1:,0],'g',label='Robot 1')
axis1[1,0].plot(tp,robots[1].robot_alphas[1:,2],'k',label='Robot 3')
axis1[1,0].set_title('Robot 2 alphas')
axis1[1,0].set_xlabel('time (s)')
axis1[1,0].legend()

# Robot 2
axis1[0,1].plot(tp,robots[2].adv_alphas[1:,0],'r',label='Adversary')
axis1[0,1].plot(tp,robots[2].robot_alphas[1:,0],'g',label='Robot 1')
axis1[0,1].plot(tp,robots[2].robot_alphas[1:,1],'k',label='Robot 2')
axis1[0,1].set_title('Robot 3 alphas')
axis1[0,1].set_xlabel('time (s)')
axis1[0,1].legend()

#### TRUST ######
# Robot 0
figure2, axis2 = plt.subplots(2, 2)
axis2[0,0].plot(tp,robots[0].trust_robots[1:,1],'g',label='Robot 2')
axis2[0,0].plot(tp,robots[0].trust_advs[1:,0],'r',label='Adversary')
axis2[0,0].plot(tp,robots[0].trust_robots[1:,2],'k',label='Robot 3')
axis2[0,0].set_title('Robot 1 trust')
axis2[0,0].set_xlabel('time (s)')
axis2[0,0].legend()

# Robot 1
axis2[1,0].plot(tp,robots[1].trust_robots[1:,0],'g',label='Robot 1')
axis2[1,0].plot(tp,robots[1].trust_advs[1:,0],'r',label='Adversary')
axis2[1,0].plot(tp,robots[1].trust_robots[1:,2],'k',label='Robot 3')
axis2[1,0].set_title('Robot 2 trust')
axis2[1,0].set_xlabel('time (s)')
axis2[1,0].legend()

# Robot 2
axis2[0,1].plot(tp,robots[2].trust_advs[1:,0],'r',label='Adversary')
axis2[0,1].plot(tp,robots[2].trust_robots[1:,0],'g',label='Robot 1')
axis2[0,1].plot(tp,robots[2].trust_robots[1:,1],'k',label='Robot 2')
axis2[0,1].set_title('Robot 3 trust')
axis2[0,1].set_xlabel('time (s)')
axis2[0,1].legend()

######### Barriers ##########
# Robot 0
figure3, axis3 = plt.subplots(2, 2)
axis3[0,0].plot(tp,robots[0].robot_hs[1:,1],'g',label='Robot 2')
axis3[0,0].plot(tp,robots[0].adv_hs[1:,0],'r',label='Adversary')
axis3[0,0].plot(tp,robots[0].robot_hs[1:,2],'k',label='Robot 3')
axis3[0,0].set_title('Robot 1 CBFs')
axis3[0,0].set_xlabel('time (s)')
axis3[0,0].legend()

# Robot 1
axis3[1,0].plot(tp,robots[1].robot_hs[1:,0],'g',label='Robot 1')
axis3[1,0].plot(tp,robots[1].adv_hs[1:,0],'r',label='Adversary')
axis3[1,0].plot(tp,robots[1].robot_hs[1:,2],'k',label='Robot 3')
axis3[1,0].set_title('Robot 2 CBFs')
axis3[1,0].set_xlabel('time (s)')
axis3[1,0].legend()

# Robot 2
axis3[0,1].plot(tp,robots[2].adv_hs[1:,0],'r',label='Adversary')
axis3[0,1].plot(tp,robots[2].robot_hs[1:,0],'g',label='Robot 1')
axis3[0,1].plot(tp,robots[2].robot_hs[1:,1],'k',label='Robot 3')
axis3[0,1].set_title('Robot 3 CBFs')
axis3[0,1].set_xlabel('time (s)')
axis3[0,1].legend()

plt.show()