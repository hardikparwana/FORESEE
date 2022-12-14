import numpy as np
import torch

class SingleIntegrator2D:
    
    def __init__(self,X0,dt,ax,id=0,num_robots=1,num_adversaries = 1, alpha=0.8,color='r',palpha=1.0,plot=True, identity = 'adversary', target='move right', predict_function = None):
        '''
        X0: iniytial state
        dt: simulation time step
        ax: plot axis handle
        id: robot id
        '''
        
        self.type = 'SingleIntegrator2D'   
        self.identity = identity   
        self.target = target
        
        ## GP  ######
        self.gp_x = []
        self.gp_y = []
        self.gp = []
        self.likelihood = []
        self.predict_function = predict_function
        ###########
        
        X0 = X0.reshape(-1,1)
        self.X = X0
        self.X_nominal = np.copy(self.X)
        self.dt = dt
        self.id = id
        self.color = color
        self.palpha = palpha

        self.U = np.array([0,0]).reshape(-1,1)
        self.U_nominal = np.array([0,0]).reshape(-1,1)
        self.nextU = self.U
        self.U_ref_nominal = np.copy(self.U)

        # Plot handles
        self.plot = plot
        if self.plot:
            self.body = ax.scatter([],[],c=color,alpha=palpha,s=10)
            self.render_plot()
        
        self.alpha = alpha*np.ones((num_robots,1))
         # for Trust computation
        self.adv_alpha =  alpha*np.ones((1,num_adversaries))# alpha*np.ones((1,num_adversaries))
        self.trust_adv = np.ones((1,num_adversaries))
        self.robot_alpha = alpha*np.ones((1,num_robots))
        self.trust_robot = np.ones((1,num_robots))
        self.adv_objective = [0] * num_adversaries
        self.robot_objective = [0] * num_robots
        self.robot_h = np.ones((1,num_robots))
        self.adv_h = np.ones((1,num_adversaries))        
        
        # Old
        # for Trust computation
        # self.adv_alpha = alpha*np.ones(num_adversaries)
        # self.trust_adv = 1
        # self.robot_alpha = alpha*np.ones(num_robots)
        # self.trust_robot = 1
        # self.adv_objective = [0] * num_adversaries
        # self.robot_objective = [0] * num_robots
        
        num_constraints1  = num_robots - 1 + num_adversaries
        self.A1 = np.zeros((num_constraints1,2))
        self.b1 = np.zeros((num_constraints1,1))
        
        # For plotting
        self.adv_alphas = alpha*np.ones((1,num_adversaries))
        self.trust_advs = np.ones((1,num_adversaries))
        self.robot_alphas = alpha*np.ones((1,num_robots))
        self.trust_robots = 1*np.ones((1,num_robots))
        
        self.adv_hs = np.ones((1,num_adversaries))
        self.robot_hs = np.ones((1,num_robots))
        
        
        ## Store state
        self.X_org = np.copy(self.X)
        self.U_org = np.copy(self.U)
        
        self.Xs = [] #np.copy(self.X)
        self.Xdots = [] #np.array([0,0]).reshape(-1,1)
        # self.Xs = X0.reshape(-1,1)
        self.Us = [] #np.array([0,0]).reshape(-1,1)
        
        
    def f(self):
        return np.array([0,0]).reshape(-1,1)
    
    def f_torch(self,x):
        return torch.tensor(np.array([0,0]).reshape(-1,1),dtype=torch.float)
    
    def g(self):
        return np.array([ [1, 0],[0, 1] ])
    
    def g_torch(self,x):
        return torch.tensor(np.array([ [1, 0],[0, 1] ]),dtype=torch.float)
        
    def step(self,U,dt,mode='actual'): #Just holonomic X,T acceleration

        xold = np.copy(self.X)
        self.U = U.reshape(-1,1)
        self.X = self.X + ( self.f() + self.g() @ self.U ) * dt
        Xdot = self.f() + self.g() @ self.U
        self.render_plot()
        
        if self.Xs == []:
            self.Xs = np.copy(xold)
            self.Us = np.copy(self.U)
            self.Xdots = np.copy(Xdot)
        else:            
            self.Xs = np.append(self.Xs,xold,axis=1)
            self.Us = np.append(self.Us,self.U,axis=1)
            self.Xdots = np.append( self.Xdots, Xdot , axis=1 )
        
        return self.X

    def render_plot(self):
        if self.plot:
            x = np.array([self.X[0,0],self.X[1,0]])

            # scatter plot update
            self.body.set_offsets([x[0],x[1]])

    def lyapunov(self, G):
        V = np.linalg.norm( self.X - G[0:2] )**2
        dV_dx = 2*( self.X - G[0:2] ).T
        return V, dV_dx
    
    def agent_barrier(self,agent,d_min):
        h = d_min**2 - np.linalg.norm(self.X - agent.X[0:2])**2
        dh_dxi = -2*( self.X - agent.X[0:2] ).T
        
        if agent.type=='SingleIntegrator2D':
            dh_dxj = 2*( self.X - agent.X[0:2] ).T
        elif agent.type=='Unicycle':
            dh_dxj = np.append( -2*( self.X - agent.X[0:2] ).T, [[0]], axis=1 )
        return h, dh_dxi, dh_dxj
    
    

def leader_motion_predict(t):
    uL = 2.0 # 
    vL = 3*np.sin(np.pi*t*4) #  0.1 # 1.2
    # uL = 1
    # vL = 1
    return uL, vL

def leader_motion(t, noise = 0.0):
    # uL = 0.5 + 0.5
    # vL = 3*np.sin(np.pi*t*4) + 2.0 * np.sin(np.pi*t*4) + 0.1#  0.1 # 1.2
    # uL = 0.5 + 0.5
    # vL = 3*np.sin(np.pi*t*4) + 0.5#  0.1 # 1.2
    uL = 2.0
    vL = 3*np.sin(np.pi*t*4) # + 0.5#  0.1 # 1.2
    
    # uL = 1
    # vL = 1
    return uL, vL

def leader_predict(t, noise = 0.0):
    uL, vL = leader_motion_predict(t)
    # print("noise", noise)
    mu = torch.tensor([[uL, vL]], dtype=torch.float).reshape(-1,1)
    bias = mu / 4
    mu = mu +  bias #torch.tensor([0.5, 0.5]).reshape(-1,1)
    cov = torch.zeros((2,2), dtype=torch.float)
    # cov[0,0] = noise
    # cov[1,1] = noise
    cov[0,0] = torch.square( torch.norm(bias)/2 )
    cov[1,1] = torch.square( torch.norm(bias)/2 )
    return mu, cov
traced_leader_predict_jit = leader_predict #torch.jit.trace( leader_predict, ( torch.tensor(0), torch.tensor(0) ) )


