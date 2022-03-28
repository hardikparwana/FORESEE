import numpy as np
from utils.utils import wrap_angle

class Unicycle:
    
    def __init__(self,X0,dt,ax,id,num_robots=1,num_adversaries = 1, alpha=0.8,color='r',palpha=1.0):
        '''
        X0: iniytial state
        dt: simulation time step
        ax: plot axis handle
        id: robot id
        '''
        
        self.type = 'Unicycle'
        
        self.X = X0.reshape(-1,1)
        self.dt = dt
        self.id = id
        
        self.U = np.array([0,0]).reshape(-1,1)
        self.x_dot_nominal = np.array([ [0],[0],[0] ])
        self.U_ref = np.array([0,0]).reshape(-1,1)
        
        # Plot handles
        self.body = ax.scatter([],[],c=color,alpha=palpha,s=10)
        self.render_plot()
        
        # for Trust computation
        self.adv_alpha = alpha*np.ones(num_adversaries)
        self.trust_adv = 1
        self.robot_alpha = alpha*np.ones(num_robots)
        self.trust_robot = 1
        self.adv_objective = [0] * num_adversaries
        self.robot_objective = [0] * num_robots
        
        num_constraints1  = num_robots - 1 + num_adversaries
        self.A1 = np.zeros((num_constraints1,2))
        self.b1 = np.zeros((num_constraints1,1))
     
    def f(self):
        return np.array([0,0,0]).reshape(-1,1)
    
    def g(self):
        return np.array([ [ np.cos(self.X[2,0]), 0 ],
                          [ np.sin(self.X[2,0]), 0],
                          [0, 1] ])       
         
    def step(self,U): 
        self.U = U.reshape(-1,1)
        self.X = self.X + ( self.f() + self.g() @ self.U )*self.dt
        self.X[2,0] = wrap_angle(self.X[2,0])
        return self.X
    
    def render_plot(self):
        x = np.array([self.X[0,0],self.X[1,0]])
        self.body.set_offsets([x[0],x[1]])
        
    def lyapunov(self, G):
        V = np.linalg.norm( self.X[0:2] - G[0:2] )**2
        dV_dx = np.append( 2*( self.X[0:2] - G[0:2] ).T, [[0]], axis=1)
        return V, dV_dx
    
    def nominal_input(self,G):
        # V, dV_dx = self.lyapunov(G)
        #Define gamma for the Lyapunov function
        k_omega = 2.0 #0.5#2.5
        k_v = 2.0 #0.5
        theta_d = np.arctan2(G.X[:,0][1]-self.X[1,0],G.X[:,0][0]-self.X[0,0])
        error_theta = wrap_angle( theta_d - self.X[2,0] )

        omega = k_omega*error_theta

        distance = max(np.linalg.norm( self.X[0:2,0]-G.X[0:2,0] ) - 0.3,0)

        v = k_v*( distance )*np.cos( error_theta )
        return np.array([v, omega]).reshape(-1,1) #np.array([v,omega])
    
    # def agent_barrier(self,agent,d_min):
    #     h = d_min**2 - np.linalg.norm(self.X[0:2] - agent.X[0:2])**2
    #     dh_dxi = np.append( -2*( self.X[0:2] - agent.X[0:2] ).T, [[0]], axis=1)
        
    #     if agent.type=='SingleIntegrator2D':
    #         dh_dxj = 2*( self.X[0:2] - agent.X[0:2] ).T
    #     elif agent.type=='Unicycle':
    #         dh_dxj = np.append( -2*( self.X[0:2] - agent.X[0:2] ).T, [[0]], axis=1 )
    #     elif agent.type=='FixedWing':
    #         print("TO DO here!!!!")
        
    #     return h, dh_dxi, dh_dxj
    
    def sigma(self,s):
        k1 = 2
        return (np.exp(k1-s)-1)/(np.exp(k1-s)+1)
    
    def sigma_der(self,s):
        k1 = 2
        return -np.exp(k1-s)/( 1+np.exp( k1-s ) ) * ( 1 + self.sigma(s) )
    
    def agent_barrier(self,agent,d_min):
        beta = 1.01
        h = beta*d_min**2 - np.linalg.norm(self.X[0:2] - agent.X[0:2])**2
        h1 = h
        
        theta = self.X[2,0]
        s = (self.X[0:2] - agent.X[0:2]).T @ np.array( [np.sin(theta),np.cos(theta)] ).reshape(-1,1)
        h = h - self.sigma(s)
        # print(f"h1:{h1}, h2:{h}")
        # assert(h1<0)
        der_sigma = self.sigma_der(s)
        dh_dxi = np.append( -2*( self.X[0:2] - agent.X[0:2] ).T - der_sigma * ( np.array([ [np.sin(theta), np.cos(theta)] ]) ),  - der_sigma * ( np.cos(theta)*( self.X[0,0]-agent.X[0,0] ) - np.sin(theta)*( self.X[1,0] - agent.X[1,0] ) ) , axis=1)
        
        if agent.type=='SingleIntegrator2D':
            dh_dxj = 2*( self.X[0:2] - agent.X[0:2] ).T
        elif agent.type=='Unicycle':
            dh_dxj = np.append( -2*( self.X[0:2] - agent.X[0:2] ).T + der_sigma * ( np.array([ [np.sin(theta), np.cos(theta)] ]) ), np.array([[0]]), axis=1 )
        else:
            dh_dxj = 2*( self.X[0:2] - agent.X[0:2] ).T
        
        return h, dh_dxi, dh_dxj
    