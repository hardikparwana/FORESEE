import jax.numpy as np
# import utils
from utils import wrap_angle

def car2D_dynamics( state, action, CarModel ):

    #Params
    m = CarModel["m"]#1000
    Iz = CarModel["Iz"]#1.0
    lf = CarModel["lf"]#1.0
    lr = CarModel["lr"]#1.0

    Cm1 = CarModel["Cm1"]#1.0
    Cm2 = CarModel["Cm2"]#1.0

    Br = CarModel["Br"]#1.0
    Cr = CarModel["Cr"]#1.0
    Dr = CarModel["Dr"]#1.0

    Bf = CarModel["Bf"]#1.0
    Cf = CarModel["Cf"]#1.0
    Df = CarModel["Df"]#1.0   
    
    Cr0 = CarModel["Cr0"]#1.0
    Cr2 = CarModel["Cr2"]#1.0
    
    
    def state_dot(state, action):
        # States
        x = state[0,0]
        y = state[1,0]
        phi = state[2,0]
        vx = state[3,0]
        vy = state[4,0]
        w = state[5,0]
        # th = state[6,0]

        # Control Input
        D = action[0,0]
        delta = action[1,0]
        # v_theta = action[3,0]

        alphaf = -np.arctan( (w*lf+vy)/vx ) + delta
        Ffy = Df * np.sin( Cf * np.arctan(Bf*alphaf) )

        alphar = -np.arctan((-w*lr+vy)/vx)
        Fry = Dr*np.sin( Cr*np.arctan(Br*alphar) )
        Frx = ( Cm1 - Cm2*vx )*D - Cr0 - Cr2*vx**2

        xdot = vx * np.cos(phi) - vy * np.sin(phi)
        ydot = vx * np.sin(phi) + vy * np.cos(phi)
        phidot = w
        vxdot = 1/m * ( Frx - Ffy*np.sin(delta) + m*vy*w )
        vydot = 1/m * ( Fry + Ffy*np.sin(delta) - m*vx*w )
        wdot = 1/Iz * ( Ffy*lf*np.cos(delta) - Fry*lr )
        # thdot = v_theta

        return np.array([ xdot, ydot, phidot, vxdot, vydot, wdot ]).reshape(-1,1)
    
    return state_dot


class Car2D:

    def __init__(self, x0, dt, ax, CarModel, plot=True):

        self.type = 'Car2D'

        self.X = x0.reshape(-1,1)
        self.U = np.zeros((2,1))

        self.dt = dt
        self.ax = ax
    
        self.plot=plot
        if self.plot: 
            self.body = ax.scatter([self.X[0,0]], [self.X[1,0]], s=100, facecolors='None', edgecolors='g')
            self.radii = 4.0#0.5
            self.axis = ax.plot([ self.X[0,0], self.X[0,0]+self.radii*np.cos(self.X[2,0]) ], [self.X[1,0], self.X[1,0]+self.radii*np.sin(self.X[2,0])], 'g', linewidth=3)
            self.render_plot()

        # dynamics function
        self.F = lambda x, u: car2D_dynamics( x, u, CarModel )

    def step( self, U ):
        xdot = self.F( self.X, U )
        self.X = self.X + xdot * self.dt
        self.X[2,0] = wrap_angle(self.X[2,0])
        if self.plot:
            self.render_plot()

    def render_plot(self):
        self.body.set_offsets([self.X[0,0], self.X[1,0]])
        self.axis[0].set_xdata( [ self.X[0,0], self.X[0,0]+self.radii*np.cos(self.X[2,0]) ] )
        self.axis[0].set_ydata( [ self.X[1,0], self.X[1,0]+self.radii*np.sin(self.X[2,0]) ] )
    

