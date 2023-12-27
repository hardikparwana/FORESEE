import numpy as np

def getMPC_vars(CarModel):


    if CarModel == "ORCA":
       
        # MPC settings ############################################################

        MPC_vars = {

            # prediction horizon
            "N" : 40,
            # sampling time
            "Ts" : 0.02,
            # used model (TODO incorparate new models)
            "ModelNo" : 1,
            # use bounds on all opt variables (TODO implement selective bounds)
            "fullBound" : 1,  
            ###########################################################################
            # state-input scaling #####################################################
            ###########################################################################
            # normalization matricies (scale states and inputs to ~ +/- 1
            "Tx" : np.diag(1./np.array([3,3,2*np.pi,4,2,7,30])),
            "Tu" : np.diag(1./np.array([1,0.35,6])),

            "invTx" : np.diag([3,3,2*np.pi,4,2,7,30]),
            "invTu" : np.diag([1,0.35,6]),

            "TDu" : np.eye(3),
            "invTDu" : np.eye(3),
            # identity matricies if inputs should not be normalized
            # "Tx" : eye(7),
            # "Tu" : eye(3),
            # 
            # "invTx" : eye(7),
            # "invTu" : eye(3),
            # 
            # "TDu" : eye(3),
            # "invTDu" : eye(3),
            ###########################################################################
            # state-input bounds ######################################################
            ###########################################################################

            # bounds for non-nomalized state-inputs
            # "bounds" : [-3,-3,-10,-0.1,-2,-7,   0,  -0.1,-0.35,0  ,  -1 ,-1,-5,
            #                     3, 3, 10,   4, 2, 7,  30,     1, 0.35,5  ,   1 , 1, 5]', 
            # bounds for nomalized state-inputs (bounds can be changed by changing
            # # normalization)
            "bounds" : np.array([
                    [-1,-1,-3, 0,-1,-1,   0,  -0.1,-1,0  ,  -1 ,-1,-5],
                    [1, 1, 3, 1, 1, 1,   1,     1, 1,1  ,   1 , 1, 5]
                ]), 

            ###########################################################################
            # Cost Parameters #########################################################
            ###########################################################################
            "qC" : 0.1, # contouring cost
            "qCNmult" : 10, # increase of terminal contouring cost
            "qL" : 1000, # lag cost
            "qVtheta" : 0.02, # theta maximization cost
            "qOmega" : 1e-5, # yaw rate regularization cost
            "qOmegaNmult" : 10, # yaw rate regularization cost

            "rD" : 1e-4, # cost on duty cycle (only used as regularization terms)
            "rDelta" : 1e-4, # cost on steering 
            "rVtheta" : 1e-4, # cost on virtual velocity

            "rdD" : 0.01, # cost on change of duty cycle
            "rdDelta" : 1, # cost on change of steering
            "rdVtheta" : 0.001, # cost on change of virtual velocity


            "q_eta" : 250, # cost on soft constraints (TODO implement soft constraints)

            "costScale" : 1 # scaling of the cost for better numerics
        }
        
    elif CarModel == "FullSize": #https://www.w3schools.com/python/python_polymorphism.asp
        
        ###########################################################################
        # MPC settings ############################################################
        ###########################################################################
    
        MPC_vars = {
    
            # prediction horizon
            "N" : 120,
            # sampling time
            "Ts" : 0.05,
            # used model (TODO incorparate new models)
            "ModelNo" : 2,
            # use bounds on all opt variables (TODO implement selective bounds)
            "fullBound" : 1,  
            ###########################################################################
            # state-input scaling #####################################################
            ###########################################################################
            # normalization matricies (scale states and inputs to ~ +/- 1
            "Tx" : np.diag(1./np.array([1,1,2*np.pi,10,10,5,10])),
            "Tu" : np.diag(1./np.array([1,0.5,10])),

            "invTx" : np.diag([1,1,2*np.pi,10,10,5,10]),
            "invTu" : np.diag([1,0.5,10]),

            "TDu" : np.eye(3),
            "invTDu" : np.eye(3),
            # identity matricies if inputs should not be normalized
            # "Tx" : eye(7),
            # "Tu" : eye(3),
            # 
            # "invTx" : eye(7),
            # "invTu" : eye(3),
            # 
            # "TDu" : eye(3),
            # "invTDu" : eye(3),
            ###########################################################################
            # state-input bounds ######################################################
            ###########################################################################

            # bounds for non-nomalized state-inputs
            # "bounds" : [-3,-3,-10,-0.1,-2,-7,   0,  -0.1,-0.35,0  ,  -1 ,-1,-5,
            #                     3, 3, 10,   4, 2, 7,  30,     1, 0.35,5  ,   1 , 1, 5]', 
            # bounds for nomalized state-inputs (bounds can be changed by changing
            # # normalization)
            "bounds" : np.array([
                [-1e4,-1e4,-3, 0.25,-3,-1,   0,    -1,-1, 0  ,  -0.25 ,-0.1,-10],
                [1e4, 1e4, 3,   10, 3, 1, 1e4,     1, 1,10  ,   0.25 , 0.1, 10]
            ]), 

            ###########################################################################
            # Cost Parameters #########################################################
            ###########################################################################
            "qC" : 0.01, # contouring cost
            "qCNmult" : 10000, # increase of terminal contouring cost
            "qL" : 1000, # lag cost
            "qVtheta" : 0.5, # theta maximization cost
            "qOmega" : 5e0, # yaw rate regularization cost
            "qOmegaNmult" : 1, # yaw rate regularization cost

            "rD" : 1e-4, # cost on duty cycle (only used as regularization terms)
            "rDelta" : 1e-4, # cost on steering 
            "rVtheta" : 1e-6, # cost on virtual velocity

            "rdD" : 0.1, # cost on change of duty cycle
            "rdDelta" : 1, # cost on change of steering
            "rdVtheta" : 0.1, # cost on change of virtual velocity

            "q_eta" : 250, # cost on soft constraints (TODO implement soft constraints)

            "costScale" : 0.01 # scaling of the cost for better numerics
    
        }
    
    else:
        print('ERROR: invalid model name')
        exit()

    return MPC_vars


