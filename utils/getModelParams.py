import numpy as np

def getModelParams(ModelNo):

    if ModelNo == 1:

        ModelParams = {

            "ModelNo": 1,
            "Scale": 1,#scale of the car (1 is a 1:43 scale car)
            
            "sx": 7, #number of states
            "su": 3, #number of inputs
            "nx": 7, #number of states
            "nu": 3, #number of inputs
            
            "stateindex_x": 1, #x position
            "stateindex_y": 2, #y position
            "stateindex_phi": 3, #orientation
            "stateindex_vx": 4, #longitudinal velocity
            "stateindex_vy": 5, #lateral velocity
            "stateindex_omega": 6, #yaw rate
            "stateindex_theta": 7, #virtual position

            "inputindex_D": 1, #duty cycle
            "inputindex_delta": 2, #steering angle
            "inputindex_vtheta": 3, #virtual speed
            
            "m":  0.041,
            "Iz":  27.8e-6,
            "lf":  0.029,
            "lr":  0.033,

            "Cm1": 0.287,
            "Cm2": 0.0545,
            "Cr0": 0.0518,
            "Cr2": 0.00035,

            "Br":  3.3852,
            "Cr":  1.2691,
            "Dr":  0.1737,

            "Bf":  2.579,
            "Cf":  1.2,
            "Df":  0.192,
            
            "L":  0.12,
            "W":  0.06,
        }
        
    elif ModelNo==2:

        ModelParams = {

            "ModelNo": 2,
            "Scale": 43,#scale of the car (1 is a 1:43 scale car)
            
            "sx": 7, #number of states
            "su": 3, #number of inputs
            "nx": 7, #number of states
            "nu": 3, #number of inputs
            
            "stateindex_x": 1, #x position
            "stateindex_y": 2, #y position
            "stateindex_phi": 3, #orientation
            "stateindex_vx": 4, #longitudinal velocity
            "stateindex_vy": 5, #lateral velocity
            "stateindex_omega": 6, #yaw rate
            "stateindex_theta": 7, #virtual position

            "inputindex_D": 1, #duty cycle
            "inputindex_delta": 2, #steering angle
            "inputindex_vtheta": 3, #virtual speed
            
            "m":  1573,
            "Iz":  2873,
            "lf":  1.35,
            "lr":  1.35,
            
            "Cm1": 17303,
            "Cm2": 175,
            "Cr0": 120,
            "Cr2": 0.5*1.225*0.35*2.5,#0.5*rho*cd*A

            "Br":  13,
            "Cr":  2,
            
            "Bf":  13,
            "Cf":  2,
            
            "L":  5,
            "W":  2.5,

            "Wight_f": 1,
            "Wight_r": 1,
            "Dr":  1,
            "Df":  1
        }
    
        ModelParams["Wight_f"]=  ModelParams["lr"]/(ModelParams["lf"]+ModelParams["lr"])
        ModelParams["Wight_r"]=  ModelParams["lf"]/(ModelParams["lf"]+ModelParams["lr"])    
        ModelParams["Dr"]=  ModelParams["Wight_f"]*ModelParams["m"]*9.81*1.2
        ModelParams["Df"]=  ModelParams["Wight_r"]*ModelParams["m"]*9.81*1.2
    else:
        print("ERROR: ModelNo invalid")
        exit()

    return ModelParams
