import numpy as np

def borderAdjustment(track2,ModelParams,safetyScaling):
    ###########################################################################
    #### shrink track on both sides with 'WidthCar' ###########################
    ###########################################################################
    WidthCar = 0.5*ModelParams["W"]*safetyScaling;
    #scale track to car size

    track2["center"] = track2["center"]*ModelParams["Scale"]
    track2["inner"] = track2["inner"]*ModelParams["Scale"]
    track2["outer"] = track2["outer"]*ModelParams["Scale"]

    # compute width of track (assumption track as a uniform width
    widthTrack = np.linalg.norm([track2["inner"][0,0]-track2["outer"][0,0],track2["inner"][1,0]-track2["outer"][1,0]])

    track = {
        "outer": np.zeros(track2["outer"].shape),
        "inner": np.zeros(track2["outer"].shape),
        "center": track2["center"]
    }


    for i in range(track2["outer"].shape[1]):
        x1 = track2["outer"][0,i];
        y1 = track2["outer"][1,i];
        x2 = track2["inner"][0,i];
        y2 = track2["inner"][1,i];
        # vector connecting right and left boundary
        numer= x2 - x1
        denom= y1 - y2 
        
        # shrinking ratio
        c =  WidthCar/widthTrack;
        d = -WidthCar/widthTrack;
        
        # shrink track
        track["outer"][0,i] = x1 + c*numer;
        track["inner"][0,i] = x2 - c*numer;
        track["outer"][1,i] = y1 + d*denom;
        track["inner"][1,i] = y2 - d*denom;    

    return track, track2
