import numpy as np
from scipy.integrate import odeint

def dy(y, t):
    return -y

def y(t, y0):
    yHat = odeint(dy, y0, t)
    return yHat

def generateData(in_width = 5):

    y0 = 5
    t  = np.linspace(0, 10, 20)
    yHat = y(t, y0)

    inpVals = []
    outVals   = []
    temp = np.zeros(in_width)
    for i in range(len(yHat.flatten()) -1):
        temp[:-1] = temp[1:]
        temp[-1]  = yHat[i]

        if i > in_width-2:
            inpVals.append( temp.copy() )
            outVals.append( yHat[i+1] )

    inpVals = np.array( inpVals )
    outVals = np.array( outVals )
    inpVals = inpVals.reshape( 
        inpVals.shape[0], inpVals.shape[1], 1 )
    
    return inpVals, outVals
