import numpy as np
from scipy.integrate import odeint

def dy(y, t):

    p = {}
    p['k1'] = 0.005
    p['k2'] = 0.05
    p['k3'] = 0.003
    p['d']  = 0.01
    p['km'] = 2.0

    dydt = np.zeros(y.shape)

    dydt[0] = - p['k1']*y[0]
    dydt[1] =   ( p['k1']*y[0] - p['k2']*y[1] + p['k3']*y[2] 
                            - p['d']*y[1]/(p['km']+y[1]) )
    dydt[2] =   p['k2']*y[1] - p['k3']*y[2]

    return dydt

def yFunc(t, y0):
    yHat = odeint(dy, [y0, 0, 0], t)
    return yHat

def generateDataOne(y0, t, in_width = 5):

    data = yFunc(t, y0)
    
    rs, cs = data.shape
    c = 0
    
    allXc = []
    allyc = []
    for c in range(cs):
        Xc = []
        yc = []
        tempX = np.zeros(in_width)
        for i, v in enumerate(data[:-1, c]):
            tempX[:-1] = tempX[1:]
            tempX[-1] = v
            Xc.append( tempX.copy() )
            yc.append( data[i+1, c] )

        Xc = np.array(Xc)
        yc = np.array(yc)

        Xc = Xc[in_width-1:]
        yc = yc[in_width-1:]

        # time/batch, convolution width, channels
        Xc = Xc.reshape( -1, in_width, 1 )
        yc = yc.reshape( -1, 1 )

        allXc.append( Xc )
        allyc.append( yc )

        
    allXc = np.concatenate( allXc, 2 )
    allyc = np.concatenate( allyc, 1 )
    
    return allXc, allyc

def generateData(t, in_width = 5, nPatients=10):

    X, y = [], []
    for y0 in np.random.uniform(0.5, 1.5, nPatients):
        allX, ally = generateDataOne(y0, t, in_width)
        X.append( allX.copy() )
        y.append( ally.copy() )

    X = np.concatenate(X, 0)
    y = np.concatenate(y, 0)

    return X, y

def main():

    in_width = 5

    t = np.linspace(0, 10, 17)
    X, y = generateData(t, in_width)

    print(X.shape)
    print(y.shape)
    
    return

if __name__ == '__main__':
    main()