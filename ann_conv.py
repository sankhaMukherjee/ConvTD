import ConvTD as ctd
import numpy as np
from dataGeneration import simpleExp as sE
from dataGeneration import multiExp as mE
import matplotlib.pyplot as plt

from tqdm import tqdm

def solveModel():

    print('Generating Data')
    in_width = 5
    t = np.linspace(0, 2000, 2001)
    X, y = sE.generateData(in_width)
    X, y = mE.generateData(t, in_width, nPatients = 20)
    
    if True:
        in_channel = y.shape[1]
        channels   = [10, 8, in_channel]
        padding    = ['SAME', 'SAME', 'VALID']

        X0 = X[:1, :, :]

        model = ctd.ConvTD(in_width, in_channel, channels, padding)
        model.fit(X, y)
        yHat = model.predict( X0, nTerms=1500 )

        print(yHat)

    print('Generating Plots')
    if True:
        print(X.shape)
        print(y.shape)

        rY, cY = y.shape

        for i in tqdm(range(1)):
            plt.clf()
            for j in range(cY):
                plt.subplot(310+j+1)
                yTP = y[(rY//20)*i : (rY//20)*(i+1)-1, j]
                plt.plot(yTP, label='actual')
                plt.plot(yHat[:, j], label='Calculated')
            plt.legend(loc='upper right')
            plt.savefig( 'img/y_{:05}.png'.format(i) )

            plt.close()

    return

def main():

    solveModel()
    return

if __name__ == '__main__':
    main()
