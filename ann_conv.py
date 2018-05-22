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

    print(X.shape)
    print(y.shape)

    rY, cY = y.shape
    yHats = []
    
    if True:
        in_channel = y.shape[1]
        channels   = [10, 8, in_channel]
        padding    = ['SAME', 'SAME', 'VALID']


        print('Fitting a model ...')
        model = ctd.ConvTD(in_width, in_channel, channels, padding)
        model.fit(X, y)

        print('Making Predictions ...')
        for i in range(5):
            pos = (rY//20)*i
            X0 = X[pos:pos+1, :, :]
            yHat = model.predict( X0, nTerms=1500 )
            print(yHat)
            yHats.append(yHat)

    print('Generating Plots')
    if True:

        yLims = [(0, 1.5), (0, 0.5), (0, 1)]
        for i in tqdm(range(5)):
            plt.clf()
            for j in range(cY):
                plt.subplot(310+j+1)
                yTP = y[(rY//20)*i : (rY//20)*(i+1)-1, j]
                plt.plot(yTP, label='actual')
                plt.plot(yHats[i][:, j], label='Calculated')
                plt.ylim(yLims[j])

            plt.legend(loc='upper right')
            plt.savefig( 'img/y_{:05}.png'.format(i) )

            plt.close()

    return

def main():

    solveModel()
    return

if __name__ == '__main__':
    main()
