import json, os
import numpy      as np
import tensorflow as tf

from tqdm import tqdm
from datetime import datetime as dt

class ConvTD:

    def __init__(self, in_width, in_channel, channels, padding):

        self.restorePoint = None
        self.errVals      = None

        self.in_channel = in_channel
        self.channels   = channels
        self.padding    = padding

        self.inp = tf.placeholder( 
                dtype = tf.float32, 
                shape = (None, in_width, in_channel))
        self.out = tf.placeholder( 
                    dtype = tf.float32, 
                    shape = (None, in_channel))

        prevChannel = in_channel
        for i, (c, p) in enumerate(zip(channels, padding)):
            if i == 0:
                v1 = tf.nn.conv1d( self.inp, 
                    tf.Variable(tf.random_normal(
                        # [filter_width, in_channels, out_channels],
                        shape=(in_width, prevChannel, c), 
                        stddev=0.1, 
                        dtype=tf.float32)),
                    stride      = 1,
                    padding     = p,
                    data_format = 'NWC' )
            else:
                v1 = tf.nn.conv1d( v1, 
                    tf.Variable(tf.random_normal(
                        # [filter_width, in_channels, out_channels],
                        shape=(in_width, prevChannel, c), 
                        stddev=0.1, 
                        dtype=tf.float32)),
                    stride      = 1,
                    padding     = p,
                    data_format = 'NWC' )

            prevChannel = c

        self.yHat = tf.reshape(v1, (-1, channels[-1]))
        self.err  = tf.reduce_mean((self.out - self.yHat)**2, axis=0)
        self.err  = tf.reduce_mean(self.err)

        self.opt = tf.train.AdamOptimizer(learning_rate=1e-3).minimize( self.err )

        # Couple of variables to make things easier
        self.saver = tf.train.Saver(var_list=tf.trainable_variables())
        self.init = tf.global_variables_initializer()

        return

    def saveModel(self, sess):
        now = dt.now().strftime('%Y-%m-%d--%H-%M-%S')
        modelFolder = 'models/{}'.format(now)
        os.makedirs(modelFolder)

        params = {
            'in_channel' : self.in_channel,
            'channels'   : self.channels  ,
            'padding'    : self.padding   }
        
        paramsJson = json.dumps(params)
        with open(os.path.join(modelFolder, 'params.json'), 'w') as f:
            f.write(paramsJson)

        path = self.saver.save(sess, os.path.join(modelFolder, 'model.ckpt'))
        self.restorePoint = path
        print('Model saved at: {}'.format(path))

    def restoreModel(self, sess, restoreLatest=True, restorePoint=None):

        if restorePoint is not None:
            try:
                self.saver.restore(sess, restorePoint)
                return True
            except Exception as e:
                print('Unable to restore to state [{}]: {}'.format( 
                    str(restorePoint), str(e) ))
                return False
        

        if restoreLatest and (self.restorePoint is not None):
            try:
                self.saver.restore(sess, self.restorePoint)
                return True
            except Exception as e:
                print('Unable to restore to state [{}]: {}'.format( 
                    str(restorePoint), str(e) ))

        return False

    def fit(self, X, y, Niter=1000, restore=False, restoreLatest=True, restorePoint=None):

        with tf.Session() as sess:
            sess.run(self.init)

            if restore:
                r = self.restoreModel(
                    sess, 
                    restoreLatest = restoreLatest, 
                    restorePoint =  restorePoint   )
                if not r:
                    print('Session has not been restored ')
                    # you can chose to not fit data here

            # Fit the data
            for i in tqdm(range(Niter)):
                _, errVal = sess.run( [self.opt, self.err], feed_dict = {
                            self.inp: X ,
                            self.out: y})

            # Save the model for future use
            path = self.saveModel(sess)

        return path

    def predict(self, X0, nTerms=10):

        finalResult = None

        with tf.Session() as sess:

            r = self.restoreModel(
                sess, restoreLatest = True)
            if not r:
                print('Session has not been restored ')
                print('The data will be terminated ...')
                return None

            for i in range(nTerms):
                y1 = sess.run(self.yHat, feed_dict = {self.inp: X0})
                X0[:,:-1,:] = X0[:,1:,:] 
                X0[:,-1,:] = y1

                if i == 0:
                    finalResult = y1.copy()
                else:
                    finalResult = np.concatenate((finalResult, y1))

        finalResult = np.array(finalResult)

        return finalResult
