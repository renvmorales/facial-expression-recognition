# Plain momentum batch gradient descend ANN model using a 'relu' 
# activation function for the ICML 2013 facial recognition dataset.
# P.S: It can take very long time to run depending if the majority of the 
# dataset is used for training.
import numpy as np
import matplotlib.pyplot as plt
from ANN_relu_Multi_batch2 import ANN_relu
# from sklearn.utils import shuffle



def main():
    npzfile = np.load('fer_sample.npz')
    X = npzfile['X']
    Y = npzfile['Y']


    print('Number of samples:',X.shape[0])
    print('Data dimension:', X.shape[1])
    print('Number of output classes:', len(np.unique(Y)))
    print('\n')


    model = ANN_relu([100,100,100,100])
    # model = ANN_relu([200,200,200])
    # model.fit(X, Y, learning_rate=1*10e-7, epochs=10000, reg=0, show_fig=True)
    model.fit(X, Y, alpha=1e-6, epochs=10000, reg=1e-2, mu=0.9, show_fig=True)
    # # print('The total score is: ', model.score(X, Y))
    Ypred = model.predict(X)
    print('\nFinal model accuracy: {:.4f}'.format(np.mean(Y==Ypred)))



    # save the model object to a file
    from sklearn.externals import joblib
    joblib.dump(model, 'fer_ANN_relu_batch2.sav')
    
    # import pickle
    # pickle.dump(model, open('ANN_relu.sav', 'wb'))



if __name__ == '__main__':
    main()

