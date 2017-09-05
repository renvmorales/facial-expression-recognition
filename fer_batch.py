# Batch gradient descent used for training a 'relu' ANN model on the
# ICML 2013 facial recognition dataset.
# 
# P.S: It can take very long time to run depending on the training
# dataset size.
import numpy as np
import matplotlib.pyplot as plt
from ANN_relu_Multi_batch import ANN_relu




def main():
    npzfile = np.load('fer2train5k.npz')
    X = npzfile['Xtrain']
    Y = npzfile['Ytrain']



    print('Number of samples:',X.shape[0])
    print('Data dimension:', X.shape[1])
    print('Number of output classes:', len(np.unique(Y)))
    print('\n')


    model = ANN_relu([100,100,100,100])

    model.fit(X, Y, alpha=1e-6, epochs=10000, reg=1e-2, show_fig=True)

    Ypred = model.predict(X)
    print('\nFinal model accuracy: {:.4f}'.format(np.mean(Y==Ypred)))



    # save the model object to a file
    from sklearn.externals import joblib
    joblib.dump(model, 'ANN_relu_batch.sav')
    
    # import pickle
    # pickle.dump(model, open('ANN_relu.sav', 'wb'))




if __name__ == '__main__':
    main()

