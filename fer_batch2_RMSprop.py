# Plain momentum batch gradient descent with RMSprop used for training a 
# 'relu' ANN model on the ICML 2013 facial recognition dataset.
# 
# P.S: It can take very long time to run depending on the training  
# dataset size.
import numpy as np
import matplotlib.pyplot as plt
from ANN_relu_Multi_batch2_RMSprop import ANN_relu
from sklearn.externals import joblib




def main():
    npzfile = np.load('fer2train5k.npz')
    X = npzfile['Xtrain']
    Y = npzfile['Ytrain']
    


    print('Number of samples:',X.shape[0])
    print('Data dimension:', X.shape[1])
    print('Number of output classes:', len(np.unique(Y)))
    print('\n')


# create the ANN model with a specific number of hidden layers/unities
    model = ANN_relu([100,100,100,100])


# train the model with a hyperparameters setting
    model.fit(X, Y, alpha=1e-5, epochs=5000, reg=0, mu=0.95, show_fig=True)


# compute the prediciton/accuracy based on input data
    Ypred = model.predict(X)
    print('\nFinal model accuracy: {:.4f}'.format(np.mean(Y==Ypred)))



# save the model object to a file
    joblib.dump(model, 'ANN_relu_batch2_RMSprop.sav')
    






if __name__ == '__main__':
    main()

