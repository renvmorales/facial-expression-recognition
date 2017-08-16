# Using a full-gradient ascend ANN model with 'relu' activation function. 
# Implemented from scratch (using mainly Numpy).
# P.S: It can take very long time to run depending if the majority of the 
# dataset is used for training.
import numpy as np
import matplotlib.pyplot as plt
from ANN_relu_Multi import ANN_relu
from sklearn.utils import shuffle



def getData(balance_ones=True):
    # images are 48x48 = 2304 size vectors
    # N = 35887
    Y = []
    X = []
    first = True
#  data available at https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data
    for line in open('fer2013.csv','r'):
        if first:
            first = False
        else:
            row = line.split(',')
            Y.append(int(row[0]))
            X.append([int(p) for p in row[1].split()])
    
    X, Y = np.array(X) / 255.0, np.array(Y)

    if balance_ones:
        # balance the 1 class
        X0, Y0 = X[Y!=1, :], Y[Y!=1]
        X1 = X[Y==1, :]
        X1 = np.repeat(X1, 9, axis=0)
        X = np.vstack([X0, X1])
        Y = np.concatenate((Y0, [1]*len(X1)))

    X, Y = shuffle(X, Y)
    # selects only the last 1000 samples
    Xvalid, Yvalid = X[-2000:], Y[-2000:] 

    return Xvalid, Yvalid



def main():
    X, Y = getData()
    print('Number of samples:',X.shape[0])
    print('Data dimension:', X.shape[1])

    model = ANN_relu([100,100,100,100])
    # model = ANN_relu([200,200,200])
    # model.fit(X, Y, learning_rate=1*10e-7, epochs=10000, reg=0, show_fig=True)
    model.fit(X, Y, alpha=1e-6, epochs=20000, reg=1e-2, show_fig=True)
    # # print('The total score is: ', model.score(X, Y))
    Ypred = model.predict(X)
    print('\nFinal model accuracy: {:.4f}'.format(np.mean(Y==Ypred)))



    # save the model object to a file
    from sklearn.externals import joblib
    joblib.dump(model, 'fer_ANN_relu.sav')
    
    # import pickle
    # pickle.dump(model, open('ANN_relu.sav', 'wb'))



if __name__ == '__main__':
    main()

