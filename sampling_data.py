# This code applies data preprocessing: 
#   - Balance the '1' class so the all training examples per class are 
# approximately the same (prevents bias results from the accuracy statistics!)
#   - Extract one sample from the ICML 2013 dataset.
#   - Save this sample to file output for a subsequent ANN training model.
# The 'sample size' needs to be defined.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle





def getData(Ns, balance_ones=True):
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
    
    if isinstance(Ns, int):
        Xvalid, Yvalid = X[-Ns:,:], Y[-Ns:] # extract a sample of size Ns
        return Xvalid, Yvalid
    else:
        return X, Y




def main():	
	X, Y = getData(Ns=4000) # the argument specifies the sample size
	print('Total input samples:',X.shape[0])
	print('Data dimension:', X.shape[1])
	print('Number of output classes:', len(np.unique(Y)))
	print('Sampling Data complete.')
	print('\n')

	# np.savez('fer_sample', X=X, Y=Y)



if __name__ == '__main__':
    main() 
