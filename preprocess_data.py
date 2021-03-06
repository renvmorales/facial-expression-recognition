# This code applies data preprocessing: 
#   - Balance the '1' class so the all training examples per class are 
# approximately the same (prevents bias results from the accuracy statistics!)
#   - Extract one sample from the ICML 2013 dataset.
#   - Save this sample to file output for a subsequent ANN training model.
# The 'sample size' needs to be defined.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle





def getData(Ns=1000, balance_ones=True):
	# images are 48x48 = 2304 size vectors
	# N = 35887
	Y = []
	X = []
	first = True

#  data available at https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data
	for line in open('large_files/fer2013.csv','r'):
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
		Xtrain, Ytrain = X[-Ns:,:], Y[-Ns:] # extract a sample of size Ns
		Xtest, Ytest = X[:1000,:], Y[:1000] # test sample data 
		return Xtrain, Ytrain, Xtest, Ytest
	else:
		return X, Y






def main():	
# the argument specifies the sample size
	Xtrain, Ytrain, Xtest, Ytest = getData(Ns=5000)
	print('Total input samples:',Xtrain.shape[0])
	print('Data dimension:', Xtrain.shape[1])
	print('Number of output classes:', len(np.unique(Ytrain)))
	print('Sampling Data complete.')
	print('\n')
	
	# save variables to a file
	np.savez('large_files/fer2train5k', Xtrain=Xtrain, Ytrain=Ytrain)
	np.savez('large_files/fer2test1k', Xtest=Xtest, Ytest=Ytest)



if __name__ == '__main__':
	main() 
