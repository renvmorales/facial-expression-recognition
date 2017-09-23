# This API implements the following tasks:
#  - Multi-layer 'relu' ANN models for multiclass problems
#  - Batch GD optimization using plain momentum
#  - Exponential learning rate decay
# 
# Implementation from scratch (mainly using Numpy).
import numpy as np
import matplotlib.pyplot as plt
from  ann_functions import *
import time
from sklearn.utils import shuffle



class ANN_relu(object):
	def __init__(self, M):
		# this assures that all hidden unities are stored in a list
		if isinstance(M, int):
			self.M = [M]  # in case there is a single hidden layer...
		else:
			self.M = M



	def fit(self, X, Y, alpha=1e-3, reg=1e-4, mu=0.8, epochs=5000, 
		show_fig=False, Nbatch=10, decay=2e-3):		
		N, D = X.shape
		K = len(np.unique(Y))
		batch_sz = int(N/Nbatch)  # batch size

		self.N = N  # this variable will be used for normalization
		self.D = D  # store the dimension of the training dataset
		self.K = K  # output dimension
		self.batch_sz = batch_sz

		# stores all hyperparameter values
		self.hyperparameters = {'alpha':alpha, 'reg':reg, 'mu':mu, 
		'epochs':epochs, 'Nbatch':Nbatch, 'decay':decay}
		

		# creates an indicator matrix for the target
		Trgt = np.zeros((N, K))
		Trgt[np.arange(N), Y.astype(np.int32)] = 1


		# creates a list with the number of hidden unities (+ input/output)
		hdn_unties = [D] + self.M + [K]
		self.W = []
		self.b = []
		self.dW = []
		self.db = []
		# initializes all weights randomly
		for k in range(1,len(self.M)+2):
			W, b = init_weights(hdn_unties[k-1], hdn_unties[k])
			self.W.append(W)
			self.b.append(b)
			self.dW.append(0)
			self.db.append(0)



		J = [] # this list stores the cost with respect to each epoch
		start = time.time()	# <-- starts measuring the optimization time from this point on...	
		alpha_0 = alpha # initial alpha before decay

		for i in range(int(epochs/Nbatch)):  # optimization loop
			Xbuf, Trgtbuf = shuffle(X, Trgt)
			for j in range(int(Nbatch)):
				X_b = Xbuf[(j*batch_sz):(j*batch_sz+batch_sz),:] # input batch sample
				Trgt_b = Trgtbuf[(j*batch_sz):(j*batch_sz+batch_sz),:] # output batch sample
				PY = self.forward(X_b)
				J.append(cross_entropy_multi2(Trgt_b, PY))
				alpha = alpha_0*np.exp(-decay*i)
				self.back_prop(Trgt_b, PY, alpha, reg, mu)
				if (i*Nbatch+j+1) % 100 == 0:
					print("Epoch: %d  Cost: %.4f  Accuracy: %.4f" % (i*Nbatch+j+1, J[-1], 
						np.mean(Y==self.predict(X))) )

		end = time.time()
		self.elapsed_t = (end-start)/60 # total elapsed time
		self.cost = J # stores all cost values


		print('\nOptimization complete')
		print('\nElapsed time: {:.3f} min'.format(self.elapsed_t))


		# customized plot with the resulting cost values
		if show_fig: 
			plt.plot(J, label='Cost function J')
			plt.title('Evolution of the Cost through a Momentum batch GD optimization with exponential l.r. decay       Total runtime: {:.3f} min'.format(self.elapsed_t)+'    Final Accuracy: {:.3f}'.format(np.mean(Y==self.predict(X))))
			plt.xlabel('Epochs')
			plt.ylabel('Cost')
			plt.legend()
			plt.show()
	



	def forward(self, X):
		self.Z = [X] # this list contains all hidden unities + input/output
		for i in range(0,len(self.M)):
			self.Z.append(relu(self.Z[i].dot(self.W[i]) + self.b[i]))
		self.Z.append(softmax(self.Z[len(self.M)].dot(self.W[len(self.M)]) + self.b[len(self.M)]))
		return self.Z[-1]



	# updating weights using plain momentum!
	def back_prop(self, Y, PY, alpha, reg, mu):
		N = len(Y)
		dZ = (PY-Y)/N
		Z = self.Z[:-1]
		Wbuf = self.W
		for i in range(1,len(self.W)+1):
			self.dW[-i] = mu*self.dW[-i] - alpha*(Z[-i].T.dot(dZ) + reg/(2*N)*self.W[-i])
			self.W[-i] += self.dW[-i]
			self.db[-i] = mu*self.db[-i] - alpha * (dZ.sum(axis=0) + reg/(2*N)*self.b[-i])
			self.b[-i] += self.db[-i]			
			dZ = dZ.dot(Wbuf[-i].T) * (Z[-i]>0)



	def predict(self, X):
		PY = self.forward(X)
		return np.argmax(PY, axis=1)





def main():
# number of samples for each class
	N_class = 1000 


# generate random 2-D points 
	X1 = np.random.randn(N_class,2)+np.array([2,2])
	X2 = np.random.randn(N_class,2)+np.array([-2,-2])
	X3 = np.random.randn(N_class,2)+np.array([-2,2])
	X4 = np.random.randn(N_class,2)+np.array([2,-2])
	X = np.vstack([X1,X2,X3,X4])


# labels associated to the input
	Y = np.array([0]*N_class+[1]*N_class+[2]*N_class+[3]*N_class)
	# Y = np.reshape(Y, (len(Y),1))


# general data information for the training process
	print('Total input samples:',X.shape[0])
	print('Data dimension:',X.shape[1])
	print('Number of output classes:',len(np.unique(Y)))
	print('\n')


# scatter plot of original labeled data
	plt.scatter(X[:,0],X[:,1],c=Y,s=50,alpha=0.5)
	plt.show()


# create an ANN model with the specified 4 hidden layers
	model = ANN_relu([10,10,10,10])


# fit the model with the hyperparameters set	
	model.fit(X, Y, alpha=1e-3, epochs=5000, reg=0.01, mu=0.9, Nbatch=20, 
		show_fig=True)
	

# compute the model accuracy	
	Ypred = model.predict(X)
	print('\nFinal model accuracy: {:.4f}'.format(np.mean(Y==Ypred)))




if __name__ == '__main__':
	main()
