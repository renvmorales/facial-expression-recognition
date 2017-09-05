# This API implements the following tasks:
#  - Multi-layer 'relu' ANN models for multiclass problems
#  - Batch GD optimization using Nesterov momentum and AdaGrad
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



	def fit(self, X, Y, alpha=1e-6, reg=1e-4, mu=0.8, epochs=5000, 
		show_fig=False):		
		N, D = X.shape
		K = len(np.unique(Y))

		self.N = N  # this variable will be used for normalization
		self.D = D  # store the dimension of the training dataset
		# stores all hyperparameter values
		self.hyperparameters = {'alpha':alpha, 'reg':reg, 'epochs':epochs}
		

		# creates an indicator matrix for the target
		Trgt = np.zeros((N, K))
		Trgt[np.arange(N), Y.astype(np.int32)] = 1


		# creates a list with the number of hidden unities (+ input/output)
		hdn_unties = [D] + self.M + [K]
		self.W = []
		self.b = []
		self.v_W = []
		self.v_b = []
		self.cache_W = []
		self.cache_b = []
		# initializes all weights randomly
		for k in range(1,len(self.M)+2):
			W, b = init_weights(hdn_unties[k-1], hdn_unties[k])
			self.W.append(W)
			self.b.append(b)
			self.v_W.append(0)
			self.v_b.append(0)
			self.cache_W.append(0)
			self.cache_b.append(0)


		Ns = 100	# number of samples / batch
		Nbatch = 10#N/Ns  # number of batches
		J = np.zeros(epochs) # this array stores the cost with respect to each epoch
		start = time.time()	# <-- starts measuring the optimization time from this point on...	

		for i in range(epochs):  # optimization loop
			Xbuf, Ybuf = shuffle(X,Y)
			for j in range(int(Nbatch)):
				Xs = Xbuf[(j*Ns):(j*Ns+Ns),:] # input batch sample
				Trgt_s = np.zeros((Ns,K))
				Trgt_s[np.arange(Ns), Ybuf[(j*Ns):(j*Ns+Ns)].astype(np.int32)] = 1
				PY = self.forward(Xs)
				J[i] = J[i] + 1/Nbatch*cross_entropy_multi2(Trgt_s, PY)
				self.back_prop(Trgt_s, PY, alpha, reg, mu)
			if i % 100 == 0:
				print('Epoch:',i,' Cost: {:.4f}'.format(J[i]), 
					" Accuracy: {:1.4f}".format(np.mean(Y==self.predict(X))))


		end = time.time()
		self.elapsed_t = (end-start)/60 # total elapsed time
		self.cost = J # stores all cost values
		self.Ns = Ns
		self.Nbatch = Nbatch

		print('\nOptimization complete')
		print('\nElapsed time: {:.3f} min'.format(self.elapsed_t))


		# customized plot with the resulting cost values
		if show_fig: 
			plt.plot(J, label='Cost function J')
			plt.title('Evolution of the Cost through a AdaGrad Nesterov batch GD optimization     Total runtime: {:.3f} min'.format(self.elapsed_t)+'    Final Accuracy: {:.3f}'.format(np.mean(Y==self.predict(X))))
			plt.xlabel('Epochs')
			plt.ylabel('Cost')
			plt.legend()
			plt.show()
	



	def forward(self, X):
		self.Z = [X] # this list contains all hidden unities + input/output
		for i in range(0,len(self.M)):
			self.Z.append(forward_step_relu(self.Z[i], self.W[i], self.b[i]))
		self.Z.append(forward_step(self.Z[len(self.M)], self.W[len(self.M)], self.b[len(self.M)]))
		self.Z[-1] = softmax(self.Z[-1])
		return self.Z[-1]



	# updating weights using Nesterov momentum + AdaGrad!
	def back_prop(self, Y, PY, alpha, reg, mu):
		dZ = PY-Y
		Z = self.Z[:-1]
		Wbuf = self.W		
		eps = 1e-10
		for i in range(1,len(self.W)+1):
			grad_W = (Z[-i].T.dot(dZ) + reg/self.N*self.W[-i])
			self.cache_W[-i] += grad_W**2
			grad_b = (dZ.sum(axis=0) + reg/self.N*self.b[-i])
			self.cache_b[-i] += grad_b**2
			v_Wbuf = self.v_W[-i]
			self.v_W[-i] = mu*v_Wbuf - alpha*grad_W/(np.sqrt(self.cache_W[-i])+eps)
			self.W[-i] += -mu*v_Wbuf + (1+mu)*self.v_W[-i]
			v_bbuf = self.v_b[-i]
			self.v_b[-i] = mu*v_bbuf - alpha*grad_b/(np.sqrt(self.cache_b[-i])+eps)
			self.b[-i] += -mu*v_bbuf + (1+mu)*self.v_b[-i]
			dZ = dZ.dot(Wbuf[-i].T) * (Z[-i]>0)



	def predict(self, X):
		PY = self.forward(X)
		return np.argmax(PY, axis=1)




def main():
# number of samples for each class
	N_class = 5000 

# generate random 2-D points 
	X1 = np.random.randn(N_class,2)+np.array([2,2])
	X2 = np.random.randn(N_class,2)+np.array([-2,-2])
	X3 = np.random.randn(N_class,2)+np.array([-2,2])
	X4 = np.random.randn(N_class,2)+np.array([2,-2])
	X = np.vstack([X1,X2,X3,X4])

# labels associated to the input
	Y = np.array([0]*N_class+[1]*N_class+[2]*N_class+[3]*N_class)
	# Y = np.reshape(Y, (len(Y),1))


# scatter plot of original labeled data
	plt.scatter(X[:,0],X[:,1],c=Y,s=50,alpha=0.5)
	plt.show()



# create the ANN model with the specified 4 hidden layers
	model = ANN_relu([10,10,10,10])


# fit the model with the hyperparameters set	
	model.fit(X, Y, alpha=1e-3, epochs=5000, reg=0, mu=.9, show_fig=True)
	

# compute the model accuracy	
	Ypred = model.predict(X)
	print('\nFinal model accuracy: {:.4f}'.format(np.mean(Y==Ypred)))




if __name__ == '__main__':
    main()
