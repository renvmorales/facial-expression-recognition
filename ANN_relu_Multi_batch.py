import numpy as np
import matplotlib.pyplot as plt
from  ann_functions import *
import time
from sklearn.utils import shuffle


class ANN_relu(object):
	def __init__(self, M):
		if isinstance(M, int):
			self.M = [M]
		else:
			self.M = M


	def fit(self, X, Y, alpha=1e-6, reg=1e-4, epochs=5000, 
		show_fig=False):
		N, D = X.shape
		K = len(np.unique(Y))
		self.N = N  # this variable will be useful for normalization later

		# creates an indicator matrix for the target
		Trgt = np.zeros((N, K))
		Trgt[np.arange(N), Y.astype(np.int32)] = 1


		# creates a list with the number of hidden unities (+ input/output)
		hdn_unties = [D] + self.M + [K]
		self.W = []
		self.b = []
		# initializes all weights randomly
		for k in range(1,len(self.M)+2):
			W, b = init_weights(hdn_unties[k-1], hdn_unties[k])
			self.W.append(W)
			self.b.append(b)


		start = time.time()
		J = np.zeros(epochs)
		Ns = 500	# number of samples / batch
		Nbat = N/Ns  # number of batches

		for i in range(epochs):
			Xbuf, Ybuf = shuffle(X,Y)
			for j in range(int(Nbat)):
				Xs = Xbuf[(j*Ns):(j*Ns+Ns)]
				Trgt_s = np.zeros((Ns,K))
				Trgt_s[np.arange(Ns), Ybuf[(j*Ns):(j*Ns+Ns)].astype(np.int32)] = 1
				# Trgt_s = Trgt[(j*Ns):(j*Ns+Ns)]
				PY = self.forward(Xs)
				J[i] = J[i] + cross_entropy_multi2(Trgt_s, PY)
				self.back_prop(Trgt_s, PY, alpha, reg)
			J[i] = J[i]/Nbat # computes the mean cost value
			if i % 100 == 0:
				print('Epoch:',i,' Cost: {:.4f}'.format(J[i]), 
					" Accuracy: {:1.4f}".format(np.mean(Y==np.argmax(self.forward(X),axis=1))))

		end = time.time()
		print('\nOptimization complete')
		print('\nElapsed time: {:.3f} min'.format((end-start)/60))

		if show_fig:
			plt.plot(J, label='Cost function J')
			plt.title('Evolution of the Cost through a Batch GD optimization     Total runtime: {:.2f} min'.format((end-start)/60)+'    Final Accuracy: {:.2f}'.format(np.mean(Y==np.argmax(self.forward(X),axis=1))))
			plt.xlabel('Epochs')
			plt.ylabel('Cost value')
			plt.legend()
			plt.show()
	


	def forward(self, X):
		self.Z = [X] # this will contain all hidden unities + output
		for i in range(0,len(self.M)):
			self.Z.append(forward_step_relu(self.Z[i], self.W[i], self.b[i]))
		self.Z.append(forward_step(self.Z[len(self.M)], self.W[len(self.M)], self.b[len(self.M)]))
		self.Z[-1] = softmax(self.Z[-1])
		return self.Z[-1]



	def back_prop(self, Y, PY, alpha, reg):
		dZ = PY-Y
		Z = self.Z[:-1]
		Wbuf = self.W
		for i in range(1,len(self.W)+1):
			self.W[-i] -= alpha * (Z[-i].T.dot(dZ) + reg/self.N*self.W[-i])
			self.b[-i] -= alpha * (dZ.sum(axis=0) + reg/self.N*self.b[-i])
			dZ = dZ.dot(Wbuf[-i].T) * 0.5*(1+np.sign(Z[-i]))
			# dZ = dZ.dot(Wbuf[-i].T) * (Z[-i]>0)



	def predict(self, X):
		PY = self.forward(X)
		return np.argmax(PY, axis=1)


def main():
# number of samples for each binary class
	N_class = 1000 

# generates random 2-D points 
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

# creates an ANN model with the specified 4 hidden layers
	model = ANN_relu([10,10,10,10])
# fits the model with the hyperparameters set	
	model.fit(X, Y, alpha=1e-5, epochs=10000, reg=0.01, show_fig=True)
	
	Ypred = model.predict(X)
	print('\nFinal model accuracy: {:.4f}'.format(np.mean(Y==Ypred)))



if __name__ == '__main__':
    main()
