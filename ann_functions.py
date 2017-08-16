# A set of useful functions for training and forward propagation  
# of artificial neural nets
import numpy as np 





# initializes with random weights
def init_weights(R, S):
	# W = np.random.rand(R,S)
	# b = np.random.rand(S)
	W = np.random.randn(R,S)/np.sqrt(R+S) # Xavier initialization
	b = np.random.randn(S)
	return W.astype(np.float32), b.astype(np.float32)



# softmax function for multiclass. problems
def softmax(a):
	expA = np.exp(a)
	return expA / expA.sum(axis=1, keepdims=True)


# defines the cross-entropy cost function for binary classification
def cross_entropy_bin(T, pY):
	return (T*np.log(pY) + (1-T)*np.log(1-pY)).sum()


# defines the cross-entropy cost function for multiclass
def cross_entropy_multi(T, pY):
	N = len(T)
	return 1/N*(T*np.log(pY)).sum()


# same as 'cross_entropy_multi' but cost computation is faster when 
# dealing with large datasets
def cross_entropy_multi2(T, pY):	
	N = len(T)
	return 1/N*np.log(pY[np.arange(N), np.argmax(T, axis=1)]).sum()


# defines the residual squares sum cost function for regression 
def resid_squares(TRGT, Y):
	return 0.5*sum((TRGT-Y)**2)


# forward step without using an activation function
def forward_step(X,W,b):
	return X.dot(W) + b


# forward step using a sigmoid activation function
def forward_step_sigmoid(X,W,b):
	A = X.dot(W) + b
	return 1 / (1 + np.exp(-A))


# forward step using a 'tanh' activation function
def forward_step_tanh(X,W,b):
	return np.tanh(X.dot(W) + b)


# forward step using a 'relu' activation function
def forward_step_relu(X,W,b):
	A = X.dot(W)+b
	return A*(A>0)
	# return np.maximum(0, A)



