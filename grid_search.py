# Grid search to perform n-fold cross-validation and determine the
# best set of parameters for the model.
import numpy as np
import matplotlib.pyplot as plt
from  ann_functions import *
import time
from sklearn.utils import shuffle




# class grid_search(object):
# 	def __init__(self, Nfold, Nparam):
# 		self.Nfold = int(Nfold)
# 		self.Nparam = Nparam


# 	def begin(self, swap_list, model, X, Y):
# 		X, Y = shuffle(X,Y)
# 		Xb = []
# 		Yb = []
# 		for i in range(self.Nfold):
# 			Xb.append(Xs[n*int(len(X)/self.Nfold):n*int(len(X)/self.Nfold) + int(len(X)/self.Nfold)])
# 			Yb.append(Xs[n*int(len(X)/self.Nfold):n*int(len(X)/self.Nfold) + int(len(X)/self.Nfold)])
# 		blk = np.arange(self.Nfold)
# 		# nfold_cv = np.zeros((self.Nparam,self.Nfold))
# 		for n in range(self.Nfold):
# 			slct = blk[blk!=n]
# 			for b in list(blk):
# 				for s in range(len(swap_list)):
# 				for i in range(len(swap_list[s])):
# 					model.fit(X=Xb, Y=Yb, ls=swap_list)
# 					Yred = model.predict(Xb)





def main():
	npzfile = np.load('large_files/fer2train5k.npz')
	X = npzfile['Xtrain']
	Y = npzfile['Ytrain']






if __name__ == '__main__':
	main()
