# Plots the resulting cost for different optimization schemes using 
# momentum batch GD.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals import joblib





def main():
# load test data
	npzfile = np.load('large_files/fer2test1k.npz')
	X = npzfile['Xtest']
	Y = npzfile['Ytest']



#######################################################################
# # This section analyzes general different optimization schemes when 
# considering different variable/adaptive learning rates approaches and 
# plain momentum batch GD.

	from ANN_relu_Multi_batch2 import ANN_relu
# load a plain momentum batch GD ANN_relu trained model
	model1 = joblib.load('large_files/ANN_relu_batch2.sav')
	cost1 = model1.cost
	print('Model 1 training time: {:.2f} min'.format(model1.elapsed_t), 
		'   Test accuracy: {:.3f}'.format(np.mean(Y==model1.predict(X))))

	
	from ANN_relu_Multi_batch2_expdecay import ANN_relu
# load a plain momentum batch GD with exponential l.r. decay 
# ANN_relu trained model
	model2 = joblib.load('large_files/ANN_relu_batch2_expdecay.sav')
	cost2 = model2.cost
	print('Model 2 training time: {:.2f} min'.format(model2.elapsed_t), 
		'   Test accuracy: {:.3f}'.format(np.mean(Y==model2.predict(X))))


	from ANN_relu_Multi_batch2_AdaGrad import ANN_relu
# load a plain momentum batch GD with AdaGrad ANN_relu trained 
# model
	model3 = joblib.load('large_files/ANN_relu_batch2_AdaGrad.sav')
	cost3 = model3.cost
	print('Model 3 training time: {:.2f} min'.format(model3.elapsed_t), 
		'   Test accuracy: {:.3f}'.format(np.mean(Y==model3.predict(X))))


	from ANN_relu_Multi_batch2_RMSprop import ANN_relu
# load a plain momentum batch GD with RMSprop ANN_relu trained
# model
	model4 = joblib.load('large_files/ANN_relu_batch2_RMSprop.sav')
	cost4 = model4.cost
	print('Model 4 training time: {:.2f} min'.format(model4.elapsed_t), 
		'   Test accuracy: {:.3f}'.format(np.mean(Y==model4.predict(X))))



# show the optimization cost evolution plots
	plt.plot(cost1, label='Plain Momentum')
	plt.plot(cost2, label='Plain Momentum + exponential decay')
	plt.plot(cost3, label='Plain Momentum + AdaGrad')
	plt.plot(cost4, label='Plain Momentum + RMSprop')

	plt.legend()
	plt.xlabel('Epochs')
	plt.ylabel('Cost value')
	plt.title('Evolution of the cost using Plain Momentum acceleration and variable/adaptive learning rates - ICML 2013 training dataset sample')
	plt.show()






if __name__ == '__main__':
	main()


