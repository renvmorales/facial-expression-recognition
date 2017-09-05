# Plots the resulting cost for different optimization schemes.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals import joblib





def main():
# load test data
	npzfile = np.load('fer2test1k.npz')
	X = npzfile['Xtest']
	Y = npzfile['Ytest']



##################################################################
# # This section analyzes general different optimization schemes
	from ANN_relu_Multi import ANN_relu
# loades Standard GD ANN_relu model
	model1 = joblib.load('ANN_relu.sav')
	cost1 = model1.cost
	print('Model 1 training time: {:.2f} min'.format(model1.elapsed_t), 
		'   Test accuracy: {:.3f}'.format(np.mean(Y==model1.predict(X))))


	from ANN_relu_Multi_batch import ANN_relu
# loades Batch GD ANN_relu model
	model2 = joblib.load('ANN_relu_batch.sav')
	cost2 = model2.cost
	print('Model 2 training time: {:.2f} min'.format(model2.elapsed_t), 
		'   Test accuracy: {:.3f}'.format(np.mean(Y==model2.predict(X))))


	from ANN_relu_Multi_batch2 import ANN_relu
# loades Plain momentum batch GD ANN_relu model
	model3 = joblib.load('ANN_relu_batch2.sav')
	cost3 = model3.cost
	print('Model 3 training time: {:.2f} min'.format(model3.elapsed_t), 
		'   Test accuracy: {:.3f}'.format(np.mean(Y==model3.predict(X))))


	from ANN_relu_Multi_batch3 import ANN_relu
# loades Nesterov Momentum batch GD ANN_relu model
	model4 = joblib.load('ANN_relu_batch3.sav')
	cost4 = model4.cost
	print('Model 4 training time: {:.2f} min'.format(model4.elapsed_t), 
		'   Test accuracy: {:.3f}'.format(np.mean(Y==model4.predict(X))))




# # Stochastic results are too noisy so the visualization isn't good...
# 	from ANN_relu_Multi_SGD import ANN_relu
# # loades Stochastic GD ANN_relu model
# 	model5 = joblib.load('fer_ANN_relu_SGD.sav')
# 	cost5 = model5.cost



# show the optimization cost evolution plots
	plt.plot(cost1, label='Standard GD')
	plt.plot(cost2, label='Batch GD')
	plt.plot(cost3, label='Momentum batch GD')
	plt.plot(cost4, label='Nesterov batch GD')
	# plt.plot(cost5, label='Stochastic GD')

	plt.legend()
	plt.xlabel('Epochs')
	plt.ylabel('Cost value')	
	plt.title('Evolution of the cost using different optimization schemes - ICML 2013 training dataset sample')
	plt.show()





# ######################################################################
# # Uncomment from this point on to visualize Nesterov with different 
# # variable/adaptive learning rates methods 
# 	from ANN_relu_Multi_batch3 import ANN_relu
# # loades Nesterov Momentum batch GD ANN_relu model
# 	model1 = joblib.load('ANN_relu_batch3.sav')
# 	cost1 = model1.cost
# 	print('Model 1 training time: {:.2f} min'.format(model1.elapsed_t), 
# 		'   Test accuracy: {:.3f}'.format(np.mean(Y==model1.predict(X))))

	
# 	from ANN_relu_Multi_batch3_expdecay import ANN_relu
# # loades Nesterov Momentum batch GD ANN_relu model
# 	model2 = joblib.load('ANN_relu_batch3_expdecay.sav')
# 	cost2 = model2.cost
# 	print('Model 2 training time: {:.2f} min'.format(model2.elapsed_t), 
# 		'   Test accuracy: {:.3f}'.format(np.mean(Y==model2.predict(X))))


# 	from ANN_relu_Multi_batch3_AdaGrad import ANN_relu
# # loades Nesterov Momentum batch GD ANN_relu model
# 	model3 = joblib.load('ANN_relu_batch3_AdaGrad.sav')
# 	cost3 = model3.cost
# 	print('Model 3 training time: {:.2f} min'.format(model3.elapsed_t), 
# 		'   Test accuracy: {:.3f}'.format(np.mean(Y==model3.predict(X))))


# 	from ANN_relu_Multi_batch3_RMSprop import ANN_relu
# # loades Nesterov Momentum batch GD ANN_relu model
# 	model4 = joblib.load('ANN_relu_batch3_RMSprop.sav')
# 	cost4 = model4.cost
# 	print('Model 4 training time: {:.2f} min'.format(model4.elapsed_t), 
# 		'   Test accuracy: {:.3f}'.format(np.mean(Y==model4.predict(X))))



# # show the optimization cost evolution plots
# 	plt.plot(cost1, label='Nesterov')
# 	plt.plot(cost2, label='Nesterov + exponential decay')
# 	plt.plot(cost3, label='Nesterov + AdaGrad')
# 	plt.plot(cost4, label='Nesterov + RMSprop')

# 	plt.legend()
# 	plt.xlabel('Epochs')
# 	plt.ylabel('Cost value')
# 	plt.title('Evolution of the cost using Nesterov acceleration and variable/adaptive learning rates - ICML 2013 training dataset sample')
# 	plt.show()






if __name__ == '__main__':
	main()


