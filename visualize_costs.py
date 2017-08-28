# Plots the resulting cost for different optimization schemes.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals import joblib





def main():
	from ANN_relu_Multi import ANN_relu
# loades Standard GD ANN_relu model
	model1 = joblib.load('fer_ANN_relu.sav')
	cost1 = model1.cost


	from ANN_relu_Multi_batch import ANN_relu
# loades Batch GD ANN_relu model
	model2 = joblib.load('fer_ANN_relu_batch.sav')
	cost2 = model2.cost


	from ANN_relu_Multi_batch2 import ANN_relu
# loades Momentum-batch GD ANN_relu model
	model3 = joblib.load('fer_ANN_relu_batch2.sav')
	cost3 = model3.cost


	from ANN_relu_Multi_batch3 import ANN_relu
# loades Nesterov Momentum-batch GD ANN_relu model
	model4 = joblib.load('fer_ANN_relu_batch3.sav')
	cost4 = model4.cost



# Stochastic results are too noisy so the visualization isn't good...
# 	from ANN_relu_Multi_SGD import ANN_relu
# # loades Stochastic GD ANN_relu model
# 	model4 = joblib.load('fer_ANN_relu_SGD.sav')
# 	cost4 = model4.cost


	plt.plot(cost1, label='Standard GD')
	plt.plot(cost2, label='Batch GD')
	plt.plot(cost3, label='Momentum-batch GD')
	plt.plot(cost4, label='Nesterov momentum')
	# plt.plot(cost4, label='Stochastic GD')


	plt.legend()
	plt.xlabel('Epochs')
	plt.ylabel('Cost value')
	plt.title('Evolution of the cost using different optimization schemes - ICML 2013 training dataset sample')
	plt.show()



if __name__ == '__main__':
    main()


