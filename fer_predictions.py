import numpy as np
import matplotlib.pyplot as plt
from ANN_relu_Multi import ANN_relu
from sklearn.externals import joblib
from facial_expr_recognition import getData

# this code loades the model for a later prediction
model = joblib.load('fer_ANN_relu.sav')



# a mapping list for each Face recognition class
label_map = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

X, Y = getData()

Ypred = model.predict(X)

while True:
    for i in range(7):
        # x, y = X[Y==i], Y[Y==i]
        N = len(Y)
        j = np.random.choice(N)
        plt.imshow(X[j].reshape(48, 48), cmap='gray')
        plt.title('True: '+label_map[Y[j]]+'  '+'Prediction: '+label_map[Ypred[j]])
        plt.show()
    prompt = input('Quit? Enter Y:\n')
    if prompt == 'Y':
        break
