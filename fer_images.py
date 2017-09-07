# This code exhibits, just for fun, some random images with the actual 
# output model 'state of mind' classification.
import numpy as np
import matplotlib.pyplot as plt
import ANN_relu_Multi_batch3_RMSprop
from sklearn.externals import joblib
from preprocess_data import getData




def main():
# this code loades the model 
    model = joblib.load('ANN_relu_batch3_RMSprop.sav')


# a mapping list for each face recognition type
    label_map = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


# load the previous samples to visualize some images
    npzfile = np.load('fer2test1k.npz')
    X = npzfile['Xtest']
    Y = npzfile['Ytest']


# you can also load all the original data and check if the accuracy 
# is still the same... 
    # X, Y = getData(Ns=None, balance_ones=False)



# get the prediction for this model
    Ypred = model.predict(X)


# computes the model accuracy
    print('\nModel accuracy:{:.3f}'.format(np.mean(Y==Ypred)))
    print('\n')



# randomly selects a image to display with the true class and the 
# respective model output
    while True:
        for i in range(7):
            # x, y = X[Y==i], Y[Y==i]
            N = len(Y)
            j = np.random.choice(N)
            plt.imshow(X[j].reshape(48, 48), cmap='gray')
            plt.title('True type: '+label_map[Y[j]]+'  |  '+'Model output: '+label_map[Ypred[j]])
            plt.show()
        prompt = input('Quit? Enter Y:\n')
        if prompt == 'Y': # if user decides to stop the display every 7 images 
            break




if __name__ == '__main__':
    main() 
