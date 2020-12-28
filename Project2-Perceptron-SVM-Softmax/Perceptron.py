import numpy as np
import scipy
from tqdm import tqdm

class Perceptron():
    def __init__(self, x):
        """
        Initialises Perceptron classifier with initializing 
        weights, alpha(learning rate) and number of epochs.
        """
        self.W = np.random.rand(x+1,10)
        self.alpha = 0.0000001
        self.epochs = 100
    def get_acc(self, pred, y_test):
        return np.sum(y_test==pred)/len(y_test)*100         
    def train(self, X_train, y_train, X_test, y_test):
        """
        Train the Perceptron classifier. Use the perceptron update rule
        as introduced in Lecture 3.

        Inputs:
        - X_train: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y_train: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        """
        B = np.arange(self.epochs)

        for s in tqdm(range(self.epochs)):
            l=(1-s/100)
            for i in range(y_train.shape[0]):
                M =self.W.T.dot(np.insert(X_train[i], 0, 1))
                if (np.argmax(M) != y_train[i]):
                    
                    M = M-(np.ones(10)*M[y_train[i]])
                    M = np.where(M > 0, 1, 0)
                    K = np.sum(M)
                    M = M.reshape((10,1))
                    
                    self.W = (self.W.T - (self.alpha*l) * M.dot(np.insert(X_train[i], 0, 1).reshape((1,3073)))).T
                    self.W.T[y_train[i]] = self.W.T[y_train[i]] + K*(self.alpha*l) * np.insert(X_train[i], 0, 1)
                              
            B[s] = self.get_acc(self.predict(X_test),y_test)
            
        return B

    def predict(self, X_test):
        """
        Predict labels for test data using the trained weights.

        Inputs:
        - X_test: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.

        Returns:
        - pred: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].  
        """
        
        X_test = np.insert(X_test, 0, 1, axis=1)
        Y_out = self.W.T.dot(X_test.T)
        pred = np.argmax(Y_out, axis=0)
        
        return pred 