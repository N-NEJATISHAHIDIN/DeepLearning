import numpy as np
import scipy
from tqdm import tqdm
class Softmax():
    def __init__(self, x):
        """
        Initialises Softmax classifier with initializing 
        weights, alpha(learning rate), number of epochs
        and regularization constant.
        """
        self.W = np.zeros((x+1,10))
        self.alpha = 0.00000001
        self.epochs = 100
        self.reg_const = 0.01
    
    def calc_gradient(self, W, X, y, reg):
        """
        Calculate gradient of the softmax loss
          
        Inputs have dimension D, there are C classes, and we operate on minibatches
        of N examples.

        Inputs:
        - X_train: A numpy array of shape (N, D) containing a minibatch of data.
        - y_train: A numpy array of shape (N,) containing training labels; y[i] = c means
          that X[i] has label c, where 0 <= c < C.

        Returns:
        - gradient with respect to weights W; an array of same shape as W
        """

        N = X.shape[0]
        grad_W = np.zeros_like(W)
        I = np.ones((1,10))
        score = np.dot(X, W)   # (N, C)
        out = np.exp(score-np.dot(np.max(score, axis=1, keepdims=True ),I))
        #print("out", out)
        out /= np.sum(out, axis=1, keepdims=True)   # (N, C)
       
        dout = np.copy(out)   # (N, C)
        dout[np.arange(N), y] -= 1
        grad_W = np.dot(X.T, dout)  # (D, C)
        grad_W /= N
        #grad_W += reg * W
        
        return grad_W
    
    def get_acc(self, pred, y_test):
        return np.sum(y_test==pred)/len(y_test)*100 
    
    def train(self, X_train, y_train,X_test,y_test):
        """
        Train Softmax classifier using stochastic gradient descent.

        Inputs:
        - X_train: A numpy array of shape (N, D) containing training data;
        N examples with D dimensions
        - y_train: A numpy array of shape (N,) containing training labels;
        
        Hint : Operate with Minibatches of the data for SGD
        """
        K = np.arange(self.epochs)
        F= y_train.shape[0]
        for s in tqdm(range(self.epochs)):
            for i in range( 0, F, 10):
                grad_W = self.calc_gradient(self.W, np.insert(X_train[i:i+10], 0, 1, axis=1), y_train[i:i+10], self.reg_const)
                self.W = self.W - self.alpha * grad_W
            K[s] = self.get_acc(self.predict(X_test),y_test)
        return K
    
    def predict(self, X_test):
        """
        Use the trained weights of softmax classifier to predict labels for
        data points.

        Inputs:
        - X_test: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.

        Returns:
        - pred: Predicted labels for the data in X_test. pred is a 1-dimensional
          array of length N, and each element is an integer giving the predicted
          class.
          
        """
        
        X_test = np.insert(X_test, 0, 1, axis=1)
        Y_out = self.W.T.dot(X_test.T)
        pred = np.argmax(Y_out, axis=0)
        
        
        return pred 