import numpy as np
from scipy.spatial import distance
from collections import Counter
from scipy import stats

class KNN():
    def __init__(self, k):
        """
        Initializes the KNN classifier with the k.
        """
        self.k = k
        self.X = None
        self.Y = None
        self.dist = None
        
    def train(self, X, y):
        """
        Train the classifier. For k-nearest neighbors this is just 
        memorizing the training data.

        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        """
        self.X = X
        self.Y = y
        
    def find_dist(self, X_test):
       
        self.dist = distance.cdist(X_test,self.X, 'braycurtis')
        
        return self.dist
    
    def predict(self, X_test, dist):
        
        #self.find_dist(X_test)       
        result = np.argsort(dist, axis=1)
        labels = np.zeros((result.shape[0],self.k))
        for j in range(result.shape[0]):
            for i in range(self.k):
                labels[j][i] = self.Y[result[j][i]]
                #np.append(M, self.Y[result[j][i]])
        #print(labels)
        final_labels = stats.mode(labels,axis=1)[0]
        final_labels = final_labels.reshape(final_labels.shape[0])
        """
        Predict labels for test data using the computed distances.

        Inputs:
        - X_test: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.

        Returns:
        - pred: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].  
        """
        return final_labels
