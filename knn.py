import numpy as np
from scipy.spatial.distance import cdist

#Datasets for testing
from sklearn.datasets import load_breast_cancer
from sklearn import preprocessing

class KNN(object):
    ''' A K-Nearest Neighbors Classifier Algorithm. '''
    def __init__(self):
        self.data = None
        self.k = None

    def fit(self, data, k=3):
        ''' Fits a K-NN model. 
        Inputs
        ======
        data: a 2D numpy array that includes the target variables in the last column
        k (optional, defaults to 3): number of nearest neighbors to consider for predictions.
        '''
        self.data = data
        self.k = k
    
    def predict(self, X_test):
        ''' Generate predictions on new data. Requires calling 'fit' first.
        Inputs
        ======
        X_test: A 2-D numpy array which does not include targets in the final column.
        Output
        ======
        Prediction: A 1-D numpy array of predictions for each row of the input data.
        '''
        dist = cdist(X_test, self.data[:, :-1])
        dist_idx = np.argpartition(dist, self.k)[:, :self.k] 
        y = self.data[:, -1]
        return np.around(np.mean(y[dist_idx], axis=1))

    def predict_proba(self, X_test):
        ''' Generate predicted probabilities on new data. Requires calling 'fit' first.
        Inputs
        ======
        X_test: A 2-D numpy array which does not include targets in the final column.
        Output
        ======
        Predicted Probability: A 1-D numpy array of predicted probabilities for each row of the input data.
        '''
        dist = cdist(X_test, self.data[:, :-1])
        dist_idx = np.argpartition(dist, self.k)[:, :self.k] 
        y = self.data[:, -1]
        return np.mean(y[dist_idx], axis=1)
    
    def score(self, prediction):
        ''' Mean Squared Error of the model
        Inputs
        ======
        prediction: A 1-D numpy array of predictions
        Output
        ======
        score: A float with the Mean Squared Error of the model.
        '''
        y = self.data[:, -1]
        return np.sum((prediction - y)**2)/y.shape[0]
        
         
if __name__ == '__main__':
    data = load_breast_cancer(return_X_y=True) 
    X = np.array(data[0])
    X = preprocessing.scale(X)
    X = np.hstack((X, np.expand_dims(data[1],1)))
    knn = KNN()
    knn.fit(X)

