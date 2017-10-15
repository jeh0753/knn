import numpy as np
from scipy.spatial.distance import cdist

#Datasets for testing
from sklearn.datasets import load_boston
from sklearn import preprocessing

class KNN_Imputer(object):
    ''' A K-Nearest Neighbors Matrix Imputer. '''
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

    
    def predict(self):
        ''' Generate predictions on new data. Requires calling 'fit' first.
        Inputs
        ======
        None
        Output
        ======
        Prediction: A 1-D numpy array of predictions for each row of the input data.
        '''
        dist = cdist(X_test, self.data[:, :-1])
        dist_idx = np.argpartition(dist, self.k)[:, :self.k] 
        y = self.data[:, -1]
        return np.around(np.mean(y[dist_idx], axis=1))
    
         
if __name__ == '__main__':
    knn = KNN()
    #knn.fit(X)

    rng = np.random.RandomState(0)

    dataset = load_boston()
    X_full, y_full = dataset.data, dataset.target
    n_samples = X_full.shape[0]
    n_features = X_full.shape[1] 
    missing_rate = 0.25
    n_missing_samples = int(np.floor(n_samples * missing_rate))
    missing_samples = np.hstack((np.zeros(n_samples - n_missing_samples,
                                          dtype=np.bool),
                                 np.ones(n_missing_samples,
                                         dtype=np.bool)))
    rng.shuffle(missing_samples)
    missing_features = rng.randint(0, n_features, n_missing_samples)

    # Estimate the score without the lines containing missing values
    X_filtered = X_full[~missing_samples, :]
    #X = preprocessing.scale(X)
