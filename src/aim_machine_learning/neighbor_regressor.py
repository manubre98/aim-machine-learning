import numpy as np
from aim_machine_learning.base_regressor import Regressor
from scipy.spatial import distance_matrix
from sklearn.neighbors import KNeighborsRegressor

class NeighborRegressor(Regressor):
    def __init__(self, k = 1, **params):
        super().__init__(**params)
        self. k = k
        self.X = None
        self.y = None

    def fit(self, X, y):
        super().fit(X,y)
        self.X, self.y = X, y

    def predict(self, X):
        super().predict(X)
        y_pred = np.zeros((X.shape[0]))
        distmatr = distance_matrix(X, self.X)
        for i in range(y_pred.shape[0]):
            idx_k = np.argpartition(distmatr[i], self.k)[:self.k]
            y_pred[i] = np.mean(self.y[idx_k])
        return y_pred

class MySklearnNeighborRegressor(KNeighborsRegressor, Regressor):
    def __init__(self, n_neighbors):
        super().__init__(n_neighbors= n_neighbors)
