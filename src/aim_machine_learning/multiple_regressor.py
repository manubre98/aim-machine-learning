from aim_machine_learning.base_regressor import Regressor
import numpy as np

class MultipleRegressor(Regressor):
    def __init__(self, a, b, **params):
        super().__init__(**params)
        self.a = a
        self.b = b

    def fit(self, X, y):
        pass

    def predict(self, X):
        return (self.b + np.dot(X, self.a)).squeeze()

    def __add__(self, otherRegressor):
        return MultipleRegressor(a = [[self.a], [otherRegressor.a]], b = self.b + otherRegressor.b)

