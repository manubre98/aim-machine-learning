import numpy as np
from aim_machine_learning.base_regressor import Regressor
from aim_machine_learning.metrics import Evaluator

class ModelEvaluator():
    def __init__(self, model_class, X, y, params):
        self.X = X
        self.y = y
        self.X_test, self.X_train = None, None
        self.y_test, self.y_train = None, None
        if issubclass(model_class, Regressor):
                self.model_class = model_class(**params)
        else:
            raise AttributeError("Deve essere un oggetto di tipo Regressor.")

    def train_test_split_eval(self, eval_obj, test_proportion):
        if test_proportion <= 1 and test_proportion >= 0:
            test_idx = int(test_proportion * self.X.shape[0])
        else:
            raise NameError("test_proportion deve essere tra 0 e 1.")

        self.X_test, self.X_train = self.X[:test_idx,:], self.X[test_idx:,:]
        self.y_test, self.y_train = self.y[:test_idx], self.y[test_idx:]
        if isinstance(eval_obj, Evaluator):
            if eval_obj.metric is None:
                raise NameError("Metric non definita per eval_obj.")
            self.model_class.fit(X = self.X_train,y = self.y_train)
            return self.model_class.evaluate(X = self.X_test, y = self.y_test, eval_obj = eval_obj)
        else:
            raise NameError("eval_obj deve essere un oggetto di tipo Evaluator.")

    def kfold_cv_eval(self, eval_obj, K):
        if not (isinstance(K,int) and K > 1):
            raise NameError("K deve essere intero maggiore di 1.")

        if isinstance(eval_obj, Evaluator):
            if eval_obj.metric is None:
                raise NameError("Metric non definita per eval_obj.")
        else:
            raise NameError("eval_obj deve essere un oggetto di tipo Evaluator.")


        step = int(np.floor(self.X.shape[0]/K))
        error = 0
        std = 0
        idx_bool = np.zeros(self.X.shape[0], bool)
        for i in range(K):
            idx_bool[i * step: (i+1) * step] = True

            self.X_test, self.X_train = self.X[idx_bool, :], self.X[~idx_bool, :]
            self.y_test, self.y_train = self.y[idx_bool], self.y[~idx_bool]

            idx_bool[i * step: (i + 1) * step] = False

            self.model_class.fit(X=self.X_train, y=self.y_train)
            score = self.model_class.evaluate(X=self.X_test, y=self.y_test, eval_obj=eval_obj)

            if eval_obj.metric is "corr":
                error += score["corr"]
            else:
                error += score["mean"]
                std += score["std"]

        if eval_obj.metric is "corr":
            return {"corr" : np.round(error/K,2)}
        else:
            return {"mean": np.round(error/K, 2), "std": np.round(std/K, 2)}