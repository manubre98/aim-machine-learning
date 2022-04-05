import numpy as np


class Evaluator():
    def __init__(self, supported_metrics):
        self.supported_metrics = supported_metrics
        self.metric = None

    def set_metric(self, new_metric):
        if new_metric not in self.supported_metrics:
            raise NameError()
        else:
            self.metric = new_metric
        return self

    def __call__(self, y_true, y_pred, *args, **kwargs):
        if self.metric == "mse":
            error = np.power(y_pred - y_true, 2)
            std = error.std()
            error = np.mean(error)
            return {"mean": np.round(error,2), "std": np.round(std,2)}
        if self.metric == "mae":
            error = np.abs(y_pred - y_true)
            std = error.std()
            error = np.mean(error)
            return {"mean": np.round(error, 2), "std": np.round(std, 2)}
        if self.metric == "corr":
            corr = np.corrcoef(y_true, y_pred)[0, 1]
            return {'corr': np.round(corr, 2)}


    def __str__(self):
        return f"Current metric is {self.metric}"
