import numpy as np
import matplotlib.pyplot as plt
from aim_machine_learning.model_evaluator import ModelEvaluator
from itertools import product

class ParametersTuner():
    def __init__(self, model_class, X, y, supported_eval_types, output_path = None):
        self.supported_eval_types = supported_eval_types
        self.X = X
        self.y = y
        self.output_path = output_path
        self.model_class = model_class


    def tune_parameters(self, params, eval_type, eval_obj, fig_name=None, **kwargs):
        if eval_type not in self.supported_eval_types:
            raise NameError(f"eval_type {eval_type} non supoortato.")

        keys = list(params.keys())

        if len(keys) == 1:
            list_params = params[keys[0]]
        else:
            list_params = params[keys[0]]
            for key in keys[1:]:
                list_params = list(product(list_params, params[key]))

        list_dicts = [0] * len(list_params)

        if len(keys) == 1:
            for i, params in enumerate(list_params):
                var_dict = dict()
                var_dict[keys[0]] = params
                list_dicts[i] = var_dict
        else:
            for i,params in enumerate(list_params):
                var_dict = dict()
                for j in range(len(params)):
                    var_dict[keys[j]] = params[j]
                list_dicts[i] = var_dict

        scores = np.zeros(len(list_dicts))

        if eval_type == "ttsplit":
            for i,param in enumerate(list_dicts):
                full_eval = ModelEvaluator(model_class=self.model_class, params=param, X=self.X, y=self.y)
                score = full_eval.train_test_split_eval(eval_obj = eval_obj, test_proportion = kwargs["test_proportion"])
                scores[i] = score["mean"] + score["std"]

        if eval_type == "kfold":
            for i,param in enumerate(list_dicts):
                full_eval = ModelEvaluator(model_class=self.model_class, params=param, X=self.X, y=self.y)
                score = full_eval.kfold_cv_eval(eval_obj = eval_obj, K = kwargs["K"])
                scores[i] = score["mean"] + score["std"]

        if self.output_path is not None and fig_name is not None:
            plt.figure()
            plt.plot(list_params, scores)
            plt.title('Error Plot')
            plt.xlabel('Parameter')
            plt.ylabel('Error')
            plt.savefig(self.output_path + fig_name)

        return list_dicts[np.argmin(scores)]






