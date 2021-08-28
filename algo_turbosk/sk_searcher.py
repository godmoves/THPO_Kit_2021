# coding=utf-8
import numpy as np
from skopt import Optimizer as SkOpt
from skopt.space import Real

# Need to import the searcher abstract class, the following are essential
from thpo.abstract_searcher import AbstractSearcher


class Searcher(AbstractSearcher):
    searcher_name = "SkoptSearcher"

    def __init__(self, parameters_config, n_iter, n_suggestion):
        """ Init searcher

        Args:
            parameters_config: parameters configuration, consistent with the definition of parameters_config of EvaluateFunction. dict type:
                    dict key: parameters name, string type
                    dict value: parameters configuration, dict type:
                        "parameter_name": parameter name
                        "parameter_type": parameter type, 1 for double type, and only double type is valid
                        "double_max_value": max value of this parameter
                        "double_min_value": min value of this parameter
                        "double_step": step size
                        "coords": list type, all valid values of this parameter.
                            If the parameter value is not in coords,
                            the closest valid value will be used by the judge program.

                    parameter configuration example, eg:
                    {
                        "p1": {
                            "parameter_name": "p1",
                            "parameter_type": 1
                            "double_max_value": 2.5,
                            "double_min_value": 0.0,
                            "double_step": 1.0,
                            "coords": [0.0, 1.0, 2.0, 2.5]
                        },
                        "p2": {
                            "parameter_name": "p2",
                            "parameter_type": 1,
                            "double_max_value": 2.0,
                            "double_min_value": 0.0,
                            "double_step": 1.0,
                            "coords": [0.0, 1.0, 2.0]
                        }
                    }
                    In this example, "2.5" is the upper bound of parameter "p1", and it's also a valid value.

        n_iteration: number of iterations
        n_suggestion: number of suggestions to return
        """
        AbstractSearcher.__init__(self, parameters_config, n_iter, n_suggestion)

        dims = self.get_sk_dims(parameters_config)

        self.dim_names = tuple([d.name for d in dims])
        self.skopt = SkOpt(
            dims,
            n_initial_points=n_suggestion * 4,  # batch size
            base_estimator="GBRT",  # GP, RF, ET, GBRT
            acq_func="EI",  # LCB, EI, PI, gp_hedge, EIps, PIps
            acq_optimizer="sampling",
            acq_func_kwargs={},
            acq_optimizer_kwargs={},
        )

    @staticmethod
    def get_sk_dims(parameters_config, transform="normalize"):
        param_list = sorted(parameters_config.keys())

        sk_dims = []
        for param_name in param_list:
            param_config = parameters_config[param_name]
            # Ensure parammeter type is double.
            assert param_config["parameter_type"] == 1
            v_min, v_max = param_config["double_min_value"], param_config["double_max_value"]
            sk_dims.append(Real(v_min, v_max, prior="uniform", transform=transform, name=param_name))
        return sk_dims

    def suggest(self, suggestion_history, n_suggestions=1):
        """ Suggest next n_suggestion parameters.

        Args:
            suggestions_history: a list of historical suggestion parameters and rewards, in the form of
                    [[Parameter, Reward], [Parameter, Reward] ... ]
                        Parameter: a dict in the form of {name:value, name:value, ...}. for example:
                            {'p1': 0, 'p2': 0, 'p3': 0}
                        Reward: a float type value

                    The parameters and rewards of each iteration are placed in suggestions_history in the order of iteration.
                        len(suggestions_history) = n_suggestion * iteration(current number of iteration)

                    For example:
                        when iteration = 2, n_suggestion = 2, then
                        [[{'p1': 0, 'p2': 0, 'p3': 0}, -222.90621774147272],
                         [{'p1': 0, 'p2': 1, 'p3': 3}, -65.26678723205647],
                         [{'p1': 2, 'p2': 2, 'p3': 2}, 0.0],
                         [{'p1': 0, 'p2': 0, 'p3': 4}, -105.8151893979122]]

            n_suggestion: int, number of suggestions to return

        Returns:
            next_suggestions: list of Parameter, in the form of
                    [Parameter, Parameter, Parameter ...]
                        Parameter: a dict in the form of {name:value, name:value, ...}. for example:
                            {'p1': 0, 'p2': 0, 'p3': 0}

                    For example:
                        when n_suggestion = 3, then
                        [{'p1': 0, 'p2': 0, 'p3': 0},
                         {'p1': 0, 'p2': 1, 'p3': 3},
                         {'p1': 2, 'p2': 2, 'p3': 2}]
        """
        # Tell the skopt what we find
        if suggestion_history:
            ys = [s[1] for s in suggestion_history]
            print("hist len: {}, best val: {}".format(len(ys), max(ys)))

            new_value = suggestion_history[-n_suggestions:]
            for x, y in new_value:
                x = [x[name] for name in self.dim_names]
                if np.isfinite(y):
                    # Skopt try to minimize value, so we filp the sign.
                    self.skopt.tell(x, -y)

        # Suggest the next points to evaluate.
        next_suggestions = self.skopt.ask(n_points=n_suggestions)
        next_suggestions = [dict(zip(self.dim_names, x)) for x in next_suggestions]

        return next_suggestions
