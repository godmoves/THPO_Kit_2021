# coding=utf-8
import copy

import numpy as np
import scipy.stats as ss
from turbo_1 import Turbo1
from utils import from_unit_cube, latin_hypercube, to_unit_cube


# Need to import the searcher abstract class, the following are essential
from thpo.abstract_searcher import AbstractSearcher


def order_stats(X):
    _, idx, cnt = np.unique(X, return_inverse=True, return_counts=True)
    obs = np.cumsum(cnt)  # Need to do it this way due to ties
    o_stats = obs[idx]
    return o_stats


def copula_standardize(X):
    X = np.nan_to_num(np.asarray(X))  # Replace inf by something large
    assert X.ndim == 1 and np.all(np.isfinite(X))
    o_stats = order_stats(X)
    quantile = np.true_divide(o_stats, len(X) + 1)
    X_ss = ss.norm.ppf(quantile)
    return X_ss


def standardize(X):
    X = (X - X.mean()) / X.std()
    return X


class Searcher(AbstractSearcher):
    searcher_name = "TurboSearcher"

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

        lower_bounds = [v["double_min_value"] for v in parameters_config.values()]
        upper_bounds = [v["double_max_value"] for v in parameters_config.values()]

        self.param_names = tuple([d["parameter_name"] for d in parameters_config.values()])
        self.dim = len(parameters_config)
        self.lb, self.ub = np.array(lower_bounds), np.array(upper_bounds)
        self.max_evals = np.iinfo(np.int32).max

        self.turbo = Turbo1(
            f=None,
            lb=self.lb,
            ub=self.ub,
            n_init=2 * self.dim + 1,
            max_evals=self.max_evals,
            batch_size=n_suggestion,
            verbose=False,
        )

        # Initialize turbo state
        self.batch_size = n_suggestion
        self.turbo.batch_size = n_suggestion
        self.turbo.failtol = np.ceil(np.max([10.0 / self.batch_size, self.dim / self.batch_size]))
        self.turbo.n_init = max(self.turbo.n_init, 5 * self.batch_size)
        self.restart()

    def restart(self):
        self.turbo._restart()
        self.turbo._X = np.zeros((0, self.turbo.dim))
        self.turbo._fX = np.zeros((0, 1))
        x_init = latin_hypercube(self.turbo.n_init, self.dim)
        self.x_init = from_unit_cube(x_init, self.lb, self.ub)

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
        if suggestion_history:
            new_obs = suggestion_history[-n_suggestions:]
            x = np.array([list(v[0].values()) for v in new_obs])
            # Turbo assume minimize problem, filp the sign
            y = np.array([-v[1] for v in new_obs])[:, None]

            if len(self.turbo._fX) >= self.turbo.n_init:
                self.turbo._adjust_length(y)

            self.turbo.n_evals += self.batch_size
            self.turbo._X = np.vstack((self.turbo._X, copy.deepcopy(x)))
            self.turbo._fX = np.vstack((self.turbo._fX, copy.deepcopy(y)))
            self.turbo.X = np.vstack((self.turbo.X, copy.deepcopy(x)))
            self.turbo.fX = np.vstack((self.turbo.fX, copy.deepcopy(y)))

            if self.turbo.length < self.turbo.length_min:
                print("We should restart turbo!")
                self.restart()

        x_next = np.zeros((n_suggestions, self.dim))

        n_init = min(len(self.x_init), n_suggestions)
        if n_init > 0:
            x_next[:n_init] = copy.deepcopy(self.x_init[:n_init])
            self.x_init = self.x_init[n_init:]

        n_adapt = n_suggestions - n_init
        if n_adapt > 0:
            if len(self.turbo._X) > 0:
                x = to_unit_cube(copy.deepcopy(self.turbo._X), self.lb, self.ub)
                # fx = copula_standardize(copy.deepcopy(self.turbo._fX).ravel())
                # fx = standardize(copy.deepcopy(self.turbo._fX).ravel())
                fx = copy.deepcopy(self.turbo._fX).ravel()
                x_cand, y_cand, _ = self.turbo._create_candidates(
                    x, fx, length=self.turbo.length, n_training_steps=100, hypers={}
                )
                x_next_turbo = self.turbo._select_candidates(x_cand, y_cand)[:n_adapt]
                x_next[-n_adapt:] = from_unit_cube(x_next_turbo, self.lb, self.ub)
            else:
                # Use some random points if no valid history available.
                x = np.random.random((n_adapt, self.dim))
                x_next[-n_adapt:] = from_unit_cube(x, self.lb, self.ub)

        next_suggestions = [dict(zip(self.param_names, x)) for x in x_next]

        return next_suggestions
