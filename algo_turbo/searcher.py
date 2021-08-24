# coding=utf-8
import math

import torch
from torch.quasirandom import SobolEngine
from botorch.utils.transforms import unnormalize
from botorch.utils.transforms import normalize
from botorch.fit import fit_gpytorch_model
from botorch.generation import MaxPosteriorSampling
from botorch.models import SingleTaskGP

from gpytorch.constraints import Interval
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood

# Need to import the searcher abstract class, the following are essential
from thpo.abstract_searcher import AbstractSearcher


class TurboState(object):
    def __init__(self,
                 dim: int,
                 batch_size: int,
                 length: float = 0.8,
                 length_min: float = 0.5 ** 7,
                 length_max: float = 1.6,
                 failure_counter: int = 0,
                 failure_tolerance: int = float("nan"),  # Note: post-initialized
                 success_counter: int = 0,
                 success_tolerance: int = 3,  # Note: 3 for original paper
                 best_value: float = -float("inf"),
                 restart_triggered: bool = False):
        self.dim = dim
        self.batch_size = batch_size
        self.length = length
        self.length_min = length_min
        self.length_max = length_max
        self.failure_counter = failure_counter
        self.failure_tolerance = failure_tolerance
        self.success_counter = success_counter
        self.success_tolerance = success_tolerance
        self.best_value = best_value
        self.restart_triggered = restart_triggered

        if math.isnan(self.failure_tolerance):
            self.post_init()

    def post_init(self):
        self.failure_tolerance = math.ceil(
            max(4.0, self.dim) / self.batch_size
        )

    def __str__(self):
        return "dim: {}, batch_size: {}, length: {}, best_value: {}, restart_triggered: {}".format(
            self.dim, self.batch_size, self.length, self.best_value, self.restart_triggered
        )

    def update_state(self, y_next):
        if max(y_next) > self.best_value + 1e-3 * math.fabs(self.best_value):
            self.success_counter += 1
            self.failure_counter = 0
        else:
            self.success_counter = 0
            self.failure_counter += 1

        if self.success_counter == self.success_tolerance:
            self.length = min(2.0 * self.length, self.length_max)
            self.success_counter = 0
        elif self.failure_counter == self.failure_tolerance:
            self.length /= 2.0
            self.failure_counter = 0

        self.best_value = max(self.best_value, max(y_next))
        if self.length < self.length_min:
            self.restart_triggered = True
        return self


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
        self.tkwargs = {
            "dtype": torch.double,
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        }
        lower_bounds = [v["double_min_value"] for v in parameters_config.values()]
        upper_bounds = [v["double_max_value"] for v in parameters_config.values()]
        self.parameters_config = parameters_config
        self.dim = len(parameters_config)
        self.bounds = torch.tensor([lower_bounds, upper_bounds], **self.tkwargs)
        self.state = TurboState(dim=self.dim, batch_size=n_suggestion)
        self.sobol = SobolEngine(self.dim, scramble=True)

        # Hyper-parameters
        self.n_candidates = min(5000, max(2000, 200 * self.dim))
        self.n_init = 20  # 4 batch * 5 points/batch

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
        if suggestion_history is None or len(suggestion_history) < self.n_init:
            x_next = self.sobol.draw(n_suggestions).to(**self.tkwargs)
            x_next = unnormalize(x_next, self.bounds).detach().numpy()
        else:
            # Process x and y values
            x = [[v for v in d[0].values()] for d in suggestion_history]
            y = [d[1] for d in suggestion_history]

            # Update turbo state
            self.state.update_state(y)

            # Convert to torch tensor
            x = torch.tensor(x, **self.tkwargs)
            y = torch.tensor(y, **self.tkwargs).unsqueeze(-1)

            # Standardize x, y
            x = normalize(x, self.bounds)
            y = (y - y.mean()) / y.std()
            assert x.min() >= 0 and x.max() <= 1.0 and torch.all(torch.isfinite(y))

            # Fit GP model
            likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-4))
            model = SingleTaskGP(x, y, likelihood=likelihood)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_model(mll)

            # Calculate trust region
            x_center = x[y.argmax(), :].clone()
            weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
            weights = weights / weights.mean()
            weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
            tr_lb = torch.clamp(x_center - weights * self.state.length / 2.0, 0.0, 1.0)
            tr_ub = torch.clamp(x_center + weights * self.state.length / 2.0, 0.0, 1.0)

            # Create perturbation mask
            sobol = SobolEngine(self.dim, scramble=True)
            pert = sobol.draw(self.n_candidates).to(**self.tkwargs)
            pert = tr_lb + (tr_ub - tr_lb) * pert
            prob_perturb = min(20.0 / self.dim, 1.0)
            mask = (
                torch.rand(self.n_candidates, self.dim, **self.tkwargs) <= prob_perturb
            )
            ind = torch.where(mask.sum(dim=1) == 0)[0]
            mask[ind, torch.randint(0, self.dim - 1, size=(len(ind),))] = 1

            # Create candidate points
            x_cand = x_center.expand(self.n_candidates, self.dim).clone()
            x_cand[mask] = pert[mask]

            # Sample on the candidate points
            thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
            x_next = thompson_sampling(x_cand, num_samples=n_suggestions)

            # Transform x back
            x_next = unnormalize(x_next, self.bounds).detach().numpy()

            print("x num: {}, best value: {}, TR len: {}".format(len(x), self.state.best_value, self.state.length))

        next_suggestions = []
        for n in range(n_suggestions):
            next_suggest = {}
            for i, k in enumerate(self.parameters_config):
                next_suggest[k] = x_next[n, i]
            next_suggestions.append(next_suggest)

        # for __ in range(n_suggestions):
        #     next_suggest = {
        #         p_name: p_conf["coords"][random.randint(0, len(p_conf["coords"]) - 1)]
        #         for p_name, p_conf in self.parameters_config.items()
        #     }
        #     next_suggestions.append(next_suggest)

        return next_suggestions
