# coding=utf-8
import random

import pandas as pd
import numpy as np

from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO

# Need to import the searcher abstract class, the following are essential
from thpo.abstract_searcher import AbstractSearcher


class Searcher(AbstractSearcher):
    searcher_name = "RandomSearcher"

    def __init__(self, parameters_config, n_iter, n_suggestion):
        AbstractSearcher.__init__(self, parameters_config, n_iter, n_suggestion)
        space = self._build_hebo_space(parameters_config)
        self.param_names = tuple([p for p in parameters_config])
        self.opt = HEBO(space=space, rand_sample=5 * n_suggestion)

    def _build_hebo_space(self, parameters_config):
        space = DesignSpace()
        params = []
        for param_name in parameters_config:
            bo_param_conf = {'name': param_name}
            param_type = parameters_config[param_name]["parameter_type"]
            if param_type == 1:
                bo_param_conf['type'] = 'num'
                bo_param_conf['lb'] = parameters_config[param_name]["double_min_value"]
                bo_param_conf['ub'] = parameters_config[param_name]["double_max_value"]
            else:
                assert False, "type %s not handled in API" % param_type
            params.append(bo_param_conf)
        space.parse(params)
        return space

    def suggest(self, suggestion_history, n_suggestions=1):
        if suggestion_history:
            new_obs = suggestion_history[-n_suggestions:]
            x = [ob[0] for ob in new_obs]
            y = [-ob[1] for ob in new_obs]
            x = pd.DataFrame(x, columns=self.param_names)
            y = np.array(y).reshape(-1, 1)
            self.opt.observe(x, y)

        rec = self.opt.suggest(n_suggestions)
        next_suggestions = rec.to_dict("records")
        return next_suggestions
