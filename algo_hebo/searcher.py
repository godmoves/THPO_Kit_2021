# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

# import time
# import math
from sklearn.preprocessing import power_transform

import numpy as np
import pandas as pd
import torch
from torch.quasirandom import SobolEngine
# from pyDOE2 import lhs
from bo.design_space.design_space import DesignSpace
from bo.models.model_factory import get_model
from bo.acquisitions.acq import Mean, Sigma, MACE
from bo.optimizers.evolution_optimizer import EvolutionOpt


# Need to import the searcher abstract class, the following are essential
from thpo.abstract_searcher import AbstractSearcher
from plot import Plot

torch.set_num_threads(min(1, torch.get_num_threads()))


class Searcher(AbstractSearcher):
    searcher_name = "HeboSearcher"

    def __init__(self, parameters_config, n_iter, n_suggestion, model_name='gpy'):
        AbstractSearcher.__init__(self, parameters_config, n_iter, n_suggestion)
        self.param_config = parameters_config
        self.param_names = tuple([p for p in parameters_config])
        self.space = self.parse_space(parameters_config)
        self.X = pd.DataFrame(columns=self.space.para_names)
        self.y = np.zeros((0, 1))
        self.model_name = model_name
        self.sobol = SobolEngine(self.space.num_paras, scramble=False)
        self.plot = Plot()
        self.n_init = 5 * n_suggestion

    def filter(self, y: torch.Tensor) -> [bool]:
        if not (np.all(y.numpy() > 0) and (y.max() / y.min() > 20)):
            return [True for _ in range(y.shape[0])], np.inf
        else:
            data = y.numpy().reshape(-1)
            quant = min(data.min() * 20, np.quantile(data, 0.95, interpolation='lower'))
            return (data <= quant).tolist(), quant

    def quasi_sample(self, n):
        samp = self.sobol.draw(n)
        # samp = torch.FloatTensor(lhs(self.space.num_paras, n))
        samp = samp * (self.space.opt_ub - self.space.opt_lb) + self.space.opt_lb
        x = samp[:, :self.space.num_numeric]
        xe = samp[:, self.space.num_numeric:]
        df_samp = self.space.inverse_transform(x, xe)
        df_samp = self.round_to_coords(df_samp)
        return df_samp

    def parse_space(self, api_config):
        space = DesignSpace()
        params = []
        for param_name in api_config:
            bo_param_conf = {'name': param_name}
            param_type = api_config[param_name]["parameter_type"]
            if param_type == 1:
                bo_param_conf['type'] = 'num'
                bo_param_conf['lb'] = api_config[param_name]["double_min_value"]
                bo_param_conf['ub'] = api_config[param_name]["double_max_value"]
            else:
                assert False, "type %s not handled in API" % param_type
            params.append(bo_param_conf)
        # print(params)
        space.parse(params)
        return space

    @property
    def model_config(self):
        if self.model_name == 'gp':
            cfg = {
                'lr': 0.01,
                'num_epochs': 100,
                'verbose': True,
                'noise_lb': 8e-4,
                'pred_likeli': False
            }
        elif self.model_name == 'gpy':
            cfg = {
                'verbose': False,
                'warp': True,
                'space': self.space
            }
        elif self.model_name == 'gpy_mlp':
            cfg = {
                'verbose': False
            }
        elif self.model_name == 'rf':
            cfg = {
                'n_estimators': 20
            }
        else:
            cfg = {}
        if self.space.num_categorical > 0:
            cfg['num_uniqs'] = [len(self.space.paras[name].categories) for name in self.space.enum_names]
        return cfg

    def round_to_coords(self, x: pd.DataFrame):
        x = x.copy()
        n, d = x.shape
        for i, v in enumerate(self.param_config.values()):
            coords = np.array(v["coords"]).reshape(1, -1)
            index = np.abs(x.iloc[:, [i]].to_numpy() - coords).argmin(axis=1)
            x.iloc[:, i] = np.array([coords[0, idx] for idx in index])
        # print(x)
        return x

    def normalize(self, x: np.ndarray):
        ub = self.space.opt_ub.numpy().squeeze()
        lb = self.space.opt_lb.numpy().squeeze()
        return (x - lb) / (ub - lb)

    def cal_rec_dist2(self, rec: pd.DataFrame, best_x: pd.DataFrame):
        rec = rec.to_numpy()
        best_x = best_x.to_numpy()[0]

        rec = self.normalize(rec)
        best_x = self.normalize(best_x)
        dist2 = (rec - best_x) ** 2 / self.space.num_paras
        dist2 = dist2.sum(axis=1)
        dist2 = dist2 / dist2.sum()
        return dist2

    def suggest(self, suggestion_history, n_suggestions=1):
        if suggestion_history:
            new_obs = suggestion_history[-n_suggestions:]
            X = [s[0] for s in new_obs]
            # HEBO try to minimize, so we flip the sign
            y = [-s[1] for s in new_obs]
            self.observe(X, y)

        if self.X.shape[0] < self.n_init:
            df_suggest = self.quasi_sample(n_suggestions)
            x_guess = df_suggest.to_dict("records")
        else:
            X, Xe = self.space.transform(self.X)
            try:
                if self.y.min() <= 0:
                    y = torch.FloatTensor(power_transform(
                        self.y / self.y.std(), method='yeo-johnson'))
                else:
                    y = torch.FloatTensor(power_transform(
                        self.y / self.y.std(), method='box-cox'))
                    if y.std() < 0.5:
                        y = torch.FloatTensor(power_transform(
                            self.y / self.y.std(), method='yeo-johnson'))
                if y.std() < 0.5:
                    raise RuntimeError('Power transformation failed')
                model = get_model(self.model_name, self.space.num_numeric,
                                  self.space.num_categorical, 1, **self.model_config)
                model.fit(X, Xe, y)
            except Exception:
                print('Error fitting GP')
                y = torch.FloatTensor(self.y).clone()
                filt, q = self.filter(y)
                print('Q = %g, kept = %d/%d' % (q, y.shape[0], self.y.shape[0]))
                X = X[filt]
                Xe = Xe[filt]
                y = y[filt]
                model = get_model(self.model_name, self.space.num_numeric,
                                  self.space.num_categorical, 1, **self.model_config)
                model.fit(X, Xe, y)
            print('Noise level: %g' % model.noise, flush=True)

            best_id = np.argmin(self.y.squeeze())
            best_x = self.X.iloc[[best_id]]
            best_y = y.min()
            py_best, ps2_best = model.predict(*self.space.transform(best_x))
            py_best = py_best.detach().numpy().squeeze()
            ps_best = ps2_best.sqrt().detach().numpy().squeeze()

            # XXX: minimize (mu, -1 * sigma)
            #      s.t.     LCB < best_y
            iters = max(1, self.X.shape[0] // n_suggestions)
            upsi = 0.5
            delta = 0.01
            # kappa = np.sqrt(upsi * 2 * np.log(iter **  (2.0 + self.X.shape[1] / 2.0) * 3 * np.pi**2 / (3 * delta)))
            kappa = np.sqrt(upsi * 2 * ((2.0 + self.X.shape[1] / 2.0) * np.log(iters) + np.log(3 * np.pi**2 / (3 * delta))))

            acq = MACE(model, py_best, kappa=kappa)  # LCB < py_best
            mu = Mean(model)
            sig = Sigma(model, linear_a=-1.)
            opt = EvolutionOpt(self.space, acq, pop=100, iters=100, verbose=True)
            rec = opt.optimize(initial_suggest=best_x, trust_region=None).drop_duplicates()
            rec = self.round_to_coords(rec)
            rec = rec[self.check_unique(rec)]

            cnt = 0
            while rec.shape[0] < n_suggestions:
                rand_rec = self.quasi_sample(n_suggestions - rec.shape[0])
                rand_rec = rand_rec[self.check_unique(rand_rec)]
                rec = rec.append(rand_rec, ignore_index=True)
                cnt += 1
                if cnt > 3:
                    break
            if rec.shape[0] < n_suggestions:
                rand_rec = self.quasi_sample(n_suggestions - rec.shape[0])
                rec = rec.append(rand_rec, ignore_index=True)

            dist2 = self.cal_rec_dist2(rec, best_x)
            # print("Rec dist2", dist2)
            # Sample according to the distance form the current best points.
            select_id = np.random.choice(rec.shape[0], n_suggestions, replace=False, p=dist2).tolist()
            x_guess = []
            with torch.no_grad():
                py_all = mu(*self.space.transform(rec)).squeeze().numpy()
                ps_all = -1 * sig(*self.space.transform(rec)).squeeze().numpy()
                best_pred_id = np.argmin(py_all)
                best_unce_id = np.argmax(ps_all)
                if best_unce_id not in select_id and n_suggestions > 2:
                    select_id[0] = best_unce_id
                if best_pred_id not in select_id and n_suggestions > 2:
                    select_id[1] = best_pred_id
                rec_selected = rec.iloc[select_id].copy()
                py, ps2 = model.predict(*self.space.transform(rec_selected))
                rec_selected['py'] = py.squeeze().numpy()
                rec_selected['ps'] = ps2.sqrt().squeeze().numpy()
                print(rec_selected)
            print('Best y is %g %g %g %g' % (self.y.min(), best_y, py_best, ps_best), flush=True)
            for idx in select_id:
                x_guess.append(rec.iloc[idx].to_dict())

        if len(suggestion_history) >= self.n_init:
            self.plot.show(
                suggestion_history=suggestion_history,
                x_next=x_guess,
                acq=acq,
                rec=rec,
                # init_pop=opt.init_pop,
            )

        return x_guess

    def observe(self, X, y):
        y = np.array(y).reshape(-1)
        valid_id = np.where(np.isfinite(y))[0].tolist()
        XX = [X[idx] for idx in valid_id]
        yy = y[valid_id].reshape(-1, 1)
        self.X = self.X.append(XX, ignore_index=True)
        self.y = np.vstack([self.y, yy])
        print("Get -YY {}, best Y: {}".format(yy, -min(self.y.squeeze())))

    def check_unique(self, rec: pd.DataFrame) -> [bool]:
        return (~pd.concat([self.X, rec], axis=0).duplicated().tail(rec.shape[0]).values).tolist()
