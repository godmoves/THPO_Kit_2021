import json
import math
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import torch


class Plot(object):
    def __init__(self):
        data_path = "input/data-30"
        ds_json = json.load(open(data_path, "r"))
        da = xr.DataArray.from_dict(ds_json)
        dims = da.dims

        x = da.attrs[dims[0]]["coords"]
        y = da.attrs[dims[1]]["coords"]
        X, Y = np.meshgrid(x, y)

        r, c = X.shape
        Z = np.zeros((r, c))
        for i in range(r):
            for j in range(c):
                p = {dims[0]: X[i, j], dims[1]: Y[i, j]}
                # Log value to make the diference clear
                Z[i, j] = -math.log(-float(da.loc[p].values))

        self.dims = dims
        self.X = X
        self.Y = Y
        self.Z = Z

        index = np.argmax(Z)
        self.opt_x = x[index % len(x)]
        self.opt_y = y[index // len(x)]

    @staticmethod
    def cal_pareto_front(acq_val):
        acq_val = acq_val.copy()
        acq_val[np.isnan(acq_val)] = -np.inf
        is_pf = np.ones(acq_val.shape[0], dtype=bool)
        for i, v in enumerate(acq_val):
            if is_pf[i]:
                is_pf[is_pf] = np.any(acq_val[is_pf] > v, axis=1)
                is_pf[i] = True
        return is_pf

    def show(self,
             suggestion_history=None,
             x_next=None,
             trust_region=None,
             acq=None,
             rec=None,
             init_pop=None):
        # Calculate acq value
        r, c = self.X.shape
        Xc = []
        for i in range(r):
            for j in range(c):
                Xc.append([self.X[i, j], self.Y[i, j]])
        Xc = torch.FloatTensor(Xc)
        acq_val = -acq.eval(Xc, None).numpy()
        rec = rec.to_numpy()
        print("Show {} NSGA2 recommendations.".format(len(rec)))
        rec_x, rec_y = rec[:, 0], rec[:, 1]
        pf = self.cal_pareto_front(acq_val)
        pf_x, pf_y = self.X.ravel()[pf], self.Y.ravel()[pf]
        init_x, init_y = init_pop[:, 0], init_pop[:, 1]

        x_all = [list(s[0].values()) for s in suggestion_history]
        y_all = [s[1] for s in suggestion_history]
        x_best = x_all[np.argmax(y_all)]
        y_best = max(y_all)
        x_next = [list(x.values()) for x in x_next]
        points = {"red": x_next, "blue": x_all}

        # Plot all figures
        f, axs = plt.subplots(2, 2, figsize=(9, 9))
        ax1, ax2, ax3, ax4 = axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]

        # EI
        ei = acq_val[:, 0].reshape(r, c)
        ax2.contourf(self.X, self.Y, ei, cmap="Blues")
        ax2.set_title("EI")
        ax2.scatter(init_x, init_y, c="grey", s=10, alpha=0.5)
        ax2.scatter(pf_x, pf_y, c="yellow", s=10)
        ax2.scatter(rec_x, rec_y, c="red", s=10)
        print("EI, min {} max {} mean {}".format(ei.min(), ei.max(), ei.mean()))

        # PI
        pi_sign = np.sign(acq_val[:, 1])
        pi_log = np.log(np.abs(acq_val[:, 1]))
        pi = (pi_sign * pi_log).reshape(r, c)
        ax3.contourf(self.X, self.Y, pi, cmap="Blues")
        ax3.set_title("PI")
        ax3.scatter(init_x, init_y, c="grey", s=10, alpha=0.5)
        ax3.scatter(pf_x, pf_y, c="yellow", s=10)
        ax3.scatter(rec_x, rec_y, c="red", s=10)
        print("PI, min {} max {} mean {}".format(np.nanmin(pi), np.nanmax(pi), np.nanmean(pi)))

        # UCB
        ucb_sign = np.sign(acq_val[:, 2])
        ucb_log = np.log(np.abs(acq_val[:, 2]))
        ucb = (ucb_sign * ucb_log).reshape(r, c)
        ax4.contourf(self.X, self.Y, ucb, cmap="Blues")
        ax4.set_title("UCB")
        ax4.scatter(init_x, init_y, c="grey", s=10, alpha=0.5)
        ax4.scatter(pf_x, pf_y, c="yellow", s=10)
        ax4.scatter(rec_x, rec_y, c="red", s=10)
        print("UCB, min {} max {} mean {}".format(np.nanmin(ucb), np.nanmax(ucb), np.nanmean(ucb)))

        ax1.contourf(self.X, self.Y, self.Z, cmap="Blues")

        # Current best point
        ax1.scatter([x_best[0]], [x_best[1]], marker="o", c="orange", s=100)
        # Optimal point
        ax1.scatter([self.opt_x], [self.opt_y], marker="*", c="orange", s=100)
        for color, ps in points.items():
            px = [p[0] for p in ps]
            py = [p[1] for p in ps]
            ax1.scatter(px, py, c=color, s=30)

        # Trust region
        if trust_region:
            lb, ub = trust_region
            x = np.array([lb[0], lb[0], ub[0], ub[0], lb[0]])
            y = np.array([lb[1], ub[1], ub[1], lb[1], lb[1]])
            ax1.plot(x, y, c="red")

        ax1.set_xlabel(self.dims[0])
        ax1.set_ylabel(self.dims[1])
        ax1.set_xlim(-0.2, 5.2)
        ax1.set_ylim(-0.2, 5.2)
        ax1.set_title("Log Z, current val: {:.3f}".format(y_best), fontsize=15)
        # ax1.colorbar()
        plt.tight_layout()
        # plt.show()
        plt.show(block=False)
        plt.pause(3)
        plt.close()


if __name__ == '__main__':
    plot = Plot()
    plot.show(points={"blue": [[1, 1], [2, 2], [3, 3], [4, 4]]})
