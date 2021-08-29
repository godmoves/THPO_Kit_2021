import json
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt


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
                Z[i, j] = float(da.loc[p].values)

        self.dims = dims
        self.X = X
        self.Y = Y
        self.Z = Z

        index = np.argmax(Z)
        self.opt_x = x[index % len(x)]
        self.opt_y = y[index // len(x)]

    def show(self, suggestion_history=None, x_next=None, trust_region=None):
        plt.figure(figsize=(14, 12))
        plt.contourf(self.X, self.Y, self.Z)

        x_all = [list(s[0].values()) for s in suggestion_history]
        y_all = [s[1] for s in suggestion_history]
        x_best = x_all[np.argmax(y_all)]
        y_best = max(y_all)
        x_next = [list(x.values()) for x in x_next]
        points = {"red": x_next, "blue": x_all}

        # Current best point
        plt.scatter([x_best[0]], [x_best[1]], marker="o", c="orange", s=200)
        # Optimal point
        plt.scatter([self.opt_x], [self.opt_y], marker="*", c="purple", s=300)
        if points:
            for color, ps in points.items():
                px = [p[0] for p in ps]
                py = [p[1] for p in ps]
                plt.scatter(px, py, c=color)

        if trust_region:
            lb, ub = trust_region
            x = np.array([lb[0], lb[0], ub[0], ub[0], lb[0]])
            y = np.array([lb[1], ub[1], ub[1], lb[1], lb[1]])
            # x = (np.max(self.X) - np.min(self.X)) * x + np.min(self.X)
            # y = (np.max(self.Y) - np.min(self.Y)) * y + np.min(self.Y)
            plt.plot(x, y, c="red")

        plt.xlabel(self.dims[0])
        plt.ylabel(self.dims[1])
        plt.xlim(-0.2, 5.2)
        plt.ylim(-0.2, 5.2)
        plt.title("best val: {}".format(y_best), fontsize=20)
        plt.colorbar()
        plt.show()
        # plt.show(block=False)
        # plt.pause(3)
        # plt.close()


if __name__ == '__main__':
    plot = Plot()
    plot.show(points={"blue": [[1, 1], [2, 2], [3, 3], [4, 4]]})
