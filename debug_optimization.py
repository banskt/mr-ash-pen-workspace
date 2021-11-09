import numpy as np
from mrashpen.inference.penalized_regression import PenalizedRegression as PLR

import sys
sys.path.append('/home/saikat/Documents/work/sparse-regression/simulation/eb-linreg-dsc/dsc/functions')
import simulate


def center_and_scale(Z):
    dim = Z.ndim
    if dim == 1:
        Znew = Z / np.std(Z)
        Znew = Znew - np.mean(Znew)
    elif dim == 2:
        Znew = Z / np.std(Z, axis = 0)
        Znew = Znew - np.mean(Znew, axis = 0).reshape(1, -1)
    return Znew


def initialize_ash_prior(k, scale = 2):
    w = np.zeros(k)
    w[0] = 1e-8
    w[1:(k-1)] = np.repeat((1 - w[0])/(k-1), (k - 2))
    w[k-1] = 1 - np.sum(w)
    sk2 = np.square((np.power(scale, np.arange(k) / k) - 1))
    prior_grid = np.sqrt(sk2)
    return w, prior_grid


if __name__ == '__main__':

    n = 200
    p = 2000
    p_causal = 20
    pve = 0.7
    k = 20

    X, y, Xtest, ytest, btrue, strue = simulate.equicorr_predictors(n, p, p_causal, pve, rho = 0.5, seed = 20)
    X      = center_and_scale(X)
    Xtest  = center_and_scale(Xtest)
    wk, sk = initialize_ash_prior(k, scale = 2)

    ## Optimize
    plr_lbfgs = PLR(method = 'L-BFGS-B', optimize_w = True, optimize_s = True, is_prior_scaled = True,
                    debug = False, display_progress = True)
    plr_lbfgs.fit(X, y, sk, binit = None, winit = wk, s2init = 1)
