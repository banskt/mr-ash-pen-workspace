import numpy as np
import collections
import patsy
from gradvi.models import basis_matrix as gvbm
from gradvi.tests import toy_data
from gradvi.tests.toy_data import ChangepointData

def hills_knots(n, kleft, kright, degree = 3):
    nsep = 2. * n / 3.
    left_knots  = np.linspace(degree, nsep, kleft + 2)[1:-1]
    right_knots = np.linspace(nsep, n - 1, kright + 1)[:-1]
    knots = np.concatenate((left_knots, right_knots))
    knots = np.array([round(x, 0) for x in knots])
    return knots


def hills_tibshirani(n = 128, kleft = 2, kright = 4, std = 0.3,
                     btrue = np.array([1.3, 4.5, -1.9, 10.4, 0.6, 6.7, 0.6, 6.7, 2.6]),
                     seed = 100):
    x     = np.linspace(0, n-1, n)
    knots = hills_knots(n, kleft, kright)
    data  = toy_data.changepoint_from_bspline(x, knots, std,
                degree = 3, signal = "fixed", bfix = btrue, seed = seed)
    return data


def doppler(n, std = 0.1):
    #x = np.logspace(-8, 0, n)
    x = np.linspace(0, 1, n+1)[1:]
    ytrue = np.sin( 4 / x) + 1.5
    noise = np.random.normal(0, std, size = n * 2)
    y     = ytrue + noise[:n]
    ytest = ytrue + noise[n:]
    H     = gvbm.trendfiltering_scaled(n, 3)
    Hinv  = gvbm.trendfiltering_inverse_scaled(n, 3)
    btrue = np.dot(Hinv, ytrue)
    snr   = np.mean(np.square(btrue[btrue != 0])) / np.square(std)
    data  = ChangepointData(H = H, Hinv = Hinv, x = x, y = y, ytest = ytest, ytrue = ytrue, btrue = btrue,
                bspline_bases = None, bspline_beta = None, snr = snr, degree = 3)
    return data
