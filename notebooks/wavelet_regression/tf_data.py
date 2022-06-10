import numpy as np
import collections
import patsy
from gradvi.models import basis_matrix as basemat
from gradvi.tests import toy_data

CData = collections.namedtuple('CData', ['x', 'y', 'ytest', 'ytrue', 'btrue', 'bspline_bases', 'bsp_beta'])

def changepoint_from_bspline (x, knots, std,
                 degree = 0, signal = "gamma", seed = None,
                 include_intercept = False, bfix = None,
                 eps = 1e-8, get_bsplines = False):
    '''
    Generate trend-filtering data.

    Parameters
    ----------
    x: ndarray
        Data points

    knots: ndarray
        Location of changepoints on x

    std: float
        Standard deviation of the noise

    degree: integer, default 0
        Degree of the B-spline basis

    signal: str, default 'gamma'
        A distribution from which the coefficients are sampled
            - 'gamma' : Gamma(40, 0.1) with random signs (-1, 1)
            - 'normal' : Normal(0, 1)
            - 'fixed' : provide fixed values of the coefficients,
                        must be used with `bfix`.

    seed: integer
        Set a seed for reproducibility

    include_intercept: bool, default False
        Whether to include a basis function for intercept

    bfix: ndarray
        specify the values of the coefficients for the basis functions
          
    Note
    ----
    number of bases = k + degree + 1 if include_intercept = True
    number of bases = k + degree     if include_intercept = False

    '''
    if seed is not None: np.random.seed(seed)
    # ------------------------------
    n = x.shape[0]
    # ------------------------------
    # Generate B-spline bases given the knots and degree
    bspline_bases = patsy.bs(x, knots = knots, degree = degree, include_intercept = include_intercept)
    nbases = knots.shape[0] + degree + int(include_intercept)
    assert bspline_bases.shape[1] == nbases, "Number of B-spline bases does not match the number of knots + degree + interecept"
    # ------------------------------
    # Generate coefficients for the bases
    beta   = toy_data.sample_coefs(nbases, method = signal, bfix = bfix)
    # ------------------------------
    # Generate the function without noise 
    ytrue = np.dot(bspline_bases, beta)
    # ------------------------------
    # Map the data to trendfiltering bases
    # set low values of beta to zero and regenerate y
    H     = basemat.trendfiltering_scaled(n, degree)
    Hinv  = basemat.trendfiltering_inverse_scaled(n, degree)
    btrue = np.dot(Hinv, ytrue)
    btrue[np.abs(btrue) <= eps] = 0.
    noise = np.random.normal(0, std, size = n * 2)
    ytrue = np.dot(H, btrue)
    y     = ytrue + noise[:n]
    # ------------------------------
    # Some test data?
    ytest = ytrue + noise[n:]
    # ------------------------------
    data  = CData(x = x, y = y, ytest = ytest, ytrue = ytrue, btrue = btrue,
                bspline_bases = bspline_bases, bsp_beta = beta)
    # ------------------------------
    # Signal to noise ratio 
    # (experimental)
    #signal = np.mean(np.square(btrue[btrue != 0]))
    #snr    = signal / np.square(std)
    #data   = CData(x = x, y = y, ytest = ytest, ytrue = ytrue, btrue = btrue)
    #if get_bsplines:
    #    data = CData(x = x, y = y, ytest = ytest, ytrue = ytrue, btrue = btrue, 
    #        bspline_bases = bspline_bases, bsp_beta = beta)
    return data


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
    x = np.linspace(0, n-1, n)
    knots = hills_knots(n, kleft, kright)
    return changepoint_from_bspline(x, knots, std, degree = 3,
                                signal = "fixed", bfix = btrue, seed = seed)


def doppler(n, std = 0.1):
    #x = np.logspace(-8, 0, n)
    x = np.linspace(0, 1, n+1)[1:]
    ytrue = np.sin( 4 / x) + 1.5
    noise = np.random.normal(0, std, size = n * 2)
    y     = ytrue + noise[:n]
    ytest = ytrue + noise[n:]
    Hinv  = basemat.trendfiltering_inverse_scaled(n, 3)
    btrue = np.dot(Hinv, ytrue)
    data  = CData(x = x, y = y, ytest = ytest, ytrue = ytrue, btrue = btrue,
                  bspline_bases = None, bsp_beta = None)
    return data
