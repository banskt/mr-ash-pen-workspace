import numpy as np
import collections

from gradvi.models import basis_matrix
from gradvi.inference import WaveletRegression
from gradvi.inference import LinearRegression
from gradvi.tests import toy_priors
from mrashpen.utils import R_trendfilter
from mrashpen.inference.mrash_wrapR import MrASHR
from mrashpen.utils import R_lasso

CRes = collections.namedtuple('CRes', ['y', 'coef', 's2', 'prior', 'obj'])


def factory(meth, y, degree, b0 = None, s0 = None, x0 = None):
    if meth == 'genlasso':
        res = genlasso_trendfilter(y, degree)
    elif meth == 'gradvi':
        res = gradvi_trendfilter(y, degree, b0 = b0, s0 = s0, x0 = x0)
    elif meth == 'cavi':
        res = cavi_trendfilter(y, degree, b0 = b0, s0 = s0, x0 = x0)
    elif meth == 'lasso':
        res = lasso_trendfilter(y, degree)
    elif meth == 'gradvi-wavelet':
        res = gradvi_wavelet_trendfilter(y, degree, x0 = x0, s0 = s0, b0 = b0)
    elif meth == 'gradvi-direct':
        res = gradvi_trendfilter_direct(y, degree, x0 = x0, s0 = s0, b0 = b0)
    return res


def lasso_trendfilter(y, degree):
    n = y.shape[0]
    H = basis_matrix.trendfiltering_scaled(n, degree)
    a0, b, _ = R_lasso.fit(H, y)
    x = np.dot(H, b) + a0
    s2 = np.var(y - x)
    return CRes(y = x, coef = b, s2 = s2, prior = None, obj = None)


def gradvi_trendfilter(y, degree, b0 = None, s0 = None, x0 = None):
    n    = y.shape[0]
    H    = basis_matrix.trendfiltering_scaled(n, degree)
    Hinv = basis_matrix.trendfiltering_inverse_scaled(n, degree)
    
    # Initialize
    if b0 is None:
        if x0 is None:
            print("GradVI: Running first pass for initialization")
            prior_init = toy_priors.get_ash_scaled(k = 10, sparsity = 0.9, skbase = (degree + 1) * 30)
            gv1 = LinearRegression(optimize_s = False, maxiter = 1000, fit_intercept = False, obj = 'direct', tol = 1e-7)
            gv1.fit(H, y, prior_init, s2_init = 0.001)
            b0 = gv1.coef
            #prior = gv1.prior
        else:
            b0 = np.dot(Hinv, x0)

    if x0 is None:
        x0 = np.dot(H, b0)
        
    if s0 is None:
        s0 = np.var(y - x0)
        
    # Run 
    prior = toy_priors.get_ash_scaled(k = 10, sparsity = 0.9, skbase = (degree + 1) * 30)
    gv2 = LinearRegression(maxiter = 10000, obj = 'reparametrize', tol = 1e-8)
    gv2.fit(H, y, prior, b_init = b0, s2_init = s0)
    
    # Result
    coef = gv2.coef
    yest = np.dot(H, coef) + gv2.intercept
    return CRes(y = yest, coef = coef, s2 = gv2.residual_var, prior = gv2.prior, obj = gv2)


def gradvi_trendfilter_direct(y, degree, b0 = None, s0 = None, x0 = None):
    n    = y.shape[0]
    H    = basis_matrix.trendfiltering_scaled(n, degree)
    Hinv = basis_matrix.trendfiltering_inverse_scaled(n, degree)
    
    # Initialize
    if b0 is None:
        if x0 is None:
            print("GradVI direct: Running first pass for initialization")
            prior_init = toy_priors.get_ash_scaled(k = 10, sparsity = 0.9, skbase = (degree + 1) * 30) 
            gv1 = LinearRegression(optimize_s = False, maxiter = 2000, fit_intercept = False, obj = 'direct', tol = 1e-7)
            gv1.fit(H, y, prior_init, s2_init = 0.001)
            b0 = gv1.coef
            #prior = gv1.prior
        else:
            b0 = np.dot(Hinv, x0) 

    if x0 is None:
        x0 = np.dot(H, b0) 
    
    if s0 is None:
        s0 = np.var(y - x0) 
    
    # Run 
    prior = toy_priors.get_ash_scaled(k = 10, sparsity = 0.9, skbase = (degree + 1) * 30) 
    gv2 = LinearRegression(maxiter = 10000, obj = 'direct', tol = 1e-8)
    gv2.fit(H, y, prior, b_init = b0, s2_init = s0)
    
    # Result
    coef = gv2.coef
    yest = np.dot(H, coef) + gv2.intercept
    return CRes(y = yest, coef = coef, s2 = gv2.residual_var, prior = gv2.prior, obj = gv2)


def gradvi_wavelet_trendfilter(y, degree, x0 = None, s0 = None, b0 = None):
    n = y.shape[0]
    H    = basis_matrix.trendfiltering_scaled(n, degree)
    Hinv = basis_matrix.trendfiltering_inverse_scaled(n, degree)
    dj   = np.sum(np.square(H), axis = 0)

    # Run
    prior = toy_priors.get_ash_scaled(k = 10, sparsity = 0.9, skbase = (degree + 1) * 30.)
    gv1 = WaveletRegression(maxiter = 10000, tol = 1e-8)
    gv1.fit(Hinv, y, prior, x_init = x0, dj = dj)
    
    # Result
    coef = gv1.coef
    yest = gv1.signal
    return CRes(y = yest, coef = coef, s2 = gv1.residual_var, prior = gv1.prior, obj = gv1)


def cavi_trendfilter(y, degree, b0 = None, s0 = None, x0 = None):
    n    = y.shape[0]
    H    = basis_matrix.trendfiltering_scaled(n, degree)
    Hinv = basis_matrix.trendfiltering_inverse_scaled(n, degree)

    if b0 is None:
        if x0 is None:
            print("CAVI: Running first pass for initialization")
            prior0 = toy_priors.get_ash_scaled(k = 10, sparsity = 0.9, skbase = (degree + 1) * 30)
            mrash0 = MrASHR(option = "r2py", debug = False)
            mrash0.fit(H, y, prior0.sk, winit = prior0.w_init, s2init = 0.001, maxiter = 2000, update_sigma2 = False)
            b0 = mrash0.coef
        else:
            b0 = np.dot(Hinv, x0)

    if x0 is None:
        x0 = np.dot(H, b0)

    if s0 is None: 
        s0 = np.var(y - x0)

    prior = toy_priors.get_ash_scaled(k = 10, sparsity = 0.9, skbase = (degree + 1) * 30)
    mrash = MrASHR(option = "r2py", debug = False)
    mrash.fit(H, y, prior.sk, binit = b0, winit = prior.w_init, s2init = s0, maxiter = 10000)
    prior.update_w(mrash.prior)
    yest = np.dot(H, mrash.coef) + mrash.intercept
    return CRes(y = yest, coef = mrash.coef, s2 = mrash.residual_var, prior = prior, obj = mrash)
    

def genlasso_trendfilter(y, degree):
    n = y.shape[0]
    Hinv = basis_matrix.trendfiltering_inverse_scaled(n, degree)
    tfR_y, tfR_fit = R_trendfilter.fit(y, order = degree)
    tfR_bhat  = np.dot(Hinv, tfR_y)
    s2 = np.var(y - tfR_y)
    return CRes(y = tfR_y, coef = tfR_bhat, s2 = s2, prior = None, obj = None)
