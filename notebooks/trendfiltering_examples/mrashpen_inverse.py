import numpy as np
import collections
from mrashpen.models.normal_means_ash_scaled import NormalMeansASHScaled

from scipy import optimize as sp_optimize
from scipy import interpolate as sp_interpolate

MINV_FIELDS = ['x', 'xpath', 'objpath', 'success', 'message', 'niter', 'is_diverging']
class MinvInfo(collections.namedtuple('_MinvInfo', MINV_FIELDS)):
    __slots__ = ()


def softmax(x, base = np.exp(1)):
    if base is not None:
        beta = np.log(base)
        x = x * beta
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x, axis = 0, keepdims = True)


def penalty_func(b, ak, std, sk, dj, 
                 jac = True, 
                 softmax_base = np.exp(1),
                 method = 'fssi-cubic',
                 ngrid = 1000
                ):
    wk = softmax(ak, base = softmax_base)
    k = ak.shape[0]
    p = b.shape[0]
    Minv = shrinkage_operator_inverse(b, std, wk, sk, dj, method = method, ngrid = ngrid)
    t = Minv.x
    nm = NormalMeansASHScaled(t, std, wk, sk, d = dj)
    s2 = nm.yvar
    #rhoMtj = - nm.logML #- 0.5 * nm.yvar * np.square(nm.logML_deriv)
    rhoMtj = - nm.logML - 0.5 * np.square(t - b) / nm.yvar
    rhoMt = np.sum(rhoMtj)
    if jac:
        dHdb = (t - b) / nm.yvar
        # dMdw = nm.yvar.reshape(-1, 1) * nm.logML_deriv_wderiv
        # dMdtinv = 1 / (1 + nm.yvar * nm.logML_deriv2)
        # dtdw = - nm.logML_deriv_wderiv * (nm.yvar * dMdtinv).reshape(-1, 1)
        ## Derivative of -0.5(t-b)^2 / s2
        # dHdw = - ((t - b) / nm.yvar).reshape(-1, 1) * dtdw
        ## Derivative of -logML(t)
        # dHdw = - nm.logML_wderiv - (nm.logML_deriv).reshape(-1, 1) * dtdw
        dHdw = - nm.logML_wderiv
        dHdw = np.sum(dHdw, axis = 0)
        akjac = np.log(softmax_base) * wk.reshape(-1, 1) * (np.eye(k) - wk)
        dHda = np.sum(dHdw * akjac, axis = 1)
        return rhoMt, dHdb, dHda
    else:
        return rhoMt


def shrinkage_operator(z, std, wk, sk, dj, jac = True):
    nm = NormalMeansASHScaled(z, std, wk, sk, d = dj)
    return shrinkage_operator_nm(nm, jac = jac)


def shrink_theta(z, std, wk, sk, dj):
    return shrinkage_operator(z, std, wk, sk, dj, jac = False)


def shrinkage_operator_nm(nm, jac = True):
    M = nm.y + nm.yvar * nm.logML_deriv
    if jac:
        M_bgrad  = 1 + nm.yvar * nm.logML_deriv2
        M_wgrad  = nm.yvar.reshape(-1, 1) * nm.logML_deriv_wderiv
        M_s2grad = (nm.logML_deriv / nm._d) + (nm.yvar * nm.logML_deriv_s2deriv)
        return M, M_bgrad, M_wgrad, M_s2grad
    else:
        return M


def shrinkage_operator_inverse(b, std, wk, sk, dj, method = 'fssi-linear', theta_init = None,
                               tol = 1.48e-08, maxiter = 1000, ngrid = 50):

    def inv_func(x, b, std, wk, sk, dj):
        return shrinkage_operator(x, std, wk, sk, dj, jac = False) - b

    if theta_init is None:
        theta_init = np.zeros_like(b)
    if method == 'newton-raphson':
        Minv = Minverse_newton_raphson(b, std, wk, sk, dj, tol, maxiter, theta_init)
    elif method == 'fssi-linear':
        x = Minverse_fssi(b, std, wk, sk, dj, ngrid = ngrid)
        Minv = MinvInfo(x = x, xpath = None, objpath = None, niter = ngrid,
                        success = True, message = 'Non iterative method', is_diverging = False)
    elif method == 'fssi-cubic':
        x = Minverse_fssi(b, std, wk, sk, dj, ngrid = ngrid, interpolate = 'cubic')
        Minv = MinvInfo(x = x, xpath = None, objpath = None, niter = ngrid,
                        success = True, message = 'Non iterative method', is_diverging = False)
    elif method == 'hybr':
        opt = sp_optimize.root(inv_func, theta_init,
                               args = (b, std, wk, sk, dj),
                               method = method, jac = None, tol = tol)
        Minv = MinvInfo(x = opt.x, xpath = None, objpath = None, niter = opt.nfev,
                        success = opt.success, message = opt.message, is_diverging = opt.success)
    return Minv


def Minverse_fssi(b, std, wk, sk, dj, ngrid = 50, interpolate = 'linear'):

    def create_spline(x, y, dydx):
        n = x.shape[0]
        c = np.empty((4, n-1), dtype = y.dtype)
        xdiff = np.diff(x)
        xdiff[np.where(xdiff == 0)] = 1e-8
        slope = (y[1:] - y[:-1]) / xdiff
        t = (dydx[:-1] + dydx[1:] - 2 * slope) / xdiff
        c[0] = t / xdiff
        c[1] = (slope - dydx[:-1]) / xdiff - t
        c[2] = dydx[:-1]
        c[3] = y[:-1]
        return sp_interpolate.PPoly(c, x)

    babs = np.abs(b)
    ymax = max(babs)
    Minv = shrinkage_operator_inverse(np.array([ymax]), std, wk, sk, np.ones(1) * dj[0], method = 'hybr')
    xmax = Minv.x[0]
    xgrid = np.logspace(-4, np.log10(xmax), ngrid)
    ygrid, xderiv, _, _ = shrinkage_operator(xgrid, std, wk, sk, np.ones(ngrid) * dj[0], jac = True)
    dgrid = 1 / xderiv
    #xposgrid = np.logspace(-4, np.log10(xmax), ngrid)
    #print (f"Max values of b and M^{-1}(b) are {ymax}, {xmax}")
    #yposgrid = shrink_theta(xposgrid, std, wk, sk, np.ones(ngrid))
    #yposgrid, xderiv, _, _ = shrinkage_operator(xposgrid, std, wk, sk, np.ones(ngrid), jac = True)
    #dposgrid = 1 / xderiv
    #xgrid = np.concatenate((-xposgrid[::-1], xposgrid))
    #ygrid = np.concatenate((-yposgrid[::-1], yposgrid))
    #dgrid = np.concatenate((-dposgrid[::-1], dposgrid))
    if interpolate == 'linear':
        t_fssi = np.interp(babs, ygrid, xgrid)
        t_fssi *= np.sign(b)
    elif interpolate == 'cubic':
        cs = create_spline(ygrid, xgrid, dgrid)
        t_fssi = cs(babs)
        t_fssi *= np.sign(b)
    return t_fssi


def is_decreasing_monotonically(x, nlast = 3):
    if len(x) > nlast:
        y = x[-nlast:]
        return all([y[i] >= y[i+1] for i in range(len(y) - 1)])
    return False


def rootfind_newton_raphson(func, x0, fprime,
                            args=(), tol=1.48e-08,
                            maxiter=50, full_output=None):
    resnorm = np.inf  # norm of residuals
    objpath = list()
    xpath   = list()
    for itr in range(maxiter):
        fval = func(x0, *args)
        if fprime is True:
            fval, jval = fval
        else:
            jval = fprime(x0, *args)
        resnorm = np.sqrt(np.sum(fval**2))
        # keep full path if requested
        if full_output:
            xpath.append(x0.copy())
            objpath.append(resnorm)
        if resnorm < tol:
            break
        newton_step = fval / jval
        # do some clever damping here,
        # or use Numerical Recipes in C ch. 9.6-9.7 and do line search
        x0 -= newton_step
    if full_output:
         #return x0, fval, resnorm, newton_step, itr
        return x0, xpath, objpath, resnorm
    return x0


def Minverse_newton_raphson(b, std, wk, sk, dj, tol, maxiter, theta_init):

    def inv_func_jac(x, b, std, wk, sk, dj):
        M, M_bgrad, _, _ = shrinkage_operator(x, std, wk, sk, dj)
        return M - b, M_bgrad

    x, xpath, objpath, resnorm = rootfind_newton_raphson(inv_func_jac, theta_init, True,
                                                args = (b, std, wk, sk, dj),
                                                tol = tol,
                                                maxiter = maxiter, full_output = True)
    niter = len(xpath)
    success = False
    is_diverging = False
    if resnorm <= tol:
        success = True
        message = f"The solution converged after {niter} iterations."
    else:
        if niter < maxiter:
            message = f"Iteration stopped before reaching tolerance!"
        else:
            if is_decreasing_monotonically(objpath):
                message = f"The solution is converging, but mean square difference did not reach tolerance. Try increasing the number of iterations."
            else:
                is_diverging = True
                message = f"The solution is diverging. Try different method."
    Minv = MinvInfo(x = x, xpath = xpath, objpath = objpath, niter = niter,
                    success = success, message = message, is_diverging = is_diverging)
    return Minv


