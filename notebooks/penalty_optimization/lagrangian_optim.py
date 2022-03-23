import numpy as np
import collections
from mrashpen.models.normal_means_ash_scaled import NormalMeansASHScaled
from scipy import optimize as sp_optimize

MINV_FIELDS = ['x', 'xpath', 'objpath', 'success', 'message', 'niter', 'is_diverging']
class MinvInfo(collections.namedtuple('_MinvInfo', MINV_FIELDS)):
    __slots__ = ()


def softmax(x, base = np.exp(1)):
    if base is not None:
        beta = np.log(base)
        x = x * beta
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x, axis = 0, keepdims = True)


def penalty_operator(z, std, wk, sk, dj):
    nm = NormalMeansASHScaled(z, std, wk, sk, d = dj)
    tvar = (std * std) / dj
    lambdaj = - nm.logML - 0.5 * tvar * np.square(nm.logML_deriv)
    return lambdaj


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


def is_decreasing_monotonically(x, nlast = 3):
    if len(x) > nlast:
        y = x[-nlast:]
        return all([y[i] >= y[i+1] for i in range(len(y) - 1)])
    return False


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



def shrinkage_operator_inverse(b, std, wk, sk, dj, method = 'hybr', theta_init = None,
                               tol = 1.48e-08, maxiter = 1000):

    def inv_func(x, b, std, wk, sk, dj):
        return shrinkage_operator(x, std, wk, sk, dj, jac = False) - b

    if theta_init is None:
        theta_init = np.zeros_like(b)
    if method == 'newton-raphson':
        Minv = Minverse_newton_raphson(b, std, wk, sk, dj, tol, maxiter, theta_init)
    else:
        opt = sp_optimize.root(inv_func, theta_init,
                               args = (b, std, wk, sk, dj),
                               method = method, jac = None, tol = tol)
        Minv = MinvInfo(x = opt.x, xpath = None, objpath = None, niter = opt.nfev,
                        success = opt.success, message = opt.message, is_diverging = opt.success)
    return Minv


## def unshrink_b(b, std, wk, sk, dj, theta = None, max_iter = 100, tol = 1e-8):
##     # this is the initial value of theta
##     if theta is None:
##         theta = np.zeros_like(b)
##     # Newton-Raphson iteration
##     for itr in range(max_iter):
##         nmash = NormalMeansASHScaled(theta, std, wk, sk, d = dj)
##         Mtheta, Mtheta_bgrad, _, _ = shrinkage_operator(nmash)
##         theta_new = theta - (Mtheta - b) / Mtheta_bgrad
##         diff = np.sum(np.square(theta_new - theta))
##         theta = theta_new
##         obj = np.sum(- nmash.logML - 0.5 * nmash.yvar * np.square(nmash.logML_deriv))
##         print(obj)
##         if diff <= tol:
##             break
##     return theta

def penalty_operator_lagrangian(z, wk, std, sk, dj, lgrng, b):
    Mt  = shrinkage_operator(z, std, wk, sk, dj, jac = False)
    hwt = penalty_operator(z, std, wk, sk, dj)
    obj = np.sum(hwt) + np.sum(lgrng * (Mt - b))
    return obj

def penalty_operator_lagrangian_deriv(z, wk, std, sk, dj, lgrng, b):
    '''
    The Normal Means model
    '''
    nmash = NormalMeansASHScaled(z, std, wk, sk, d = dj)
    '''
    gradient w.r.t lambda_j (lagrangian penalty)
    '''
    M, M_bgrad, M_wgrad, M_s2grad  = shrinkage_operator_nm(nmash)
    dLdl = M - b
    '''
    gradient w.r.t wk (prior mixture coefficients)
    '''
    tvar  = (std * std) / dj
    v2_ld_ldwd = tvar.reshape(-1, 1) * nmash.logML_deriv.reshape(-1, 1) * nmash.logML_deriv_wderiv
    ## gradient of first term and second term of the lagrangian
    l1_wgrad = - nmash.logML_wderiv - v2_ld_ldwd
    l2_wgrad = lgrng.reshape(-1, 1) * M_wgrad
    dLdw = np.sum(l1_wgrad + l2_wgrad, axis = 0)
    '''
    gradient w.r.t theta
    '''
    l1_tgrad = - nmash.logML_deriv  - tvar * nmash.logML_deriv * nmash.logML_deriv2
    l2_tgrad = lgrng * (1 + tvar * nmash.logML_deriv2)
    dLdt = l1_tgrad + l2_tgrad
    return dLdl, dLdw, dLdt

def objective_numeric_lagrangian(params, std, sk, dj, b, p, k, softmax_base):
    zj = params[:p]
    lj = params[p:2*p]
    ak = params[2*p:]
    wk = softmax(ak, base = softmax_base)
    dLdl, dLdw, dLdt = penalty_operator_lagrangian_deriv(zj, wk, std, sk, dj, lj, b)
    akjac = np.log(softmax_base) * wk.reshape(-1, 1) * (np.eye(k) - wk)
    dLda = np.sum(dLdw * akjac, axis = 1)
    obj = np.sqrt(np.sum(np.square(dLdl)) + np.sum(np.square(dLda)) + np.sum(np.square(dLdt)))
    return obj
