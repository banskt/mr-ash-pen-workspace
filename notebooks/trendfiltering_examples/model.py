import numpy as np
import collections


def basis_matrix(n, p, k):
    '''
    adapted from [Tibshirani, 2014](https://doi.org/10.1214/13-AOS1189)
    Equation (22) [page 303]
    '''
    m = min(n, p)
    X = np.zeros((m, m))
    if k == 0:
        for j in range(m):
            X[j:m, j] = 1
    else:
        # j = 1, ..., k+1
        seq = np.arange(1, m+1).reshape(m,1)
        X[:, :k + 1] = np.power(np.tile(seq, k+1), np.arange(k+1)) / np.power(m, np.arange(k+1))
        # j > k + 1
        for j in range(k+1, m):
            khalf = int(k / 2) if k % 2 == 0 else int((k + 1) / 2)
            X[(j - khalf + 1):, j] = np.power(np.arange(j - khalf + 1, m) - j + khalf, k) / np.power(m, k)
    G = np.zeros((n, p))
    if n >= p:
        G[:p, :] = X
    else:
        G[:, :n] = X
    return G


def sample_betas (p, bidx, method="normal", bfix=None):
    '''
    Sample betas from a distribution (method = normal / gamma)
    or use a specified value for all betas:
        bfix = const -> all non-zero betas will have beta = const
        bfix = [a, b, c, ...] -> all non-zero betas can be specified using an array
    Note: 
        when sampling from the gamma distribution,
        a random sign (+/-) will be assigned
    '''
    beta = np.zeros(p)
    s = bidx.shape[0]

    # helper function to obtain random sign (+1, -1) with equal proportion (f = 0.5)
    def sample_sign(n, f = 0.5):
        return np.random.choice([-1, 1], size=n, p=[f, 1 - f]) 

    # sample beta from Gaussian(mean = 0, sd = 1)
    if method == "normal":
        beta[bidx] = np.random.normal(size = s)

    # receive fixed beta input
    elif method == "fixed":
        assert bfix is not None, "bfix is not specified for fixed signal"
        if isinstance(bfix, (collections.abc.Sequence, np.ndarray)):
            assert len(bfix) == s, "Length of input coefficient sequence is different from the number of non-zero coefficients"
            beta[bidx] = bfix
        else:
            beta[bidx] = np.repeat(bfix, s)

    # sample beta from a Gamma(40, 0.1) distribution and assign random sign
    elif method == "gamma":
        params = [40, 0.1]
        beta[bidx] = np.random.gamma(params[0], params[1], size = s)
        beta[bidx] = np.multiply(beta[bidx], sample_sign(s))

    return beta


def get_responses (X, b, sd):
    return np.dot(X, b) + sd * np.random.normal(size = X.shape[0])


def changepoint (n, p, s, std, 
                 order = 0, signal = "gamma", seed = None, 
                 bfix = None,
                 bidx = None):
    '''
    Generate trend-filtering data, using the following inputs.
        n: number of samples
        p: number of variables (usually n = p)
        s: number of changepoints
        std: standard deviation of the noise
        order: order of the basis matrix, default 0
        signal: distribution from which beta is sampled, default Gamma(40, 0.1)
        seed: set a seed for reproducibility
        bfix: specify the values of beta
        bidx: specify the changepoint locations
    '''
    if seed is not None: np.random.seed(seed)
    X     = basis_matrix(n, p, order)
    Xtest = X.copy()
    # sample betas
    m = min(n, p)
    imin = order + 1
    imax = m
    # if the changepoints are not specified
    if bidx is None:
        bidx  = np.random.choice(np.arange(imin, imax), s, replace = False)
    # obtain values of beta
    beta   = sample_betas(p, bidx, method = signal, bfix = bfix)
    # obtain signal-to-noise ratio from betas
    signal = np.mean(np.square(beta[beta!=0]))
    snr    = signal / np.square(std)
    # calculate the responses
    y      = get_responses(X,     beta, std)
    ytest  = get_responses(Xtest, beta, std)
    return X, y, Xtest, ytest, beta, snr
