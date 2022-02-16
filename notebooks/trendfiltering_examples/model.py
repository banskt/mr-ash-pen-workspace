import numpy as np
import collections
import patsy # for the B-spline bases


def truncated_power_basis_matrix(n, k):
    '''
    Truncated power basis matrix for degree k, order k + 1 with m data points
    adapted from [Tibshirani, 2014](https://doi.org/10.1214/13-AOS1189)
    Equation (22) [page 303]
    '''
    # Note: Python is zero indexed.
    X = np.zeros((n, n))
    if k == 0:
        for j in range(n):
            X[j:n, j] = 1
    else:
        npowerk = np.power(n, k)
        '''
        j = 1, ..., k+1
        '''
        # create a nx1 matrix with 1, ..., n
        seq = np.arange(1, n+1).reshape(n, 1)
        # repeat (tile) the matrix k+1 times.
        # np.tile(seq, k+1) is a n x k+1 matrix whose each column contains 1..n
        # raise each column to the power of 0, ..., k 
        X[:, :k + 1] = np.power(np.tile(seq, k+1), np.arange(k+1)) / np.power(n, np.arange(k+1))
        '''
        j > k + 1
        '''
        for j in range(k+1, n):
            khalf = int(k / 2) if k % 2 == 0 else int((k + 1) / 2)
            # non-zero value if row i > j - khalf, that is from i = j + 1 - khalf
            # for column j, one-base indices of those rows are np.arange(j + 1 - khalf, n) + 1
            X[(j - khalf + 1):, j] = np.power(np.arange(j - khalf + 1, n) - j + khalf, k) / npowerk
    return X


def discrete_difference_operator_check(n, k):
    '''
    Returns D(k+1)
    This is the exact definition used in [Tibshirani, 2014]
    and should only be used for checking the faster implementation below
    '''
    if k == 0:
        D = np.eye(n)
    else:
        # define D(1)
        D = np.zeros((n-1, n))
        for i in range(n-1):
            D[i, i] = -1
            D[i, i+1] = 1
        D1 = D.copy()
        for j in range(1, k):
            Dj = D.copy()
            D = np.dot(D1[:n-j-1, :n-j], Dj)
    return D


def discrete_difference_operator(n, k, return_row = False):
    '''
    Returns D(k+1)
    A fast implementation without any dot product.
        return_row = True returns only the first row D[0, :]
    '''
    Drow = np.zeros((2, k + 2))
    for i in range(2):
        Drow[i, i] = 1
    for j in range(k):
        Drow[0, :]     = Drow[1,:] - Drow[0, :]
        Drow[1, 1:j+3] = Drow[0, :j+2]
    if not return_row:
        D = np.zeros((n - k, n))
        for irow in np.arange(n - k):
            D[irow, irow:irow + k + 1] = Drow[0, :k+1]
    else:
        D = np.zeros(n)
        D[:k+2] = Drow[0, :k+2]
    return D


def trendfiltering_basis_matrix_inverse_check(n, k):
    '''
    Returns the inverse of the trendfiltering basis matrix H
    This is the exact definition used in [Tibshirani, 2014]
    and should only be used for checking the faster implementation below.
    '''
    Dk = discrete_difference_operator(n, k + 1)
    Minv = np.zeros((n, n))
    for i in range(k + 1):
        Drow = discrete_difference_operator(n, i, return_row = True)
        Minv[i, :] = Drow
    Minv[i+1:, :] = Dk
    #tconst = np.power(n, k) / np.math.factorial(k)
    return Minv


def trendfiltering_basis_matrix_inverse(n, k):
    '''
    Returns the inverse of the trendfiltering basis matrix H
    This is a faster implementation without any dot product.
    Check output with trendfiltering_basis_matrix(n, k)
    '''
    Hinv = np.zeros((n, n))
    for i in range(2):
        Hinv[i, i] = 1
    for i in range(1, k + 2):
        Hinv[i, :i+2] = Hinv[i, :i+2] - Hinv[i-1, :i+2]
        Hinv[i+1, 1:i+3] = Hinv[i, :i+2]
    for j in range(1, n-k-2):
        irow = i + j + 1
        Hinv[irow, j+1:j+k+3] = Hinv[i, :k+2]
    return Hinv


def trendfiltering_basis_matrix_check(n, k):
    '''
    Returns the trendfiltering basis matrix H
    This is the exact definition used in [Tibshirani, 2014]
    and should only be used for checking the faster implementation below.
    '''
    #tconst = np.power(n, k) / np.math.factorial(k)
    def getMi(n, i):
        M = np.zeros((n, n))
        M[:i, :i] = np.eye(i)
        M[i:, i:] = np.tril(np.ones((n-i, n-i)))
        return M
    M = getMi(n, 0)
    for i in range(1, k+1):
        M = np.dot(M, getMi(n, i))
    return M


def trendfiltering_basis_matrix(n, k):
    '''
    Returns the trendfiltering basis matrix H
    This is a faster implementation without any dot product.
    Check output with trendfiltering_basis_matrix_check(n, k)
    '''
    H = np.zeros((n, n))
    A = list([np.ones(n) for i in range(k + 1)])
    for i in range(1, k + 1):
        A[i] = np.cumsum(A[i-1])
    for j in range(k):
        H[j:, j] = A[j][:n-j]
    for j in range(k, n):
        H[j:, j] = A[k][:n-j]
    return H


def trendfiltering_basis_matrix_tibshirani(n, k):
    '''
    This is an alternate definition of the trendfiltering basis matrix H,
    see Eq 27 in [Tibshirani, 2014].
    I have not calculated the inverse of this matrix.
    '''
    # ----------------------------
    # Let's not delete the explicit version, slow
    # but this is what we are doing.
    # ----------------------------
    # H = np.zeros((n, n))
    # npowerk = np.power(n, k)
    # seq = np.arange(1, n+1).reshape(n, 1)
    # H[:, :k + 1] = np.power(np.tile(seq, k+1), np.arange(k+1)) / np.power(n, np.arange(k+1))
    # for j in range(k+1, n):
    #     for i in range(n):
    #         if i > j - 1:
    #             Hij = 1.0
    #             for l in range(1, k+1):
    #                 Hij *= (i - j + k - l + 1)
    #             H[i, j] = Hij #/ np.power(n, k)
    # ----------------------------
    # ----------------------------
    # A function for fast calculation of the lower triangular matrix
    # obtained from the third condition in Eq 27
    def lower_tril_from_vector(S):
        n = S.shape[0]
        X = np.zeros((n, n))
        X[:, 0] = S
        for j in range(1, n):
            X[j:, j] = S[:-j]
        return X
    # ----------------------------
    # instead of calculating each element
    # precalculate the vector of products in the third condition only once 
    # and fill up the lower triangular matrix of the basis
    npowerk = np.power(n, k)
    kfact = np.math.factorial(k)
    S = np.ones(n - k - 1)
    for i in range(1, n - k - 1):
        S[i] = S[i - 1] * (i + k) / i
    # ----------------------------
    H = np.zeros((n, n))
    seq = np.arange(1, n+1).reshape(n, 1)
    H[:, :k + 1] = np.power(seq, np.arange(k+1)) / np.power(n, np.arange(k+1))
    H[k+1:, k+1:] = lower_tril_from_vector(S * kfact / npowerk)
    return H
   

def sample_betas (p, method="normal", bfix=None):
    '''
    Sample betas from a distribution (method = normal / gamma)
    or use a specified value for all betas:
        bfix = const -> all betas will have beta = const
        bfix = [a, b, c, ...] -> all betas can be specified using an array
    Note: 
        when sampling from the gamma distribution,
        a random sign (+/-) will be assigned
    '''
    beta = np.zeros(p)

    # helper function to obtain random sign (+1, -1) with equal proportion (f = 0.5)
    def sample_sign(n, f = 0.5):
        return np.random.choice([-1, 1], size=n, p=[f, 1 - f]) 

    # sample beta from Gaussian(mean = 0, sd = 1)
    if method == "normal":
        beta = np.random.normal(size = p)

    # receive fixed beta input
    elif method == "fixed":
        assert bfix is not None, "bfix is not specified for fixed signal"
        if isinstance(bfix, (collections.abc.Sequence, np.ndarray)):
            assert len(bfix) == p, "Length of input coefficient sequence is different from the number of non-zero coefficients"
            beta = bfix
        else:
            beta = np.repeat(bfix, p)

    # sample beta from a Gamma(40, 0.1) distribution and assign random sign
    elif method == "gamma":
        params = [40, 0.1]
        beta = np.random.gamma(params[0], params[1], size = p)
        beta = np.multiply(beta, sample_sign(p))

    return beta


def get_responses (X, b, sd):
    return np.dot(X, b) + sd * np.random.normal(size = X.shape[0])


def changepoint (x, knots, std, 
                 degree = 0, signal = "gamma", seed = None,
                 include_intercept = False, bfix = None,
                 eps = 1e-4):
    '''
    Generate trend-filtering data, using the following inputs.
        x: data points
        knots: location of changepoints
        std: standard deviation of the noise
        degree: degree of the B-spline basis, default 0
        signal: distribution from which beta is sampled, default Gamma(40, 0.1)
        seed: set a seed for reproducibility
        include_intercept: whether to include a basis function for intercept
        bfix: specify the values of betas for the basis functions
              Note: number of bases = k + degree + 1 if include_intercept = True
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
    beta   = sample_betas(nbases, method = signal, bfix = bfix)
    # ------------------------------
    # Generate the function without noise 
    ytrue = np.dot(bspline_bases, beta)
    # ------------------------------
    # Map the data to trendfiltering bases
    # set low values of beta to zero and regenerate y
    H     = trendfiltering_basis_matrix(n, degree)
    Hinv  = trendfiltering_basis_matrix_inverse(n, degree)
    btrue = np.dot(Hinv, ytrue)
    btrue[np.abs(btrue) <= eps] = 0.
    noise = np.random.normal(0, std, size = n * 2)
    ytrue = np.dot(H, btrue)
    y     = ytrue + noise[:n]
    # ------------------------------
    # Some test data?
    ytest  = ytrue + noise[n:]
    # ------------------------------
    # Signal to noise ratio 
    # (experimental)
    signal = np.mean(np.square(btrue[btrue != 0]))
    snr    = signal / np.square(std)
    return H, y, ytest, btrue, snr


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
    H, y, ytest, btrue, snr = changepoint(x, knots, std, degree = 3,
                                          signal = "fixed", bfix = btrue,
                                          seed = seed)
    return H, y, ytest, btrue, snr

