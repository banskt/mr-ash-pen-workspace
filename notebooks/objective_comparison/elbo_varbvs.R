source('/home/saikat/Documents/work/sparse-regression/mr-ash-penalized/mr-ash-pen/src/mrashpen/utils/elbo_varbvs.R')

get_posterior <- function(X, y, k = 20) {
  fit <- varbvs::varbvsmix(X, NULL, y, k, verbose = FALSE,
                           drop.threshold = 0, maxiter = 1)
  return (fit)
}

fit_varbvsmix <- function (X, y, post, maxiter = 2) {
  fit <- varbvs::varbvsmix(X, NULL, y, post$sa, post$sigma,
                           post$w, post$alpha, post$mu,
                           FALSE, FALSE, FALSE,
                           verbose = FALSE,
                           drop.threshold = 0, maxiter = 1)
  b   <- as.vector(coef(fit))
  return(list(fit = fit, mu = b[1], beta = b[-1]))
}

set.seed(100)
sd = 1
n = 100
p = n
X = matrix(0,nrow=n,ncol=n)
for(i in 1:n){
  X[i:n,i] = 1:(n-i+1)
}
btrue = rep(0,n)
btrue[40] = 8
btrue[41] = -8

Y = X %*% btrue + sd*rnorm(n)
norm = colSums(X^2)
X = t(t(X)/sqrt(norm))
btrue = btrue * sqrt(norm)
plot(Y)
lines(X %*% btrue)

init_posterior = get_posterior(X, Y)
res = fit_varbvsmix(X, Y, init_posterior)
lines(X %*% res$beta + res$mu)
logZ <- ash_elbo(X, Y, res$fit$sa, res$fit$w, res$fit$sigma, 
                  res$fit$alpha, res$fit$mu, res$fit$s)
res$fit$logZ[length(res$fit$logZ)]
logZ
