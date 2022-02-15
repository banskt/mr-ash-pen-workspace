library(genlasso)
library(ggplot2)
setwd("/home/saikat/Documents/work/sparse-regression/mr-ash-penalized/notebooks/trendfiltering_examples")
hills_data = readRDS("hills_data.rds")
out = genlasso::trendfilter(hills_data$y, ord=3)
cv = genlasso::cv.trendfilter(out)
plot(out, lambda = cv$lambda.1se, main = "One SE rule")

fit_trendfilter <- function (y, order = 1, nfolds = 5, cvlambda = "1se") {
  #pos   <- 1:length(y)
  #out   <- genlasso::trendfilter(y, pos, X, ord = order)
  out   <- genlasso::trendfilter(y, ord = order)
  cvout <- genlasso::cv.trendfilter(out, k = nfolds)
  cvlam <- if (cvlambda == "1se") cvout$lambda.1se else cvout$lambda.min
  cvidx <- if (cvlambda == "1se") cvout$i.1se else cvout$i.min
  b     <- coef(out, lambda = cvlam)
  #b    <- out$beta[,cvidx]
  ypred <- out$fit[, cvidx]
  return (list(fit = out, cv = cvout, mu = 0, 
               beta = as.vector(b$beta), ypred = ypred, df = b$df))
}

gcv_genlasso <- function(object) {
  lams <- object$lambda
  df   <- object$df
  n    <- length(object$y)
  ymat <- matrix(object$y, n, length(lams))
  pred <- object$fit
  err  <- colMeans((ymat-pred)^2)/(1-df/n)^2
  names(err) <- round(lams,3)
  lam_min <- lams[which.min(err)]
  out <- list(err = err, lambda = lams, lambda.min = lam_min, i.min = which(lams == lam_min))
  class(out) <- c("gcv.genlasso", "list")
  return (out)
}


fit_genlasso <- function (y, order = 1) {
  n     <- length(y)
  D     <- genlasso::getDtf(n, order)
  out   <- genlasso::genlasso(y = y, X = diag(n), D = D)
  cvout <- gcv_genlasso(out)
  cvidx <- cvout$i.min
  cvlam <- cvout$lambda.min
  b     <- coef(out, lambda = cvlam)
  ypred <- out$fit[, cvidx]
  return (list(fit = out, cv = cvout, mu = 0, 
               beta = as.vector(b$beta), ypred = ypred, df = b$df))
}

outtf <- fit_trendfilter(hills_data$y, order = 3)
outglasso <- fit_genlasso(hills_data$y, order = 3)
pdata <- data.frame(x=seq(length(hills_data$y)), 
                    y=hills_data$y, 
                    trendfilter = outtf$ypred, 
                    genlasso = outglasso$ypred)
ggplot(pdata, aes(x = x, y = y)) + geom_point() + 
  geom_line(aes(x = x, y = trendfilter), color = 'blue') +
  geom_line(aes(x = x, y = genlasso), color = 'red') +
  xlab("Sample indices") +
  ylab("y")