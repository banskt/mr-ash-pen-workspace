library("dscrutils")

get_input_modules <- function (x, s, cmark) {
  xnew <- gsub(cmark, "$", x)
  return (c(strsplit(xnew, s)[[1]]))
}

dsc_outdir <- "/home/saikat/Documents/work/sparse-regression/mr-ash-penalized/notebooks/simulation_setup/trial"
separator <- "::"
cmarker <- "##"
#target_str <- 'simulate::simulate.dims::simulate.se::simulate.rho::simulate.sfix::simulate.pve::fit::fit.DSC_TIME::mse.err::coef_mse.err'
#targets <- get_input_modules(target_str, separator, cmarker)
targets <- c("simulate", "simulate.dims", "simulate.se", "simulate.rho", "simulate.sfix", "simulate.pve", "fit", 
             "fit.DSC_TIME", "mse.err", "coef_mse.err")
dscout <- dscrutils::dscquery(dsc.outdir = dsc_outdir, targets = targets)