#' Mixed multivariate spike-and-slab LASSO wrapper
#'
#' \\code{mixed_mssl} interfaces with the underlying mSSL C++ and R code to
#' fit a mixed-outcome multivariate spike-and-slab LASSO (mSSL) model. It
#' accepts a design matrix \code{X} and response matrix \code{Y} with both
#' binary and continuous columns, and automatically reorders the columns
#' so that all binary responses are processed first.
#'
#'
#' @param X A numeric \code{n x p} design matrix of covariates.
#' @param Y A numeric \code{n x q} response matrix, with binary entries
#'   coded as 0/1 and continuous entries as real values.
#' @param response_types A character vector of length \code{q} specifying
#'   each response as either "binary" or "continuous".
#' @param lambdas A list with elements \code{lambda1} (slab penalty) and
#'   \code{lambda0} (spike penalty). Both can be scalars or vectors for grid.
#' @param xis A list with elements \code{xi1} (slab penalty for Omega) and
#'   \code{xi0} (spike penalty for Omega). Scalars or vectors for grid.
#' @param theta_hyper_params Numeric vector length 2 giving Beta prior
#'   hyperparameters \eqn{(a_theta, b_theta)} for slab mixing weight.
#' @param eta_hyper_params Numeric vector length 2 giving Beta prior
#'   hyperparameters \eqn{(a_eta, b_eta)} for precision mixing weight.
#' @param diag_penalty Non-negative scalar penalty applied to diagonal
#'   elements of Omega (default is TRUE or 1).
#' @param max_iter Maximum number of outer iterations for the EM algorithm.
#' @param eps Convergence tolerance for EM updates.
#' @param s_max_condition Maximum condition number allowed for X'X.
#'   If \code{NULL}, defaults to \code{10 * n}.
#' @param obj_counter_max Number of successive non-increasing objective
#'   updates allowed before termination.
#' @param verbose Integer flag (0/1) controlling verbosity of mSSL output.
#' @param nrep Number of Monte Carlo samples used in the LinESS inner loop.
#' @param nskp Number of initial samples to discard in LinESS.
#'
#' @return A list with components:
#' \describe{
#'   \item{B}{A \code{p x q} matrix of estimated regression coefficients, in
#'     the original response order.}
#'   \item{Omega}{A \code{q x q} estimated precision (inverse covariance)
#'     matrix, in the original response order.}
#' }
#'
#'
mixed_mssl <- function(
    X,                    # n x p design matrix
    Y,                    # n x q response matrix
    response_types,       # character vector of length q: "binary" or "continuous"
    lambdas,             # list(lambda1=..., lambda0=...)
    xis,                 # list(xi1=..., xi0=...)
    theta_hyper_params,  # c(a_theta, b_theta)
    eta_hyper_params,    # c(a_eta, b_eta)
    diag_penalty    = 1,
    max_iter        = 10000,
    eps             = 1e-4,
    s_max_condition = NULL,
    obj_counter_max = 5,
    verbose         = 0,
    nrep            = 2000,
    nskp            = 1
) {
  if (!exists(".mixed_mssl_initialized", envir = .GlobalEnv)) {
    suppressMessages({
      suppressWarnings({
        Rcpp::sourceCpp("mSSL/src/mSSL.cpp", showOutput = FALSE)
        source("mSSL/R/mSSL.R", echo = FALSE, verbose = FALSE)
      })
    })
    assign(".mixed_mssl_initialized", TRUE, envir = .GlobalEnv)
  }
  
  if (length(response_types) != ncol(Y)) {
    stop("response_types must have length q = ncol(Y)")
  }
  if (!all(response_types %in% c("binary","continuous"))) {
    stop("response_types entries must be 'binary' or 'continuous'")
  }
  
  n <- nrow(X)
  if (is.null(s_max_condition)) s_max_condition <- 10 * n
  
  bin_cols   <- which(response_types == "binary")
  cont_cols  <- which(response_types == "continuous")
  new_order  <- c(bin_cols, cont_cols)
  Y2         <- Y[, new_order, drop = FALSE]
  binidxend  <- length(bin_cols)-1
  
  fit <- mpcSSL_dpe(
    X, Y2,
    binidxend           = binidxend,
    lambdas             = lambdas,
    xis                 = xis,
    theta_hyper_params  = theta_hyper_params,
    eta_hyper_params    = eta_hyper_params,
    diag_penalty        = diag_penalty,
    max_iter            = max_iter,
    eps                 = eps,
    s_max_condition     = s_max_condition,
    obj_counter_max     = obj_counter_max,
    verbose             = verbose,
    nrep                = nrep,
    nskp                = nskp
  )
  
  inv_order  <- match(seq_along(new_order), new_order)
  B_shuf     <- fit$B       # p x q
  Omega_shuf <- fit$Omega   # q x q
  
  B_est      <- B_shuf[, inv_order, drop = FALSE]
  Omega_est  <- Omega_shuf[inv_order, inv_order, drop = FALSE]
  
  return(list(
    B     = B_est,
    Omega = Omega_est
  ))
}



