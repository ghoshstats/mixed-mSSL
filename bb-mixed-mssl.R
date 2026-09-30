# Bayesian-bootstrap Spike-and-Slab LASSO for mixed-mSSL
#
# This file adds an approximate posterior sampler based on independently
# weighted and prior-jittered mixed-mSSL optimizations. It deliberately uses a
# fixed pair of spike penalties in every draw. Select the final penalties with
# the existing dynamic posterior exploration before running this routine.

.bb_mixed_mssl_state <- new.env(parent = baseenv())

.bb_mixed_mssl_source_file <- tryCatch(
  normalizePath(sys.frame(1)$ofile, winslash = "/", mustWork = TRUE),
  error = function(e) NA_character_
)

.bb_mixed_mssl_state$root <- if (!is.na(.bb_mixed_mssl_source_file)) {
  dirname(.bb_mixed_mssl_source_file)
} else {
  normalizePath(getwd(), winslash = "/", mustWork = TRUE)
}

.bb_mixed_mssl_initialize <- function(rebuild = FALSE, show_output = FALSE) {
  if (exists("bb_mpcssl_fixed_cpp",
             envir = .bb_mixed_mssl_state,
             inherits = FALSE) && !isTRUE(rebuild)) {
    return(invisible(TRUE))
  }

  if (!requireNamespace("Rcpp", quietly = TRUE)) {
    stop("Package 'Rcpp' is required.", call. = FALSE)
  }
  if (!requireNamespace("RcppArmadillo", quietly = TRUE)) {
    stop("Package 'RcppArmadillo' is required.", call. = FALSE)
  }
  if (!requireNamespace("mSSL", quietly = TRUE)) {
    stop(
      paste0(
        "Package 'mSSL' is required. Install the development branch with ",
        "remotes::install_github('YunyiShen/mSSL@dev')."
      ),
      call. = FALSE
    )
  }

  cpp_file <- file.path(
    .bb_mixed_mssl_state$root,
    "mSSL", "src", "bb_mixed_mssl.cpp"
  )
  if (!file.exists(cpp_file)) {
    stop("Cannot find mSSL/src/bb_mixed_mssl.cpp.", call. = FALSE)
  }

  Rcpp::sourceCpp(
    file = cpp_file,
    env = .bb_mixed_mssl_state,
    rebuild = rebuild,
    showOutput = show_output,
    verbose = FALSE
  )
  invisible(TRUE)
}

.bb_mixed_mssl_jitter <- function(n, rate) {
  signs <- ifelse(stats::runif(n) < 0.5, -1, 1)
  signs * stats::rexp(n, rate = rate)
}

.bb_mixed_mssl_validate_scalar <- function(x, name, lower = -Inf,
                                           strict = FALSE) {
  if (!is.numeric(x) || length(x) != 1L || !is.finite(x)) {
    stop(name, " must be one finite numeric value.", call. = FALSE)
  }
  bad <- if (strict) x <= lower else x < lower
  if (bad) {
    relation <- if (strict) "greater than" else "at least"
    stop(name, " must be ", relation, " ", lower, ".", call. = FALSE)
  }
  invisible(x)
}

.bb_mixed_mssl_reorder_fit <- function(fit, inverse_order) {
  fit$B <- fit$B[, inverse_order, drop = FALSE]
  fit$centered_B <- fit$centered_B[, inverse_order, drop = FALSE]
  fit$center <- fit$center[, inverse_order, drop = FALSE]
  fit$centered_B_std <- fit$centered_B_std[, inverse_order, drop = FALSE]
  fit$center_std <- fit$center_std[, inverse_order, drop = FALSE]
  fit$Omega <- fit$Omega[inverse_order, inverse_order, drop = FALSE]
  fit$Sigma <- fit$Sigma[inverse_order, inverse_order, drop = FALSE]
  fit$alpha <- fit$alpha[inverse_order]
  fit
}

.bb_mixed_mssl_empty_arrays <- function(p, q, n_draws,
                                        predictor_names, response_names) {
  list(
    B = array(
      NA_real_, c(p, q, n_draws),
      dimnames = list(predictor_names, response_names, NULL)
    ),
    centered_B = array(
      NA_real_, c(p, q, n_draws),
      dimnames = list(predictor_names, response_names, NULL)
    ),
    Omega = array(
      NA_real_, c(q, q, n_draws),
      dimnames = list(response_names, response_names, NULL)
    ),
    Sigma = array(
      NA_real_, c(q, q, n_draws),
      dimnames = list(response_names, response_names, NULL)
    ),
    alpha = matrix(
      NA_real_, q, n_draws,
      dimnames = list(response_names, NULL)
    ),
    theta = rep(NA_real_, n_draws),
    eta = rep(NA_real_, n_draws)
  )
}

#' Bayesian-bootstrap SSL uncertainty for mixed-mSSL
#'
#' Generates approximate posterior draws by repeatedly fitting a fixed-penalty
#' mixed-mSSL objective after (i) assigning normalized Dirichlet weights to the
#' observations and (ii) centering every SSL penalty for B at an independent
#' draw from its spike Laplace distribution. The graphical SSL prior for Omega
#' remains centered at zero; Omega nevertheless varies between draws through
#' the weighted likelihood and the jointly updated B.
#'
#' @param X Numeric n by p design matrix. Do not include an intercept column.
#' @param Y Numeric n by q response matrix.
#' @param response_types Length-q character vector containing "binary" or
#'   "continuous".
#' @param n_draws Number of BB-SSL pseudo-posterior draws.
#' @param lambdas List with scalar lambda1 and lambda0. BB-SSL targets one fixed
#'   posterior, so paths are not accepted here.
#' @param xis List with scalar xi1 and xi0.
#' @param theta_hyper_params Length-two Beta-prior hyperparameters for theta.
#' @param eta_hyper_params Length-two Beta-prior hyperparameters for eta.
#' @param initial_fit Optional mixed_mssl fit used to initialize every draw.
#'   Supplying the endpoint selected by dynamic posterior exploration is
#'   recommended. If NULL, a fixed-penalty unweighted fit is computed first.
#' @param weight_alpha Symmetric Dirichlet concentration. The BB default is 1.
#' @param jitter Logical; if TRUE, use BB-SSL random shrinkage targets for B.
#' @param jitter_rate Laplace rate of the random targets. Defaults to lambda0.
#' @param diag_penalty Whether to penalize diagonal precision entries.
#' @param max_iter Maximum weighted MCECM iterations per draw.
#' @param eps Numerical convergence tolerance.
#' @param s_max_condition Maximum allowed condition number of the weighted
#'   residual scatter matrix.
#' @param obj_counter_max Number of consecutive small objective improvements
#'   allowed before stopping.
#' @param nrep Monte Carlo samples per latent-variable E-step.
#' @param nskp Thinning/skip argument passed to the latent Gaussian sampler.
#' @param n_cores Number of forked workers on non-Windows systems.
#' @param seed Optional master random seed.
#' @param keep_weights Store the n by n_draws weight matrix.
#' @param keep_centers Store the p by q by n_draws random-center array.
#' @param fail_action Either "warn" (retain failed draws as NA) or "stop".
#' @param verbose 0 for silence, 1 for replicate progress, 2 for C++ iteration
#'   output as well.
#' @param rebuild_cpp Force recompilation of the C++ engine.
#'
#' @return An object of class bb_mixed_mssl containing draw arrays, the base
#'   fit, diagnostics, settings, and response-order metadata.
#' @export
bb_mixed_mssl <- function(
    X,
    Y,
    response_types,
    n_draws = 500L,
    lambdas,
    xis,
    theta_hyper_params = c(1, ncol(X) * ncol(Y)),
    eta_hyper_params = c(1, ncol(Y)),
    initial_fit = NULL,
    weight_alpha = 1,
    jitter = TRUE,
    jitter_rate = NULL,
    diag_penalty = 1,
    max_iter = 500L,
    eps = 1e-3,
    s_max_condition = NULL,
    obj_counter_max = 5L,
    nrep = 200L,
    nskp = 1L,
    n_cores = 1L,
    seed = NULL,
    keep_weights = FALSE,
    keep_centers = FALSE,
    fail_action = c("warn", "stop"),
    verbose = 1L,
    rebuild_cpp = FALSE
) {
  fail_action <- match.arg(fail_action)
  X <- as.matrix(X)
  Y <- as.matrix(Y)
  storage.mode(X) <- "double"
  storage.mode(Y) <- "double"

  if (!is.numeric(X) || !is.numeric(Y) || any(!is.finite(X)) ||
      any(!is.finite(Y))) {
    stop("X and Y must be finite numeric matrices.", call. = FALSE)
  }
  if (nrow(X) != nrow(Y)) {
    stop("X and Y must have the same number of rows.", call. = FALSE)
  }
  if (ncol(X) < 1L || ncol(Y) < 1L || nrow(X) < 2L) {
    stop("X and Y must have positive dimensions and at least two rows.",
         call. = FALSE)
  }
  if (length(response_types) != ncol(Y) ||
      !all(response_types %in% c("binary", "continuous"))) {
    stop(
      "response_types must have length ncol(Y) and contain only 'binary' or 'continuous'.",
      call. = FALSE
    )
  }
  binary_columns <- which(response_types == "binary")
  if (length(binary_columns) > 0L &&
      !all(Y[, binary_columns, drop = FALSE] %in% c(0, 1))) {
    stop("Binary responses must be coded exactly as 0/1.", call. = FALSE)
  }
  if (!is.list(lambdas) || !all(c("lambda1", "lambda0") %in% names(lambdas)) ||
      !is.list(xis) || !all(c("xi1", "xi0") %in% names(xis))) {
    stop(
      "lambdas and xis must be named lists containing lambda1/lambda0 and xi1/xi0.",
      call. = FALSE
    )
  }

  lambda1 <- lambdas$lambda1
  lambda0 <- lambdas$lambda0
  xi1 <- xis$xi1
  xi0 <- xis$xi0
  .bb_mixed_mssl_validate_scalar(lambda1, "lambdas$lambda1", 0, TRUE)
  .bb_mixed_mssl_validate_scalar(lambda0, "lambdas$lambda0", lambda1)
  .bb_mixed_mssl_validate_scalar(xi1, "xis$xi1", 0, TRUE)
  .bb_mixed_mssl_validate_scalar(xi0, "xis$xi0", xi1)
  .bb_mixed_mssl_validate_scalar(weight_alpha, "weight_alpha", 0, TRUE)
  .bb_mixed_mssl_validate_scalar(eps, "eps", 0, TRUE)

  n_draws <- as.integer(n_draws)
  max_iter <- as.integer(max_iter)
  obj_counter_max <- as.integer(obj_counter_max)
  nrep <- as.integer(nrep)
  nskp <- as.integer(nskp)
  n_cores <- as.integer(n_cores)
  verbose <- as.integer(verbose)
  if (anyNA(c(n_draws, max_iter, obj_counter_max, nrep, nskp, n_cores)) ||
      n_draws < 1L || max_iter < 1L || obj_counter_max < 1L ||
      nrep < 2L || nskp < 1L || n_cores < 1L) {
    stop("Invalid integer control parameter.", call. = FALSE)
  }
  if (length(theta_hyper_params) != 2L ||
      length(eta_hyper_params) != 2L ||
      any(!is.finite(theta_hyper_params)) ||
      any(!is.finite(eta_hyper_params)) ||
      any(theta_hyper_params <= 0) || any(eta_hyper_params <= 0)) {
    stop("Beta-prior hyperparameters must be positive length-two vectors.",
         call. = FALSE)
  }

  n <- nrow(X)
  p <- ncol(X)
  q <- ncol(Y)
  if (is.null(s_max_condition)) s_max_condition <- 10 * n
  .bb_mixed_mssl_validate_scalar(
    s_max_condition, "s_max_condition", 1, TRUE
  )

  if (is.null(jitter_rate)) jitter_rate <- lambda0
  .bb_mixed_mssl_validate_scalar(jitter_rate, "jitter_rate", 0, TRUE)

  predictor_names <- colnames(X)
  if (is.null(predictor_names)) predictor_names <- paste0("X", seq_len(p))
  response_names <- colnames(Y)
  if (is.null(response_names)) response_names <- paste0("Y", seq_len(q))

  continuous_columns <- which(response_types == "continuous")
  new_order <- c(binary_columns, continuous_columns)
  inverse_order <- order(new_order)
  Y_ordered <- Y[, new_order, drop = FALSE]
  q_binary <- length(binary_columns)
  binidxend <- q_binary - 1L

  .bb_mixed_mssl_initialize(
    rebuild = rebuild_cpp,
    show_output = verbose >= 2L
  )

  cpp_fit <- .bb_mixed_mssl_state$bb_mpcssl_fixed_cpp

  if (!is.null(seed)) {
    if (length(seed) != 1L || !is.finite(seed)) {
      stop("seed must be NULL or one finite value.", call. = FALSE)
    }
    set.seed(as.integer(seed))
  }

  if (is.null(initial_fit)) {
    if (verbose >= 1L) {
      message("Computing the fixed-penalty unweighted initialization fit...")
    }
    base_raw <- cpp_fit(
      X = X,
      Y = Y_ordered,
      binidxend = binidxend,
      weights = rep(1, n),
      center_std = matrix(0, p, q),
      lambda1 = lambda1,
      lambda0 = lambda0,
      xi1 = xi1,
      xi0 = xi0,
      theta_hyper_params = theta_hyper_params,
      eta_hyper_params = eta_hyper_params,
      diag_penalty = as.integer(isTRUE(as.logical(diag_penalty))),
      max_iter = max_iter,
      eps = eps,
      s_max_condition = s_max_condition,
      obj_counter_max = obj_counter_max,
      verbose = as.integer(verbose >= 2L),
      nrep = nrep,
      nskp = nskp,
      initial_B = matrix(0, p, q),
      initial_Omega = diag(q),
      initial_theta = 0.5,
      initial_eta = 0.5
    )
    base_fit <- .bb_mixed_mssl_reorder_fit(base_raw, inverse_order)
    initial_B_ordered <- base_raw$B
    initial_Omega_ordered <- base_raw$Omega
    initial_theta <- base_raw$theta
    initial_eta <- base_raw$eta
  } else {
    if (!is.list(initial_fit) || is.null(initial_fit$B) ||
        is.null(initial_fit$Omega)) {
      stop("initial_fit must contain B and Omega.", call. = FALSE)
    }
    B0 <- as.matrix(initial_fit$B)
    Omega0 <- as.matrix(initial_fit$Omega)
    if (!identical(dim(B0), c(p, q)) ||
        !identical(dim(Omega0), c(q, q)) ||
        any(!is.finite(B0)) || any(!is.finite(Omega0))) {
      stop("initial_fit has incompatible or non-finite B/Omega matrices.",
           call. = FALSE)
    }
    initial_B_ordered <- B0[, new_order, drop = FALSE]
    initial_Omega_ordered <- Omega0[new_order, new_order, drop = FALSE]
    initial_theta <- if (!is.null(initial_fit$theta)) initial_fit$theta else 0.5
    initial_eta <- if (!is.null(initial_fit$eta)) initial_fit$eta else 0.5
    base_fit <- initial_fit
  }

  replicate_seeds <- sample.int(.Machine$integer.max, n_draws, replace = FALSE)

  fit_one <- function(draw_index) {
    set.seed(replicate_seeds[[draw_index]])
    raw_weights <- stats::rgamma(n, shape = weight_alpha, rate = 1)
    observation_weights <- n * raw_weights / sum(raw_weights)
    center_std <- if (isTRUE(jitter)) {
      matrix(
        .bb_mixed_mssl_jitter(p * q, rate = jitter_rate),
        nrow = p,
        ncol = q
      )
    } else {
      matrix(0, p, q)
    }

    if (verbose >= 1L && n_cores == 1L) {
      message("BB-SSL draw ", draw_index, " of ", n_draws)
    }

    tryCatch({
      fit <- cpp_fit(
        X = X,
        Y = Y_ordered,
        binidxend = binidxend,
        weights = observation_weights,
        center_std = center_std,
        lambda1 = lambda1,
        lambda0 = lambda0,
        xi1 = xi1,
        xi0 = xi0,
        theta_hyper_params = theta_hyper_params,
        eta_hyper_params = eta_hyper_params,
        diag_penalty = as.integer(isTRUE(as.logical(diag_penalty))),
        max_iter = max_iter,
        eps = eps,
        s_max_condition = s_max_condition,
        obj_counter_max = obj_counter_max,
        verbose = as.integer(verbose >= 2L),
        nrep = nrep,
        nskp = nskp,
        initial_B = initial_B_ordered,
        initial_Omega = initial_Omega_ordered,
        initial_theta = initial_theta,
        initial_eta = initial_eta
      )
      fit <- .bb_mixed_mssl_reorder_fit(fit, inverse_order)
      list(
        ok = TRUE,
        fit = fit,
        weights = if (keep_weights) observation_weights else NULL,
        center = if (keep_centers) fit$center else NULL,
        error = NA_character_
      )
    }, error = function(e) {
      list(
        ok = FALSE,
        fit = NULL,
        weights = if (keep_weights) observation_weights else NULL,
        center = NULL,
        error = conditionMessage(e)
      )
    })
  }

  if (n_cores > 1L && .Platform$OS.type != "windows") {
    results <- parallel::mclapply(
      seq_len(n_draws),
      fit_one,
      mc.cores = min(n_cores, n_draws),
      mc.preschedule = FALSE,
      mc.set.seed = FALSE
    )
  } else {
    if (n_cores > 1L && .Platform$OS.type == "windows") {
      warning(
        "Forked parallelism is unavailable on Windows; running sequentially.",
        call. = FALSE
      )
    }
    results <- lapply(seq_len(n_draws), fit_one)
  }

  succeeded <- vapply(results, `[[`, logical(1), "ok")
  if (!all(succeeded)) {
    errors <- unique(vapply(
      results[!succeeded], `[[`, character(1), "error"
    ))
    failure_message <- paste0(
      sum(!succeeded), " of ", n_draws, " BB-SSL draw(s) failed. ",
      paste(errors, collapse = " | ")
    )
    if (fail_action == "stop") stop(failure_message, call. = FALSE)
    warning(failure_message, call. = FALSE)
  }

  draws <- .bb_mixed_mssl_empty_arrays(
    p, q, n_draws, predictor_names, response_names
  )
  centers <- if (keep_centers) {
    array(
      NA_real_, c(p, q, n_draws),
      dimnames = list(predictor_names, response_names, NULL)
    )
  } else NULL
  weight_draws <- if (keep_weights) {
    matrix(NA_real_, n, n_draws)
  } else NULL

  diagnostics <- data.frame(
    draw = seq_len(n_draws),
    success = succeeded,
    converged = FALSE,
    early_terminate = FALSE,
    objective_terminate = FALSE,
    iterations = NA_integer_,
    objective = NA_real_,
    elapsed_seconds = NA_real_,
    error = vapply(results, `[[`, character(1), "error"),
    stringsAsFactors = FALSE
  )

  for (r in which(succeeded)) {
    fit <- results[[r]]$fit
    draws$B[, , r] <- fit$B
    draws$centered_B[, , r] <- fit$centered_B
    draws$Omega[, , r] <- fit$Omega
    draws$Sigma[, , r] <- fit$Sigma
    draws$alpha[, r] <- fit$alpha
    draws$theta[r] <- fit$theta
    draws$eta[r] <- fit$eta
    if (keep_centers) centers[, , r] <- fit$center
    if (keep_weights) weight_draws[, r] <- results[[r]]$weights

    diagnostics$converged[r] <- isTRUE(fit$converged)
    diagnostics$early_terminate[r] <- isTRUE(fit$early_terminate)
    diagnostics$objective_terminate[r] <- isTRUE(fit$objective_terminate)
    diagnostics$iterations[r] <- fit$iterations
    diagnostics$objective[r] <- fit$objective
    diagnostics$elapsed_seconds[r] <- fit$elapsed_seconds
  }

  if (keep_weights && any(!succeeded)) {
    for (r in which(!succeeded)) {
      weight_draws[, r] <- results[[r]]$weights
    }
  }

  out <- list(
    draws = draws,
    centers = centers,
    weights = weight_draws,
    base_fit = base_fit,
    diagnostics = diagnostics,
    response_types = response_types,
    settings = list(
      n_draws = n_draws,
      lambdas = list(lambda1 = lambda1, lambda0 = lambda0),
      xis = list(xi1 = xi1, xi0 = xi0),
      theta_hyper_params = theta_hyper_params,
      eta_hyper_params = eta_hyper_params,
      weight_alpha = weight_alpha,
      jitter = jitter,
      jitter_rate = jitter_rate,
      diag_penalty = diag_penalty,
      max_iter = max_iter,
      eps = eps,
      s_max_condition = s_max_condition,
      obj_counter_max = obj_counter_max,
      nrep = nrep,
      nskp = nskp,
      n_cores = n_cores,
      seed = seed
    ),
    call = match.call()
  )
  class(out) <- "bb_mixed_mssl"
  out
}

#' @export
print.bb_mixed_mssl <- function(x, ...) {
  n_success <- sum(x$diagnostics$success)
  cat("BB-mixed-mSSL pseudo-posterior\n")
  cat("  requested draws:", nrow(x$diagnostics), "\n")
  cat("  successful draws:", n_success, "\n")
  if (n_success > 0L) {
    cat(
      "  converged among successful:",
      sprintf("%.1f%%", 100 * mean(x$diagnostics$converged[x$diagnostics$success])),
      "\n"
    )
    cat(
      "  median seconds per successful draw:",
      sprintf("%.3f", stats::median(
        x$diagnostics$elapsed_seconds[x$diagnostics$success], na.rm = TRUE
      )),
      "\n"
    )
  }
  invisible(x)
}

.bb_mixed_mssl_quantiles <- function(x, probs) {
  apply(
    x,
    seq_len(length(dim(x)) - 1L),
    stats::quantile,
    probs = probs,
    na.rm = TRUE,
    names = FALSE
  )
}

#' @export
summary.bb_mixed_mssl <- function(object,
                                  probs = c(0.025, 0.5, 0.975),
                                  selection_tol = 1e-10,
                                  ...) {
  if (!inherits(object, "bb_mixed_mssl")) {
    stop("object must inherit from 'bb_mixed_mssl'.", call. = FALSE)
  }
  probs <- sort(unique(probs))
  if (length(probs) < 1L || any(!is.finite(probs)) ||
      any(probs < 0 | probs > 1)) {
    stop("probs must lie in [0,1].", call. = FALSE)
  }

  B_inclusion <- apply(
    abs(object$draws$centered_B) > selection_tol,
    c(1, 2),
    mean,
    na.rm = TRUE
  )
  Omega_inclusion <- apply(
    abs(object$draws$Omega) > selection_tol,
    c(1, 2),
    mean,
    na.rm = TRUE
  )
  diag(Omega_inclusion) <- NA_real_

  out <- list(
    B_mean = apply(object$draws$B, c(1, 2), mean, na.rm = TRUE),
    B_quantiles = .bb_mixed_mssl_quantiles(object$draws$B, probs),
    B_inclusion = B_inclusion,
    Omega_mean = apply(object$draws$Omega, c(1, 2), mean, na.rm = TRUE),
    Omega_quantiles = .bb_mixed_mssl_quantiles(object$draws$Omega, probs),
    Omega_inclusion = Omega_inclusion,
    alpha_mean = apply(object$draws$alpha, 1, mean, na.rm = TRUE),
    alpha_quantiles = apply(
      object$draws$alpha, 1, stats::quantile,
      probs = probs, na.rm = TRUE, names = FALSE
    ),
    theta_quantiles = stats::quantile(
      object$draws$theta, probs = probs, na.rm = TRUE, names = FALSE
    ),
    eta_quantiles = stats::quantile(
      object$draws$eta, probs = probs, na.rm = TRUE, names = FALSE
    ),
    probs = probs,
    diagnostics = object$diagnostics
  )
  class(out) <- "summary.bb_mixed_mssl"
  out
}

#' @export
print.summary.bb_mixed_mssl <- function(x, ...) {
  successful <- x$diagnostics$success
  cat("BB-mixed-mSSL summary\n")
  cat("  successful draws:", sum(successful), "of", length(successful), "\n")
  if (any(successful)) {
    cat(
      "  converged:",
      sprintf("%.1f%%", 100 * mean(x$diagnostics$converged[successful])),
      "\n"
    )
  }
  cat("  interval probabilities:", paste(x$probs, collapse = ", "), "\n")
  invisible(x)
}

#' @export
confint.bb_mixed_mssl <- function(object,
                                  parm = c("B", "Omega", "alpha", "theta", "eta"),
                                  level = 0.95,
                                  ...) {
  parm <- match.arg(parm)
  .bb_mixed_mssl_validate_scalar(level, "level", 0, TRUE)
  if (level >= 1) stop("level must be less than 1.", call. = FALSE)
  probs <- c((1 - level) / 2, 1 - (1 - level) / 2)
  if (parm %in% c("theta", "eta")) {
    return(stats::quantile(
      object$draws[[parm]], probs = probs, na.rm = TRUE, names = FALSE
    ))
  }
  .bb_mixed_mssl_quantiles(object$draws[[parm]], probs)
}

#' Predict posterior mean/probability surfaces from BB-mixed-mSSL
#'
#' @param object A bb_mixed_mssl object.
#' @param newdata Numeric matrix with the same p columns as the training X.
#' @param scale "response" returns binary probabilities and continuous means;
#'   "latent" returns latent linear predictors for every outcome.
#' @param summary "draws", "mean", or "quantiles".
#' @param probs Quantiles when summary="quantiles".
#' @param ... Unused.
#' @export
predict.bb_mixed_mssl <- function(object,
                                  newdata,
                                  scale = c("response", "latent"),
                                  summary = c("draws", "mean", "quantiles"),
                                  probs = c(0.025, 0.5, 0.975),
                                  ...) {
  scale <- match.arg(scale)
  summary <- match.arg(summary)
  newdata <- as.matrix(newdata)
  storage.mode(newdata) <- "double"
  if (ncol(newdata) != dim(object$draws$B)[1L] ||
      any(!is.finite(newdata))) {
    stop("newdata must be finite with ncol(newdata) equal to nrow(B).",
         call. = FALSE)
  }

  n_new <- nrow(newdata)
  q <- dim(object$draws$B)[2L]
  n_draws <- dim(object$draws$B)[3L]
  response_names <- dimnames(object$draws$B)[[2L]]
  out <- array(
    NA_real_, c(n_new, q, n_draws),
    dimnames = list(NULL, response_names, NULL)
  )

  for (r in seq_len(n_draws)) {
    if (!object$diagnostics$success[r]) next
    eta <- sweep(
      newdata %*% object$draws$B[, , r],
      2,
      object$draws$alpha[, r],
      `+`
    )
    if (scale == "response") {
      binary <- which(object$response_types == "binary")
      if (length(binary) > 0L) {
        latent_sd <- sqrt(diag(object$draws$Sigma[, , r]))
        eta[, binary] <- stats::pnorm(
          sweep(eta[, binary, drop = FALSE], 2, latent_sd[binary], `/`)
        )
      }
    }
    out[, , r] <- eta
  }

  if (summary == "draws") return(out)
  if (summary == "mean") {
    return(apply(out, c(1, 2), mean, na.rm = TRUE))
  }
  .bb_mixed_mssl_quantiles(out, probs)
}
