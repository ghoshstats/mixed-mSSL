library(Matrix)       
library(MASS)         
library(mvtnorm)      
library(purrr)        
library(dplyr)        
source("mixed-mssl.R")  
source("error_metrics.R")      
source("simulation_settings.R")

uniform_signal <- function(p, q, frac = 0.3) {
  rsparsematrix(p, q, frac, rand.x = function(n) runif(n, -5, 5))
}
disjoint_signal <- function(p, q, frac = 0.3) {
  rsparsematrix(p, q, frac,
                rand.x = function(n) sample_disjoint(n, -5, -2, 2, 5)
  )
}

get_hyper_params <- function(n, p, q, signal_name) {
  theta_hyp <- c(1, p * q)
  eta_hyp   <- c(1, q)
  
  if (identical(c(n,p,q), c(200,500,4))) {
    if (signal_name=="uniform") {
      lam1 <- 1; lam0 <- 50
      xi1  <- 0.0001; xi0  <- 15
    } else {
      lam1 <- 1; lam0 <- 50
      xi1  <- 0.0001; xi0  <- 15
    }
  } else if (identical(c(n,p,q), c(500,1000,4))) {
    if (signal_name=="uniform") {
      lam1 <- 1; lam0 <- 80
      xi1  <- 0.0001; xi0  <- 15
    } else {
      lam1 <- 1; lam0 <- 80
      xi1  <-  0.0001; xi0  <- 15
    }
  } else if (identical(c(n,p,q), c(800,1000,6))) {
    lam1 <- 0.001;    lam0 <- 100
    xi1  <-  0.001; xi0 <- 15
  } else {
    stop("No hyperparams for (",n,",",p,",",q,")")
  }
  
  list(
    lambdas     = list(lambda1 = lam1, lambda0 = lam0),
    xis         = list(xi1     = xi1,     xi0 = xi0),
    theta_hyper = theta_hyp,
    eta_hyper   = eta_hyp
  )
}

# one iteration
run_one_iter_omega <- function(n, p, q, graph_fn, signal_fn, response_types,
                               lambdas, xis, theta_hyp, eta_hyp) {
  B_true <- as.matrix(signal_fn(p, q))
  X      <- simulate_X(n, p, rho = 0.5, sigma2 = 1)
  graph  <- graph_fn(q)
  Sigma  <- graph$Sigma
  Omega <- graph$Omega
  mu     <- runif(q, -1, 1)
  
  Y_lat  <- X %*% B_true + rep(1, n) %*% t(mu) +
    mvrnorm(n, rep(0, q), Sigma)
  Y      <- Y_lat
  bin_idx <- which(response_types=="binary")
  Y[,bin_idx] <- 1L * (Y_lat[,bin_idx] >= 0)
  
  X_sc    <- scale(X)
  Y_sc    <- Y
  cont_idx<- which(response_types=="continuous")
  if (length(cont_idx))
    Y_sc[,cont_idx] <- scale(Y_sc[,cont_idx])
  
  out <- mixed_mssl(
    X_sc, Y_sc, response_types,
    lambdas, xis,
    theta_hyp, eta_hyp,
    verbose = 0
  )
  Omega_est <- out$Omega
  
  
  perf_omega <- error_Omega(Omega_est, Omega) 
  
  as.list(perf_omega)
}

run_scenario_omega <- function(n, p, q, graph_name, signal_name, iter = 100) {
  graph_fn      <- switch(graph_name,
                          AR1 = g_model1, AR2 = g_model2,
                          BD  = g_model3, SG  = g_model4,
                          SW  = g_model5, TN  = g_model6
  )
  signal_fn     <- if (signal_name == "uniform") uniform_signal else disjoint_signal
  response_types<- c(rep("binary", q/2), rep("continuous", q/2))
  
  hyp <- get_hyper_params(n, p, q, signal_name)
  
  map_dfr(1:iter, function(i) {
    set.seed(i)
    metrics_list <- run_one_iter_omega(
      n, p, q, graph_fn, signal_fn, response_types,
      hyp$lambdas, hyp$xis,
      hyp$theta_hyper, hyp$eta_hyper
    )
    tibble::as_tibble_row(metrics_list) %>%
      mutate(iter = i)
  })
}
valid_combos <- tibble::tibble(
  n = c(200, 500, 800),
  p = c(500, 1000, 1000),
  q = c(4,   4,    6)
)

all_scenarios <- valid_combos %>%
  tidyr::crossing(
    graph_name  = c("AR1","AR2","BD","SG","SW","TN"),
    signal_name = c("uniform","disjoint")
  )

results_omega <- purrr::pmap_dfr(
  all_scenarios,
  function(n, p, q, graph_name, signal_name) {
    run_scenario_omega(n, p, q, graph_name, signal_name, iter = 100) %>%
      dplyr::mutate(
        n      = n,
        p      = p,
        q      = q,
        graph  = graph_name,
        signal = signal_name
      )
  }
)

summary_omega <- results_omega %>%
  group_by(n, p, q, graph, signal) %>%
  summarise(across(c(TP, TN, FP, FN, SEN, SPE, PREC, ACC, F1, MCC, FROB),
                   mean, .names = "{.col}"),
            .groups = "drop")

print(summary_omega)

#' For example, run
#'
# run_scenario_omega(n=200,p=500,q=4,graph_name="AR1",signal_name="uniform",iter=2)

