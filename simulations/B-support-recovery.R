library(Matrix)       
library(MASS)         
library(mvtnorm)      
library(purrr)        
library(dplyr)        
library(MtMBSP)       
source("error_metrics.R")   
source("simulation_settings.R") 
source("other_methods.R")
source("mixed-mssl.R")

uniform_signal <- function(p, q, frac = 0.3) {
  rsparsematrix(p, q, frac, rand.x = function(n) runif(n, -5, 5))
}
disjoint_signal <- function(p, q, frac = 0.3) {
  rsparsematrix(p, q, frac,
                rand.x = function(n) sample_disjoint(n, -5, -2, 2, 5)
  )
}

#HYPER‐PARAMETER LOOKUP
get_hyper_params <- function(n, p, q, signal_name) {
  theta_hyp <- c(1, p * q)
  eta_hyp   <- c(1, q)
  
  if (identical(c(n,p,q), c(200,500,4))) {
    if (signal_name == "uniform") {
      lam1 <- 0.13; lam0 <- 40
      xi1  <- 0.13; xi0  <- 40
    } else {
      lam1 <- 0.07; lam0 <- 23
      xi1  <- 0.07; xi0  <- 23
    }
  } else if (identical(c(n,p,q), c(500,1000,4))) {
    if (signal_name == "uniform") {
      lam1 <- 0.09; lam0 <- 70
      xi1  <- 0.09; xi0  <- 70
    } else {
      lam1 <- 0.09; lam0 <- 63
      xi1  <- 0.09; xi0  <- 63
    }
  } else if (identical(c(n,p,q), c(800,1000,6))) {
    lam1 <- 0.001; lam0 <- 80
    xi1  <- 8; xi0  <- 80
  } else {
    stop("No hyperparams for scenario (", n, ",", p, ",", q, ")")
  }
  
  list(
    lambdas     = list(lambda1 = lam1, lambda0 = lam0),
    xis         = list(xi1     = xi1,     xi0 = xi0),
    theta_hyper = theta_hyp,
    eta_hyper   = eta_hyp
  )
}

# One iteration
run_one_iter <- function(n, p, q, graph_fn, signal_fn, response_types,
                         lambdas, xis, theta_hyp, eta_hyp) {
  B_true <- as.matrix(signal_fn(p, q))
  X      <- simulate_X(n, p, rho = 0.5, sigma2 = 1)
  graph  <- graph_fn(q)
  Sigma  <- graph$Sigma
  mu     <- runif(q, -1, 1)
  
  Y_lat  <- X %*% B_true + rep(1, n) %*% t(mu) +
    mvrnorm(n, rep(0, q), Sigma)
  Y      <- Y_lat
  bin_idx <- which(response_types == "binary")
  Y[, bin_idx] <- 1L * (Y_lat[, bin_idx] >= 0)
  
  X_sc <- scale(X)
  Y_sc <- Y
  cont_idx <- which(response_types == "continuous")
  if (length(cont_idx))
    Y_sc[, cont_idx] <- scale(Y_sc[, cont_idx])
  
  #  Fit mixed-mSSL
  t0 <- Sys.time()
  out_mssl <- mixed_mssl(
    X_sc, Y_sc, response_types,
    lambdas, xis,
    theta_hyp, eta_hyp,
    verbose = 0
  )
  t1 <- Sys.time()
  perf_mssl <- error_B(out_mssl$B, B_true) %>%
    as.list() %>%
    append(list(time = as.numeric(t1 - t0)))
  
  # Fit Mt-MBSP
  t0 <- Sys.time()
  out_mbsp <- Mt_MBSP(X, Y, response_types)
  t1 <- Sys.time()
  B_mbsp   <- out_mbsp[["B_est"]]
  cls      <- out_mbsp[["B_active"]]
  TP <- sum(cls == 1 & B_true != 0)
  TN <- sum(cls == 0 & B_true == 0)
  FP <- sum(cls == 1 & B_true == 0)
  FN <- sum(cls == 0 & B_true != 0)
  sens <- TP / (TP + FN)
  spec <- TN / (TN + FP)
  prec <- TP / (TP + FP)
  acc  <- (TP + TN) / (TP + TN + FP + FN)
  F1   <- 2 * TP / (2*TP + FP + FN)
  mcc  <- (TP*TN - FP*FN) / sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
  rmse_mbsp <- sqrt(mean((B_mbsp - B_true)^2))
  perf_mbsp <- list(
    TP   = TP,
    TN   = TN,
    FP   = FP,
    FN   = FN,
    SEN  = sens,
    SPE  = spec,
    PREC = prec,
    ACC  = acc,
    F1   = F1,
    MCC  = mcc,
    RMSE = rmse_mbsp,
    time = as.numeric(t1 - t0)
  )
  
  # Fit sep-SSL
  t0 <- Sys.time()
  out_sepssl <- suppressWarnings(sepSSL(X, Y_sc, type = response_types,
                       lambda1 = 0.04, lambda0 = 0.5))
  t1 <- Sys.time()
  perf_sepssl <- error_B(out_sepssl$B_est, B_true) %>%
    as.list() %>%
    append(list(time = as.numeric(t1 - t0)))
  
  #  Fit sep-GLM
  t0 <- Sys.time()
  out_sepglm <- sepGLMnet(as.matrix(X_sc), as.matrix(Y_sc),
                          type = response_types)
  t1 <- Sys.time()
  perf_sepglm <- error_B(out_sepglm$B_est, B_true) %>%
    as.list() %>%
    append(list(time = as.numeric(t1 - t0)))
  
  bind_rows(
    mSSL   = perf_mssl,
    MtMBSP = perf_mbsp,
    sepSSL = perf_sepssl,
    sepGLM = perf_sepglm,
    .id = "method"
  )
}

run_scenario <- function(n, p, q, graph_name, signal_name, iter = 100) {
  graph_fn <- switch(graph_name,
                     AR1 = g_model1, AR2 = g_model2,
                     BD  = g_model3, SG  = g_model4,
                     SW  = g_model5, TN  = g_model6
  )
  signal_fn      <- if (signal_name == "uniform") uniform_signal
  else disjoint_signal
  response_types <- c(rep("binary", q/2), rep("continuous", q/2))
  
  hyp <- get_hyper_params(n, p, q, signal_name)
  
  map_dfr(1:iter, function(i) {
    run_one_iter(n, p, q, graph_fn, signal_fn, response_types,
                 hyp$lambdas, hyp$xis,
                 hyp$theta_hyper, hyp$eta_hyper) %>%
      mutate(iter = i)
  })
}

all_scenarios <- expand.grid(
  n           = c(200, 500, 800),
  p           = c(500,1000,1000),
  q           = c(4,   4,   6),
  graph_name  = c("AR1","AR2","BD","SG","SW","TN"),
  signal_name = c("uniform","disjoint"),
  stringsAsFactors = FALSE
)

results <- pmap_dfr(all_scenarios, 
                    function(n, p, q, graph_name, signal_name) {
                      run_scenario(n, p, q, graph_name, signal_name, iter = 100) %>%
                        mutate(n = n, p = p, q = q,
                               graph  = graph_name,
                               signal = signal_name)
                    }
)

summary_tbl <- results %>%
  group_by(n,p,q,graph,signal,method) %>%
  summarise(
    SEN  = mean(SEN),
    SPE  = mean(SPE),
    PREC = mean(PREC),
    ACC  = mean(ACC),
    RMSE = mean(RMSE),
    time = mean(time),
    .groups = "drop"
  )

print(summary_tbl)

#' Example:: To replicate the results of the AR1 setting with Uniform[-5,5] signals and (n,p,q)=(200,500,4), run the following:
#'
# run_scenario(n=200,p=500,q=4,graph_name = "AR1",signal_name = "uniform",iter=100)

