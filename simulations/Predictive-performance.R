
library(Matrix)      
library(MASS)         
library(mvtnorm)      
library(purrr)        
library(dplyr)        

source("other_methods.R")       
source("error_metrics.R")        
source("simulation_settings.R")  
source("Prediction_analysis.R")  
source("mixed-mssl.R")           

uniform_signal <- function(p, q, frac = 0.3) {
  rsparsematrix(p, q, frac, rand.x = function(n) runif(n, -5, 5))
}
disjoint_signal <- function(p, q, frac = 0.3) {
  rsparsematrix(p, q, frac,
                rand.x = function(n) sample_disjoint(n, -5, -2, 2, 5)
  )
}

# ONE ITERATION of Prediction 
run_one_iter_pred <- function(n, p, q, graph_fn, signal_fn, response_types) {
  B_true <- as.matrix(signal_fn(p, q))
  X      <- simulate_X(n, p, rho = 0.5, sigma2 = 1)
  graph  <- graph_fn(q)
  Sigma  <- graph$Sigma
  Omega  <- graph$Omega
  mu     <- runif(q, -1, 1)
  
  Y_lat  <- X %*% B_true + rep(1, n) %*% t(mu) + mvrnorm(n, rep(0, q), Sigma)
  Y      <- Y_lat
  bin_idx <- which(response_types == "binary")
  Y[, bin_idx] <- 1L * (Y_lat[, bin_idx] >= 0)
  
  X_sc     <- scale(X)
  Y_sc     <- Y
  cont_idx <- which(response_types == "continuous")
  if (length(cont_idx)) Y_sc[, cont_idx] <- scale(Y_sc[, cont_idx])
  
  n_test   <- n %/% 2
  X_test   <- simulate_X(n_test, p, rho = 0.5, sigma2 = 1)
  Y_test   <- X_test %*% B_true +
    matrix(rep(mu, each = n_test), n_test, q) +
    mvrnorm(n_test, rep(0, q), Sigma)
  Y_test[, bin_idx] <- 1L * (Y_test[, bin_idx] >= 0)
  X_test_sc <- scale(X_test)
  Y_test_sc <- Y_test
  if (length(cont_idx)) Y_test_sc[, cont_idx] <- scale(Y_test_sc[, cont_idx])
  
  q_bin    <- length(bin_idx)
  EY_true  <- true_EY(X_test_sc, B_true, Omega, q_bin)
  
  lambdas  <- list(lambda1 = 0.04, lambda0 = 0.5)
  xis      <- list(xi1 = 0.01 * n, xi0 = seq(0.1 * n, n, length = 10))
  theta_hyp <- c(1, p * q)
  eta_hyp   <- c(1, q)
  
  L2 <- function(A,B) sqrt(rowSums((A-B)^2))  
  # 1) mixed-mSSL
  out1 <- mixed_mssl(X_sc, Y_sc, response_types,
                     lambdas, xis,
                     theta_hyp, eta_hyp,
                     verbose = 0)
  Yrep1 <- draw_mSSL_wrapper(out1$B, out1$Omega, q_bin, X_test_sc)()$Yrep
  reg1  <- mean(L2(Yrep1, EY_true))
  rmse1 <- sqrt(mean((Yrep1[, cont_idx] - Y_test_sc[, cont_idx])^2))
  auc1  <- mean(sapply(bin_idx, function(j) {
    pROC::roc(Y_test_sc[, j], Yrep1[, j], quiet = TRUE)$auc
  }))
  
  # 2) Mt-MBSP
  out2 <- Mt_MBSP(X_sc, Y_sc, response_types)
  draws2 <- lapply(seq_along(out2$B_samples), function(i) list(
    B     = out2$B_samples[[i]],
    Omega = solve(out2$Sigma_samples[[i]])
  ))
  pred2 <- Reduce("+", lapply(draws2, function(d) 
    true_EY(X_test_sc, d$B, d$Omega, q_bin))) / length(draws2)
  reg2  <- mean(L2(pred2, EY_true))
  rmse2 <- sqrt(mean((pred2[, cont_idx] - Y_test_sc[, cont_idx])^2))
  auc2  <- mean(sapply(bin_idx, function(j) {
    pROC::roc(Y_test_sc[, j], pred2[, j], quiet = TRUE)$auc
  }))
  
  # 3) sep-SSL
  f3   <- suppressWarnings(sepSSL(X_sc, Y_sc, type = response_types,
                 lambda1 = 0.04, lambda0 = 10))
  B3   <- do.call(cbind, asplit(f3$B_est, 2))
  pred3 <- true_EY(X_test_sc, B3, Omega, q_bin)
  reg3  <- mean(L2(pred3, EY_true))
  rmse3 <- sqrt(mean((pred3[, cont_idx] - Y_test_sc[, cont_idx])^2))
  auc3  <- mean(sapply(bin_idx, function(j) {
    pROC::roc(Y_test_sc[, j], pred3[, j], quiet = TRUE)$auc
  }))
  
  # 4) sep-GLM
  f4   <- sepGLMnet(X_sc, Y_sc, type = response_types)
  B4   <- do.call(cbind, asplit(f4$B_est, 2))
  pred4 <- true_EY(X_test_sc, B4, Omega, q_bin)
  reg4  <- mean(L2(pred4, EY_true))
  rmse4 <- sqrt(mean((pred4[, cont_idx] - Y_test_sc[, cont_idx])^2))
  auc4  <- mean(sapply(bin_idx, function(j) {
    pROC::roc(Y_test_sc[, j], pred4[, j], quiet = TRUE)$auc
  }))
  
  tibble(
    RFE_mSSL   = reg1, RMSE_mSSL   = rmse1, AUC_mSSL   = auc1,
    RFE_mtMBSP = reg2, RMSE_mtMBSP = rmse2, AUC_mtMBSP = auc2,
    RFE_sepSSL = reg3, RMSE_sepSSL = rmse3, AUC_sepSSL = auc3,
    RFE_sepGLM = reg4, RMSE_sepGLM = rmse4, AUC_sepGLM = auc4
  )
}


valid_combos <- tibble(n=c(200,500,800),
                       p=c(500,1000,1000),
                       q=c(4,  4,   6))

all_scenarios <- valid_combos %>%
  tidyr::crossing(
    graph_name  = c("AR1","AR2","BD","SG","SW","TN"),
    signal_name = c("uniform","disjoint")
  )

results_pred <- purrr::pmap_dfr(
  all_scenarios,
  function(n,p,q,graph_name,signal_name) {
    run_scenario_pred <- function(i){
      set.seed(i)
      run_one_iter_pred(
        n,p,q,
        switch(graph_name,
               AR1=g_model1, AR2=g_model2,
               BD=g_model3, SG=g_model4,
               SW=g_model5, TN=g_model6
        ),
        if(signal_name=="uniform") uniform_signal else disjoint_signal,
        c(rep("binary", q/2), rep("continuous", q/2))
      )
    }
    bind_rows(
      map(1:5, run_scenario_pred),    # 5 iterations
      .id="iter"
    ) %>%
      mutate(n=n,p=p,q=q,
             graph=graph_name, signal=signal_name)
  }
)

summary_pred <- results_pred %>%
  group_by(n,p,q,graph,signal) %>%
  summarise(across(starts_with(c("RFE","RMSE","AUC")),
                   mean, .names="{.col}"), .groups="drop")

print(summary_pred)
