library(MASS)   
library(pROC)

true_EY <- function(Xnew, B0, Omega0, q_bin) {
  mu_lat <- Xnew %*% B0                       
  sd_lat <- sqrt(diag(solve(Omega0)))[seq_len(q_bin)]
  prob   <- pnorm( sweep(mu_lat[,seq_len(q_bin)], 2, sd_lat, "/") )
  cbind(prob, mu_lat[ , -(seq_len(q_bin)), drop = FALSE])
}

##  Regression-function error  
score_regfn <- function(X_test, EY_true,
                        pred_mSSL,           
                        pred_mtMBSP,         
                        pred_sepSSL,
                        pred_sepglmnet)         
{
  L2 <- function(A,B) sqrt(rowSums((A-B)^2))
  c(
    RFE_mSSL   = mean(L2(pred_mSSL  , EY_true)),
    RFE_mtMBSP = mean(L2(pred_mtMBSP, EY_true)),
    RFE_sepSSL = mean(L2(pred_sepSSL, EY_true)),
    RFE_sepglm = mean(L2(pred_sepglmnet, EY_true))
  )
}

score_predict <- function(X_test, Y_test,
                          q_bin,
                          draw_mSSL,         
                          draw_mtMBSP,       
                          draw_sepSSL,
                          draw_sepglmnet,
                          M = 100)          
{
  rmse <- function(Yhat,Ytrue, idx) {
    sqrt(mean((Yhat[,idx,drop=FALSE] - Ytrue[,idx,drop=FALSE])^2))
  }
  auc_bin <- function(Yprob,Ytrue, idx) {
    mean( sapply(seq_along(idx), function(j){
      as.numeric(roc(Ytrue[,idx[j]], Yprob[,j], quiet=TRUE)$auc)
    }) )
  }
  bin_idx  <- seq_len(q_bin)
  cont_idx <- (q_bin+1):ncol(Y_test)
  
  Yhat_mSSL <- draw_mSSL()$Yrep
  rmse_mSSL <- rmse(Yhat_mSSL, Y_test, cont_idx)
  auc_mSSL  <- auc_bin(Yhat_mSSL[,bin_idx], Y_test, bin_idx)
  
  rmse_mt  <- auc_mt <- 0
  for(m in 1:M){
    Ym        <- draw_mtMBSP()$Yrep
    rmse_mt   <- rmse_mt + rmse(Ym, Y_test, cont_idx)
    auc_mt    <- auc_mt  + auc_bin(Ym[,bin_idx], Y_test, bin_idx)
  }
  rmse_mt  <- rmse_mt / M
  auc_mt   <- auc_mt  / M
  
  Yhat_sep  <- draw_sepSSL()$Yrep
  rmse_sep  <- rmse(Yhat_sep, Y_test, cont_idx)
  auc_sep   <- auc_bin(Yhat_sep[,bin_idx], Y_test, bin_idx)
  
  Yhat_sepglmnet  <- draw_sepglmnet()$Yrep
  rmse_sepglmnet  <- rmse(Yhat_sepglmnet, Y_test, cont_idx)
  auc_sepglmnet   <- auc_bin(Yhat_sepglmnet[,bin_idx], Y_test, bin_idx)
  
  c(RMSE_mSSL   = rmse_mSSL,  AUC_mSSL   = auc_mSSL,
    RMSE_mtMBSP = rmse_mt ,   AUC_mtMBSP = auc_mt ,
    RMSE_sepSSL = rmse_sep,   AUC_sepSSL = auc_sep,
    RMSE_sepglm = rmse_sepglmnet, AUC_sepglm = auc_sepglmnet)
}


## mixed-mSSL wrapper
draw_mSSL_wrapper <- function(Bhat, Omegahat, q_bin, X_test){
  n_q   <- ncol(Bhat)
  sdLat <- sqrt(diag(solve(Omegahat)))[seq_len(q_bin)]
  function(){
    Z <- X_test %*% Bhat + mvrnorm(nrow(X_test), rep(0,n_q),
                                   Sigma = solve(Omegahat))
    Y <- Z
    Y[,seq_len(q_bin)] <- 1*(Z[,seq_len(q_bin)] > 0)
    list(Yrep = Y)
  }
}

## mt-MBSP wrapper
draw_mtMBSP_wrapper <- function(draw_list, q_bin, X_test)
{
  n_q <- ncol(draw_list[[1]]$B)         
  n_test <- nrow(X_test)                 
  
  function()
  {
    d     <- sample(draw_list, 1)[[1]]   
    Sigma <- solve(d$Omega)              
    Z_mean <- X_test %*% d$B             
    Z      <- Z_mean + MASS::mvrnorm(n_test,
                                     mu    = rep(0, n_q),
                                     Sigma = Sigma)
    
    Y_rep        <- Z
    Y_rep[, seq_len(q_bin)] <- 1 * (Z[, seq_len(q_bin)] > 0)  
    list(Yrep = Y_rep)
  }
}


## sep-SSL wrapper
draw_sepSSL_wrapper <- function(beta_list, sigma2_list, q_bin, X_test){
  function(){
    Y <- matrix(NA, nrow(X_test), length(beta_list))
    for(k in seq_along(beta_list)){
      eta <- as.numeric(X_test %*% beta_list[[k]])
      if(k<=q_bin){
        Y[,k] <- rbinom(nrow(X_test), 1, pnorm(eta))
      }else{
        Y[,k] <- rnorm(nrow(X_test), eta, sqrt(sigma2_list[[k]]))
      }
    }
    list(Yrep = Y)
  }
}

## sep-GLMnet wrapper
draw_sepGLMnet_wrapper <- function(beta_list, sigma2_list, q_bin, X_test) {
  Xmat <- as.matrix(X_test)                           
  function() {
    n <- nrow(Xmat);  q <- length(beta_list)
    Yrep <- matrix(NA_real_, n, q)
    
    for (k in seq_len(q)) {
      beta <- beta_list[[k]]         
      eta  <- Xmat%*% beta 
      
      if (k <= q_bin) {               
        p        <- plogis(eta)       
        Yrep[,k] <- rbinom(n, 1, p)
      } else {                        
        sd       <- sqrt(sigma2_list[[k]])
        Yrep[,k] <- rnorm(n, eta, sd)
      }
    }
    list(Yrep = Yrep)
  }
}
################################################################################
# EY_true <- true_EY(X_test_sc, B, Omega, q_bin)
# B_est <- mpcSSL_dpe_res[["B"]]
# Omega_est <- mpcSSL_dpe_res[["Omega"]]
# draw_mSSL    <- draw_mSSL_wrapper (B_est, Omega_est, q_bin,X_test_sc)
# ################################################################################
# nDraws <- length(output$B_samples)
# mtMBSP_draws <- lapply(seq_len(nDraws), function(i){
#   list(
#     B     = output$B_samples[[i]],
#     Omega = solve(output$Sigma_samples[[i]])   # precision matrix
#   )
# })
# draw_mtMBSP  <- draw_mtMBSP_wrapper(mtMBSP_draws, q_bin,X_test_sc)
# ################################################################################
# fit_sepSSL <- sepSSL(X_sc,Y_sc,type,lambda1=0.04,lambda0=0.5)
# beta_sep <- asplit(fit_sepSSL$B_est,MARGIN=2)
# sigma_sep <- fit_sepSSL$sigma2_list
# draw_sepSSL  <- draw_sepSSL_wrapper(beta_sep, sigma_sep, q_bin,X_test_sc)
# ################################################################################
# fit_sepglmnet <- sepGLMnet(X_sc,Y_sc,type)
# beta_sepglmnet <- asplit(fit_sepglmnet$B_est,MARGIN=2)
# sigma_sepglmnet <- fit_sepglmnet$sigma2_list
# draw_sepglmnet  <- draw_sepGLMnet_wrapper(beta_sepglmnet, sigma_sepglmnet, q_bin,X_test_sc)
# ################################################################################
# 
# pred_mSSL    <- draw_mSSL()$Yrep           
# pred_mtMBSP  <- Reduce("+", lapply(mtMBSP_draws, function(d){
#   true_EY(X_test_sc, d$B, d$Omega, q_bin)}))/length(mtMBSP_draws)
# pred_sepSSL  <- true_EY(X_test_sc, do.call(cbind, beta_sep), Omega, q_bin)  
# pred_sepglmnet  <- true_EY(X_test_sc, do.call(cbind, beta_sepglmnet), Omega, q_bin)  
# 
# reg_scores <- score_regfn(X_test_sc, EY_true,
#                           pred_mSSL, pred_mtMBSP, pred_sepSSL,pred_sepglmnet)
# 
# pred_scores <- score_predict(X_test_sc, Y_test_sc,
#                              q_bin,
#                              draw_mSSL, draw_mtMBSP, draw_sepSSL, draw_sepglmnet, M = 1000)
