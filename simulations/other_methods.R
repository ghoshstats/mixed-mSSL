########################## Separately modelling mixed type outcomes ##############################
##################################################################################################
######################## separate SSLASSOs ###############################
library(BhGLM)
library(SSLASSO)
library(glmnet)

sepSSL <- function(X, Y, type,
                   center = TRUE, scale = TRUE, lambda1, lambda0) {
  n  <- nrow(X); p <- ncol(X); q <- ncol(Y)
  if (length(type) != q)
    stop("'type' must have length = ncol(Y).")
  type <- match.arg(type,
                    choices = c("continuous", "binary"),
                    several.ok = TRUE)
  
  Xc <- scale(X, center = center, scale = scale)
  sigma2_list  <- list()
  B_est    <- matrix(0, p, q)
  for (k in seq_len(q)) {
    yk <- Y[,k]
    if (type[k] == "continuous") {
      fit <- SSLASSO::SSLASSO(X=Xc,y=yk,lambda1 = lambda1, lambda0 = lambda0)
      b_hat <- fit[["beta"]]  
      sigma2_list[[k]] <- fit[["sigmas"]]
    } else {            
      fit <- BhGLM::bmlasso(x=Xc,y=yk,family="binomial",ss=c(lambda1,lambda0))
      b_hat <- as.numeric(fit[["coefficients"]][-1])                   
    }
    B_est[, k]    <- b_hat
  }
  colnames(B_est)    <- colnames(Y)
  rownames(B_est)    <- colnames(X)
  return(list(B_est = B_est, sigma2_list = sigma2_list))
}

######################### separate Glmnets #############################
sepGLMnet <- function(X, Y, types, alpha = 1L) {
  n <- nrow(X)
  q <- ncol(Y)
  B_est <- matrix(0, nrow = ncol(X), ncol = q)
  sigma2_list <- vector("list", q)      
  for (k in seq_len(q)) {
    yk <- Y[,k]
    fam <- types[k]
    
    if (fam == "continuous") {
      cv_model <- cv.glmnet(X, yk, alpha=1)
      best_lambda <- cv_model$lambda.min
      fitk <- glmnet::glmnet(x = X, y = yk, family = "gaussian", alpha = alpha,lambda=best_lambda)
      betahat <- as.numeric(coef(fitk))[-1]
      B_est[, k] <- betahat
      y_hat             <- X %*% betahat
      sigma2_list[[k]]  <- mean((yk - y_hat)^2)
    } else if (fam == "binary") {
      cv_model <- cv.glmnet(X, yk, alpha=1)
      best_lambda <- cv_model$lambda.min
      fitk <- glmnet::glmnet(x = X, y = yk,  family=binomial(link = "probit"), alpha = alpha,lambda=best_lambda)
      betahat <- as.numeric(coef(fitk))[-1]
      B_est[, k] <- betahat
    } else {
      stop(sprintf("Unknown family '%s' for column %d; use either 'continuous' or 'binary'.",
                   fam, k))
    }
    
  }
  return(list(B_est = B_est, sigma2_list = sigma2_list))
}
