######################## Simulation settings (mixed-type) ############################
######################################################################################
library(mvtnorm)
library(MASS)
library(Matrix)
################# Generate X #####################
simulate_X = function(n, p, rho=0.5, sigma2=1){
  times <- 1:p
  H <- abs(outer(times, times, "-"))
  U <- sigma2*rho^H
  mu <- matrix(0, p)
  X <- mvtnorm::rmvnorm(n, mu, U)
  return(X)
}

################ Covariance structures #####################
### (1) AR(1) ###
g_model1 <- function(k, rho=.7, tol = 1e-10){
  temp <- matrix(rep(1:k,k),ncol = k)
  Sigma <- rho ^ (abs(temp-t(temp)))
  Omega <- solve(Sigma)
  Omega <- Omega * (abs(Omega)>tol)
  return(list(Sigma = Sigma, Omega = Omega))
}

### (2) AR(2) ###
g_model2 <- function(k, rho = .5){
  temp <- matrix(rep(1:k,k),ncol = k)
  Omega <- rho ^ (abs(temp-t(temp))) * (abs(temp-t(temp)) <= 2)
  return(list(Omega = Omega, Sigma = solve(Omega)))
}

### (3) BD ###
g_model3 <- function(k, rho = .5){
  row_ind <- matrix(rep(1:k,k),ncol = k)
  col_ind <- t(row_ind)
  
  Sigma <- diag(1,k,k)
  
  Sigma[(1<=row_ind ) &
          (1<=col_ind ) &
          (row_ind != col_ind) & 
          (row_ind <= k/2) & 
          (col_ind <= k/2)] <- rho
  Sigma[((1+k/2)<=row_ind ) & 
          ((1+k/2)<=col_ind ) &
          (row_ind != col_ind)] <- rho # & 
  #(row_ind <= 10) & 
  #(col_ind <= 10)] 
  Omega <- solve(Sigma)
  
  return(list(Sigma = Sigma, Omega = Omega))
}
### (4) SG ###
g_model4 <- function(k, rho=.1){
  
  Omega <- diag(1,k,k)
  Omega[2:k,1] <- rho
  Omega[1,2:k] <- rho
  Sigma <- solve(Omega)
  return(list(Sigma = Sigma, Omega = Omega))
}
### (5) SW ###
g_model5 <- function(k, rhos = c(2,1,.9)){
  row_ind <- matrix(rep(1:k,k),ncol = k)
  col_ind <- t(row_ind)
  Omega <- diag(rhos[1],k,k)
  Omega[abs(row_ind-col_ind)==1] <- rhos[2]
  Omega[1,k] <- Omega[k,1] <- rhos[3]
  return(list(Sigma = solve(Omega), Omega = Omega))
}
### (6) TN ###
g_model6 <- function(k, rhos = c(2,1)){
  Omega <- matrix(rhos[2], k, k)
  diag(Omega) <- rhos[1]
  
  return(list(Sigma = solve(Omega),Omega = Omega))
}

##################### Signal settings #########################
###############################################################
sample_disjoint <- function(n, a,b,c,d){
  sample_vec <- rep(0,n)
    for(ntimes in 1:n){
    y <- stats::runif(1, 0, b-a+d-c)
    if( y < (b-a) ){
      x <- a + y
    }else{
      x <- c + y - (b-a)
    }
    sample_vec[ntimes] <- x
  }
  return(sample_vec)
}

sample_uniform <- function(n,a,b){
  sample_vec <- stats::runif(n,a,b)
  return(sample_vec)
}
