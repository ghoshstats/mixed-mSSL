# mixed-mSSL

`mixed-mSSL` provides a simple wrapper, `mixed_mssl()`, around the **mSSL** C++/R implementation to fit a mixed‐outcome multivariate spike‐and‐slab LASSO model.  It automatically handles mixed binary and continuous responses.


## Usage

```r
library(Rcpp)       # required by mSSL C++ code
library(mixed.mSSL) # your package

set.seed(12)
p <- 2; q <- 5; n <- 200

# 30% of entries nonzero, values ~ Unif(-1,1)
B <- as.matrix(
  rsparsematrix(p, q, 0.3,
    rand.x = function(n) runif(n, -1, 1)
  )
)

# Design matrix
X <- matrix(rnorm(n * p), nrow = n, ncol = p)

# Response covariance & intercept
graph <- g_model1(q)
Sigma <- graph$Sigma
Omega <- graph$Omega
mu    <- runif(q, -1, 1)

# Generate latent and observed Y
set.seed(42)
Y_latent <- X %*% B + rep(1, n) %*% t(mu) + mvrnorm(n, rep(0, q), Sigma)
Y <- Y_latent
# Threshold first two columns to binary
Y[, 1:2] <- 1 * (Y_latent[, 1:2] >= 0)

# Fit mixed‐mSSL
mixed_model <- mixed_mssl(
  X, Y,
  response_types      = c("binary","binary","continuous","continuous","continuous"),
  lambdas             = list(lambda1 = 1, lambda0 = 50),
  xis                 = list(xi1     = 0.01 * n,
                               xi0     = seq(0.1*n, n, length = 10)),
  theta_hyper_params  = c(1, ncol(X) * ncol(Y)),
  eta_hyper_params    = c(1, ncol(Y))
)

# Extract estimates
B_est     <- mixed_model$B
Omega_est <- mixed_model$Omega

# Evaluate performance on B and Omega
perf_B     <- error_B(B_est, B)
perf_Omega <- error_Omega(Omega_est, Omega)
print(perf_B)
print(perf_Omega)
```

You may adjust the penalty grid and hyperparameters to tune sensitivity vs. specificity/precision as needed. 

Original mSSL source is available at [https://github.com/YunyiShen/mSSL](https://github.com/YunyiShen/mSSL).
