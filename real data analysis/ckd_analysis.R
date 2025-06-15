################################# CKD analysis ################################
source("other_methods.R")
cleaned_CKD <- read.csv("~/updated_chronic_kidney_disease.csv")
cleaned_CKD$class <- ifelse(cleaned_CKD$class == "ckd", 1, 0)
Y <- as.matrix(cbind(cleaned_CKD$class, cleaned_CKD$sg))
continuous_covariates <- c("bp", "al", "su", "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wbcc", "rbcc")

min_max_scaling <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}

cleaned_CKD[continuous_covariates] <- lapply(cleaned_CKD[continuous_covariates], min_max_scaling)

X <- as.matrix(cleaned_CKD[,-c(1,4,26)])

###### mixed-mSSL #######
mixed_model <- mixed_mssl(X,Y,response_types = c("binary","continuous"),lambdas = list(lambda1 = 0.0001, lambda0 = 30),
                          xis = list(xi1 = 0.001 * nrow(X), xi0 = seq(0.01 * nrow(X), nrow(X), length = 10)),
                          theta_hyper_params = c(1, ncol(X) * ncol(Y)),
                          eta_hyper_params = c(1, ncol(Y)),
                          diag_penalty = 0,
                          max_iter = 500,
                          eps = 1e-3,
                          s_max_condition = 10 * nrow(X),
                          obj_counter_max = 5,
                          verbose = 1, nrep = 1000, nskp = 1)

B_matrix <- mixed_model$B
non_zero_rows <- apply(B_matrix, 1, function(row) any(row != 0))
which(non_zero_rows)

# 3,9,14,15,17,18,19

######### Mt-MBSP #######
response_types <- c('binary','continuous')
output <- Mt_MBSP(X, Y, response_types)
B_matrix <- output$B_active
non_zero_rows <- apply(B_matrix, 1, function(row) any(row != 0))
which(non_zero_rows)

# 3,11,12,14,19

######### sepSSL #########
B_matrix <- sepSSL(X,Y,type=response_types,lambda1 = 0.0001, lambda0 = 30)
non_zero_rows <- apply(B_matrix, 1, function(row) any(row != 0))
which(non_zero_rows)
# 3, 14, 15, 18, 19 


######### sepglm ########
B_matrix <- sepGLMnet(X,Y,type=response_types)[["B_est"]]
non_zero_rows <- apply(B_matrix, 1, function(row) any(row != 0))
which(non_zero_rows)

# 2,3,4,5,7,8,9,11,12,14,15,16,17,18,19,21 

############################### Prediction Analysis ###############################

library(caret)
library(pROC)

############## Predictive AUC ####################
set.seed(123)  
folds <- createFolds(Y[, 1], k = 5) 

auc_values <- numeric(length(folds))
roc_curves <- list()
all_prob_pred <- c()
all_Y_test <- c()


for (i in seq_along(folds)) {
  test_indices <- folds[[i]]
  train_indices <- setdiff(seq_len(nrow(X)), test_indices)
  Y <- as.matrix(cbind(cleaned_CKD$class, cleaned_CKD$sg))
  
  X_train <- X[train_indices, ]
  X_test <- X[test_indices, ]
  Y_train <- Y[train_indices,]  
  Y_test <- Y[test_indices,]
  
  ### Replace this chunk with mt-MBSP, sepssl, sepglm ###
  B_matrix <- mixed_mssl(X_train,Y_train,response_types = c("binary","continuous"),lambdas = list(lambda1 = 0.0001, lambda0 = 30),
                         xis = list(xi1 = 0.01 * nrow(X_train), xi0 = seq(0.1 * nrow(X_train), nrow(X_train), length = 10)),
                         theta_hyper_params = c(1, ncol(X_train) * ncol(Y_train)),
                         eta_hyper_params = c(1, ncol(Y_train)),
                         diag_penalty = 0,
                         max_iter = 500,
                         eps = 1e-3,
                         s_max_condition = 10 * nrow(X_train),
                         obj_counter_max = 5,
                         verbose = 1, nrep = 1000, nskp = 1))
  #########################################################
  non_zero_rows <- apply(B_matrix, 1, function(row) any(row != 0))
  non_zero_indices <- which(non_zero_rows)
  X_non_zero <- X[, non_zero_indices]
  
  X_train <- X_non_zero[train_indices,]
  X_test <- X_non_zero[test_indices,]
  Y_train <- Y[train_indices,1]
  Y_test <- Y[test_indices,1]
  model <- glm(as.factor(Y_train) ~ ., data = as.data.frame(X_train), family = binomial(link = "probit"))
  prob_pred <- predict(model, newdata = as.data.frame(X_test), type = "response")
  all_prob_pred <- c(all_prob_pred, prob_pred)
  all_Y_test <- c(all_Y_test, Y_test)
  
  auc_values[i] <- auc(Y_test, prob_pred)
}

average_roc <- roc(all_Y_test, all_prob_pred)
mean_auc <- mean(auc_values)
print(mean_auc)

########### Predictive RMSE #################
set.seed(123)  
folds <- createFolds(as.matrix(Y[, 1]), k = 5)  
mse <- numeric(length(folds))


for (i in seq_along(folds)) {
  test_indices <- folds[[i]]
  train_indices <- setdiff(seq_len(nrow(X)), test_indices)
  X_train <- X[train_indices, ]
  X_test <- X[test_indices, ]
  Y_train <- Y[train_indices,]  
  Y_test <- Y[test_indices,]
  B_matrix <- mixed_mssl(X_train,Y_train,response_types = c("binary","continuous"),lambdas = list(lambda1 = 0.0001, lambda0 = 30),
                         xis = list(xi1 = 0.01 * nrow(X_train), xi0 = seq(0.1 * nrow(X_train), nrow(X_train), length = 10)),
                         theta_hyper_params = c(1, ncol(X_train) * ncol(Y_train)),
                         eta_hyper_params = c(1, ncol(Y_train)),
                         diag_penalty = 0,
                         max_iter = 500,
                         eps = 1e-3,
                         s_max_condition = 10 * nrow(X_train),
                         obj_counter_max = 5,
                         verbose = 1, nrep = 1000, nskp = 1))
  non_zero_rows <- apply(B_matrix, 1, function(row) any(row != 0))
  non_zero_indices <- which(non_zero_rows)
  X_non_zero <- X[, non_zero_indices]
  
  X_train <- X_non_zero[train_indices,]
  X_test <- X_non_zero[test_indices,]
  Y_train <- Y[train_indices,3]
  Y_test <- Y[test_indices,3]
  model <- lm(Y_train$BMI ~ ., data = as.data.frame(X_train))
  Y_pred <- predict(model, newdata = as.data.frame(X_test), type = "response")
  mse[i] <-sqrt(mean((Y_pred-Y_test$BMI)^2))
}

average_mse <- mean(mse)
print(average_mse)
