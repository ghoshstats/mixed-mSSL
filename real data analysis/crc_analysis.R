load("meta.RData")
load("count.RData")
library(MtMBSP)
library(caret)
library(pROC)
source("mixed-mssl.R")
source("other_methods.R")

X <- count
norm_count <- count/rowSums(count)
col_means <- colMeans(norm_count > 0)
indices <- which(col_means > 0.2)
sorted_indices <- indices[order(col_means[indices], decreasing=TRUE)]
dcount <- count[,sorted_indices]

Y <- meta[, c("Diabetes", "Group", "BMI")]
Y <- Y[complete.cases(Y), ]
Y$Group <- ifelse(Y$Group == "CRC",      1L, 0L)
row_idx <- which(!is.na(meta$BMI) & !is.na(meta$Diabetes) & !is.na(meta$Group))
X <- dcount[row_idx,]

################## mixed-mSSL ##################
mixed_model <- mixed_mssl(X,Y,response_types = c("binary","binary","continuous"),lambdas = list(lambda1 = 0.01, lambda0 = 40),
                          xis = list(xi1 = 0.001 * nrow(X), xi0 = seq(0.01 * nrow(X), nrow(X), length = 10)),
                          theta_hyper_params = c(1, ncol(X) * ncol(Y)),
                          eta_hyper_params = c(1, ncol(Y)),
                          diag_penalty = 0,
                          max_iter = 500,
                          eps = 1e-3,
                          s_max_condition = 10 * nrow(X),
                          obj_counter_max = 5,
                          verbose = 0, nrep = 1000, nskp = 1)
B_matrix <- mixed_model$B
non_zero_rows <- apply(B_matrix, 1, function(row) any(row != 0))
which(non_zero_rows)
colnames(X[,non_zero_rows])
Omega <- mixed_model$Omega

############### Mt-MBSP ###############
response_types <- c('binary','binary','continuous')
output <- Mt_MBSP(X, Y, response_types)
B_matrix <- output$B_active
non_zero_rows <- apply(B_matrix, 1, function(row) any(row != 0))
which(non_zero_rows)

############## sepSSL ################
B_est <- sepSSL(X,Y,type=c("binary","binary","continuous"),lambda1 = 0.01, lambda0 = 40)
non_zero_rows <- apply(B_est, 1, function(row) any(row != 0))
which(as.numeric(non_zero_rows) ==1)

############# sepglm #################
B_est <- sepGLMnet(X,Y,type=c("binary","binary","continuous"))
non_zero_rows <- apply(B_est, 1, function(row) any(row != 0))
which(as.numeric(non_zero_rows) ==1)


##################### Prediction Analysis #########################
########## AUC ###############
set.seed(123)  
folds <- createFolds(as.matrix(Y[, 2]), k = 5)  
auc_values <- numeric(length(folds))
roc_curves <- list()
all_prob_pred <- c()
all_Y_test <- c()

for (i in seq_along(folds)) {
  test_indices <- folds[[i]]
  train_indices <- setdiff(seq_len(nrow(X)), test_indices)
  X_train <- X[train_indices, ]
  X_test <- X[test_indices, ]
  Y_train <- Y[train_indices,]  
  Y_test <- Y[test_indices,]
  
  
  mixed_model <- mixed_mssl(as.matrix(X_train),as.matrix(Y_train),response_types = c("binary","binary","continuous"),lambdas = list(lambda1 = 0.0001, lambda0 = 30),
                               xis = list(xi1 = 0.01 * nrow(X_train), xi0 = seq(0.1 * nrow(X_train), nrow(X_train), length = 10)),
                               theta_hyper_params = c(1, ncol(X_train) * ncol(Y_train)),
                               eta_hyper_params = c(1, ncol(Y_train)),
                               diag_penalty = 0,
                               max_iter = 500,
                               eps = 1e-3,
                               s_max_condition = 10 * nrow(X_train),
                               obj_counter_max = 5,
                               verbose = 1, nrep = 1000, nskp = 1)
  
  B_matrix <- mixed_model$B
  
  non_zero_rows <- apply(B_matrix, 1, function(row) any(row != 0))
  non_zero_indices <- which(non_zero_rows)
  X_non_zero <- X[, non_zero_indices]
  
  X_train <- X_non_zero[train_indices,]
  X_test <- X_non_zero[test_indices,]
  Y_train <- Y[train_indices,2]
  Y_test <- Y[test_indices,2]
  model <- glm(Y_train$Group ~ ., data = as.data.frame(X_train), family = "binomial")
  prob_pred <- predict(model, newdata = as.data.frame(X_test), type = "response")
  all_prob_pred <- c(all_prob_pred, prob_pred)
  all_Y_test <- c(all_Y_test, Y_test)
  auc_values[i] <- auc(Y_test$Group, prob_pred)
}

average_roc <- roc(unlist(all_Y_test), all_prob_pred)
mean_auc <- mean(auc_values)

######## RMSE ##########

set.seed(123)  # For reproducibility
folds <- createFolds(as.matrix(Y[, 1]), k = 5)  # 5-fold CV
mse <- numeric(length(folds))


for (i in seq_along(folds)) {
  test_indices <- folds[[i]]
  train_indices <- setdiff(seq_len(nrow(X)), test_indices)

  X_train <- X[train_indices, ]
  X_test <- X[test_indices, ]
  Y_train <- Y[train_indices,]  
  Y_test <- Y[test_indices,]
  
  mixed_model <- mixed_mssl(as.matrix(X_train),as.matrix(Y_train),response_types = c("binary","binary","continuous"),lambdas = list(lambda1 = 0.0001, lambda0 = 30),
                            xis = list(xi1 = 0.01 * nrow(X_train), xi0 = seq(0.1 * nrow(X_train), nrow(X_train), length = 10)),
                            theta_hyper_params = c(1, ncol(X_train) * ncol(Y_train)),
                            eta_hyper_params = c(1, ncol(Y_train)),
                            diag_penalty = 0,
                            max_iter = 500,
                            eps = 1e-3,
                            s_max_condition = 10 * nrow(X_train),
                            obj_counter_max = 5,
                            verbose = 1, nrep = 1000, nskp = 1)
  
  B_matrix <- mixed_model$B
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
