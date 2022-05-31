### SL HW2 Test Mimmo

# Library
library(data.table)
library(caret)
library(ggplot2)
library(kernlab)
library(tidymodels)
library(tidyverse)
library(readr)
library(gsw)
library(oce)
library(parallel)
library(foreach)
library(doFuture)
library(dplyr)

# Import Data Train and Test ------------------------------------------------------------
train.path <- '/Users/domenicospoto/Desktop/Sapienza/MScDataScience/StatisticalLearning/HW2/data4final_hw/train4final_hw.csv'
test.path <- '/Users/domenicospoto/Desktop/Sapienza/MScDataScience/StatisticalLearning/HW2/data4final_hw/test4final_hw.csv'

train <- fread(train.path)
test <- fread(test.path)
print(dim(train)) # 1561x7042

# Point 1 ------------------------------------------------------------
# randomly pick m = 10 observations from the training set and put them aside
set.seed(123) # set seed for reproducibility
sample.m <- sample(1:nrow(train), 10)
m <- train[sample.m,]
train.new <- train[!sample.m,]
dim(train.new)

# convert to a matrix and split the X and y in train
X_train <- as.matrix(train.new[,-c('tempo')])
y_train <- as.matrix(train.new[,'tempo'])
dim(X_train)
dim(y_train)

# Smooth the MFCC and get the image ------------------------------------------------------------
audio_num <- 2
audio <- matrix(X_train[2,1:6840], nrow = 171, ncol = 40) %>% scale()
col_palette <- c("#FCFFA4FF", 
                 "#F5DB4BFF", 
                 "#FCAD12FF", 
                 "#F78311FF", 
                 "#E65D2FFF", 
                 "#CB4149FF",
                 "#A92E5EFF", 
                 "#85216BFF", 
                 "#60136EFF", 
                 "#3A0963FF", 
                 "#140B35FF", 
                 "#000004FF")
audio_smooth <- matrixSmooth(audio, passes = 2)
image(x = 1:171, y = 1:40, z = audio, 
      xlab = "Time instants", ylab = "Mel-Frequencies",
      col = col_palette)





# EDA ------------------------------------------------------------
## Which features are most (linearly) correlated to tempo
mel_columns <- 1:6840
freq_columns <- 6841:7011
time_columns <- 7012:7015
signal_columns <- 7034:7041

cor(X_train[,mel_columns], y = y_train) %>% 
  tibble("corr" = as.vector(.), "feat" = rownames(.)) %>%
  mutate("abs_corr" = abs(corr), feat) %>%
  arrange(desc(abs_corr)) %>%
  head(50) %>%
  select(feat, abs_corr) %>%
  ggplot() +
  aes(x = abs_corr, y = reorder(feat, abs_corr)) +
  geom_col(position=position_dodge(), width = 0.8) +
  theme(axis.text=element_text(size=5))


## Which not MEL feature are most (linearly) correlated to tempo
cor(X_train[, -c(1:6840)], y = y_train) %>%
  tibble("corr" = as.vector(.), "feat" = rownames(.)) %>%
  filter(feat != "tempo" & feat != "id") %>%
  mutate("abs_corr" = abs(corr), feat) %>%
  arrange(desc(abs_corr)) %>%
  select(feat, abs_corr) %>%
  head(50) %>%
  ggplot() +
  aes(x = abs_corr, y = reorder(feat, abs_corr)) +
  geom_col(position=position_dodge(), width = 0.8) +
  theme(axis.text=element_text(size=5))

## Which features are most (linearly) correlated to the frequency median
cor(X_train[, -c(1:7011)], y = X_train[,'freq.M']) %>%
  tibble("corr" = as.vector(.), "feat" = rownames(.)) %>%
  filter(feat != "freq.M" & feat != "id") %>%
  mutate("abs_corr" = abs(corr), feat) %>%
  arrange(desc(abs_corr)) %>%
  select(feat, abs_corr) %>%
  head(50) %>%
  ggplot() +
  aes(x = abs_corr, y = reorder(feat, abs_corr)) +
  geom_col(position=position_dodge(), width = 0.8) +
  theme(axis.text=element_text(size=5))

# Dimensionality Reduction ------------------------------------------------------------
# get only mel col
X_train_kpca <- X_train[, c(mel_columns)]
dim(X_train_kpca)

## Kernel PCA Test for multiple kernel
# ?kpca
# K_pca_mat <- data.frame(matrix(ncol = 3, nrow = 100))
# 
# #provide column names
# colnames(K_pca_mat) <- c('rbfdot', 'polydot', 'vanilladot')# 'laplacedot', 'besseldot', 'anovadot',,  'splinedot', 'tanhdot'
# 
# for(col in 1:ncol(K_pca_mat)){
#   print(colnames(K_pca_mat)[col])
#   if (colnames(K_pca_mat)[col] == 'polydot'){
#     X_kpca <- kpca(X_train,  kernel = colnames(K_pca_mat)[col], kpar = list(degree = 0.01, scale = 0.01, offset = 0.01), features=100)
#     K_pca_mat[, colnames(K_pca_mat)[col]] <- X_kpca@eig
#   }
#   else if (colnames(K_pca_mat)[col] == 'vanilladot'){
#     X_kpca <- kpca(X_train,  kernel = colnames(K_pca_mat)[col], kpar=list(), features=100)
#     K_pca_mat[, colnames(K_pca_mat)[col]] <- X_kpca@eig
#   }
#   # else if (colnames(K_pca_mat)[col] == 'tanhdot'){
#   #   X_kpca <- kpca(X_train,  kernel = colnames(K_pca_mat)[col], kpar = list(scale = 0.001, offset = 0.001), features=100) 
#   #   K_pca_mat[, colnames(K_pca_mat)[col]] <- X_kpca@eig
#   # }
#   # else if (colnames(K_pca_mat)[col] == 'laplacedot'){
#   #   X_kpca <- kpca(X_train,  kernel = colnames(K_pca_mat)[col], kpar=list(sigma = 1), features=100)
#   #   K_pca_mat[, colnames(K_pca_mat)[col]] <- X_kpca@eig
#   # }
#   # else if (colnames(K_pca_mat)[col] == 'besseldot'){
#   #   X_kpca <- kpca(X_train,  kernel = colnames(K_pca_mat)[col], kpar = list(), features=100)
#   #   K_pca_mat[, colnames(K_pca_mat)[col]] <- X_kpca@eig
#   # }
#   # else if (colnames(K_pca_mat)[col] == 'anovadot'){
#   #   X_kpca <- kpca(X_train,  kernel = colnames(K_pca_mat)[col], kpar = list(sigma = 0.01, degree = 0.01), features=100)
#   #   K_pca_mat[, colnames(K_pca_mat)[col]] <- X_kpca@eig
#   # }
#   # else if (colnames(K_pca_mat)[col] == 'splinedot'){
#   #     X_kpca <- kpca(X_train,  kernel = colnames(K_pca_mat)[col], kpar = list(), features=100)
#   #     K_pca_mat[, colnames(K_pca_mat)[col]] <- X_kpca@eig
#   # }
#   else{
#     X_kpca <- kpca(X_train,  kernel = colnames(K_pca_mat)[col], kpar = list(sigma = 0.01), features=100)
#     K_pca_mat[, colnames(K_pca_mat)[col]] <- X_kpca@eig
#   }
# }
# 
# # Get the principal component vectors
# X_pcv <- data.frame(pcv(K_pca_mat$vanilladot))
# names(X_pcv) <- paste("Comp", 1:ncol(X_pcv), sep = ".")
# X_pcv$tempo <- y_train
# head(X_pcv)
# dim(X_pcv)
# 
# colo1 <- RColorBrewer::brewer.pal(3, "PuOr")[c(1,3)]
# pairs(X_pcv, lower.panel = NULL, cex = .4, asp = 1, 
#       pch = 21, bg = colo1[X_pcv$tempo])



## Linear PCA
X_pca <- kpca(X_train_kpca,  kernel = 'vanilladot', kpar = list(), features=30)
X_pca@eig
X_pcv <- data.frame(pcv(X_pca))
names(X_pcv) <- paste("Comp", 1:ncol(X_pcv), sep = ".")

X_train_kpca_2 <- X_train[, c(time_columns, signal_columns)]
dim(X_train_kpca_2)


Train_pca <- bind_cols(X_pcv, X_train_kpca_2, y_train)
dim(Train_pca)
head(Train_pca)

as.data.frame(Train_pca) |> 
  ggplot(aes(Comp.1, Comp.3)) + 
  geom_point(aes(colour = tempo)) + 
  scale_colour_gradientn(colours = RColorBrewer::brewer.pal(4, "RdYlBu"))


# Modelling ------------------------------------------------------------

## tidymodels
data_to_model <- tibble(Train_pca)
head(Train_pca)

# Training/testing
set.seed(123)
tr_te_split <- data_to_model %>% initial_split(prop = 0.7, strata = tempo)
training <- training(tr_te_split)
testing <- testing(tr_te_split)

# model
svm_rbf_spec <- svm_rbf(cost = tune(), rbf_sigma = tune()) %>%
  set_mode("regression") %>%
  set_engine("kernlab")

# svm_rbf_spec <- svm_linear(cost = tune(), margin = tune()) %>%
#   set_mode("regression") %>%
#   set_engine("kernlab")

# fitting workflow
svm_rbf_wf <- workflow() %>%
  add_model(svm_rbf_spec) %>%
  add_formula(tempo ~ .)

# Parallel
num_cores <- detectCores() - 1
registerDoFuture()

# parallel computing
cl <- makeCluster(num_cores)
plan(cluster, workers = cl)

# construct a 5x5 grid of sensible values
# of cost and sigma to check
svm_rbf_grid <- grid_regular(cost(), rbf_sigma(), levels = 5)

# 10 folds for cross validation
folds <- vfold_cv(training, v = 10, strata = tempo)

# fit for each set of folds
# for all parameters in the grid
svm_rbf_res <- svm_rbf_wf %>%
  tune_grid(resamples = folds,
            grid = svm_rbf_grid)


# Compare the parameters
svm_rbf_res %>%
  collect_metrics() %>%
  mutate(cost = factor(cost)) %>%
  ggplot(aes(rbf_sigma, mean, color = cost)) +
  geom_line(size = 1.5, alpha = 0.6) +
  geom_point(size = 2) +
  facet_wrap(~ .metric, scales = "free", nrow = 2) +
  scale_x_log10(labels = scales::label_number()) +
  scale_color_viridis_d(option = "plasma", begin = .9, end = 0)

# Fit with the best parameters
best_svm <- svm_rbf_res %>% select_best("rmse")
svm_rbf_wf_final <- svm_rbf_wf %>% finalize_workflow(best_svm)
svm_rbf_fit <- svm_rbf_wf_final %>% last_fit(tr_te_split) 
stopCluster(cl)

# Final RMSE
svm_rbf_fit %>% 
  collect_metrics() %>%
  filter(.metric == "rmse") %>%
  select(.estimate) %>%
  deframe %>%
  paste("RMSE =", .) %>%
  cat()

final_svm_rbf <- extract_workflow(svm_rbf_fit)
final_svm_rbf

library(vip)

final_svm_rbf %>% 
  extract_fit_parsnip() %>% 
  vip()


# svm_rbf_fit$.predictions

# Save model
# svm_rbf_fit %>% 
#   extract_workflow() %>% 
#   readr::write_rds("svm_rbf.rds")

