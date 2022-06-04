### SL HW2 Test Mimmo Part 1

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
library(plotly)
library(factoextra)

# Import Data Train and Test ------------------------------------------------------------
folder_path <- '/Users/domenicospoto/Desktop/Sapienza/MScDataScience/StatisticalLearning/HW2/SL-HW2/' 
train.path <- '/Users/domenicospoto/Desktop/Sapienza/MScDataScience/StatisticalLearning/HW2/data4final_hw/train4final_hw.csv'
test.path <- '/Users/domenicospoto/Desktop/Sapienza/MScDataScience/StatisticalLearning/HW2/data4final_hw/test4final_hw.csv'

train <- fread(train.path)
test <- fread(test.path)
print(dim(train)) # 1561x7042
print(dim(test)) # 670x7042

# Parallel
num_cores <- detectCores() - 1
registerDoFuture()

# Part A ------------------------------------------------------------
# randomly pick m = 10 observations from the training set and put them aside
set.seed(123) # set seed for reproducibility
sample.m <- sample(1:nrow(train), 10)
m <- as.matrix(train[sample.m,])
train.new <- train[!sample.m,]
dim(train.new)

# convert to a matrix and split the X and y in train
X_train <- as.matrix(train.new[,-c('id','tempo')])
tempo <- train.new$tempo
dim(X_train)
dim(y_train)

mel_columns <- 1:6840
freq_columns <- 6841:7011
time_columns <- 7012:7015
signal_columns <- 7034:7041


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

## Linear PCA only MEL coeff
X_train_pca <- X_train[, c(mel_columns)]
pc <- X_train_pca %>%
  prcomp(center = TRUE, scale = TRUE)


dim(pc$rotation)
X_pca_mel <- dim(pc$x)[, 1:20] %>% as.data.frame()
colnames(X_pca_mel) <- paste("pc_mel", 1:ncol(X_pca_mel), sep = "")


rownames(pc$x)

# Check separation of the components
X_pca_mel %>%
  ggplot(aes(pc_mel3, pc_mel4)) +
  geom_point(aes(colour = tempo)) +
  scale_colour_gradientn(colours = RColorBrewer::brewer.pal(4, "RdYlBu"))

# Linear KPCA only MEL coeff (only a test)
# X_pca <- kpca(X_train_kpca,  kernel = 'vanilladot', kpar = list())
#
# X_pcv <- data.frame(pcv(X_pca))
# names(X_pcv) <- paste("Comp", 1:ncol(X_pcv), sep = ".")

## Kernel PCA of all dominant frequencies and frequency related statistics
X_kpca_freq <- X_train[, c(6841:7011, 7016:7032, 7034:7039)] %>%
  scale() %>%
  kpca(x = .,
       kernel = "rbfdot",
       kpar = list(sigma = 0.3),
       features = 5) %>%
  pcv() %>%
  as.data.frame()
colnames(X_kpca_freq) <- paste("pc_freq", 1:ncol(X_kpca_freq), sep = "")

# Check separation of the components
# Separation of KPCA components
X_kpca_freq %>%
  ggplot(aes(pc_freq1, pc_freq2)) +
  geom_point(aes(colour = tempo)) +
  scale_colour_gradientn(colours = RColorBrewer::brewer.pal(4, "RdYlBu"))

X_kpca_freq %>% plot_ly(
  x = ~ pc_freq1,
  y = ~ pc_freq2,
  z = ~ pc_freq3,
  type = "scatter3d",
  size = 1,
  mode = "markers",
  color = tempo,
  colors = "RdYlBu"
)

## Group everything in final dataset
Train_pca <- bind_cols(X_pca_mel, X_kpca_freq, genre = X_train[, "genre"])
dim(Train_pca)

## One-hot encoding of genre
Train_pca[, paste("genre_", as.character(unique(Train_pca[, "genre"])), sep = "")] <- 0

for (i in 1:nrow(Train_pca)){
  genre <- Train_pca[i, "genre"]
  Train_pca[i, paste("genre_", genre, sep = "")] <- 1
}
head(Train_pca)

# Group Final dataset
data_to_model <- tibble(Train_pca) %>% select(-genre)
data_to_model$tempo <- tempo
head(data_to_model)
dim(data_to_model)

# Modelling ------------------------------------------------------------

# Steps
# 
# 1.  Split data in 70% Training and 30% Testing with stratified sampling
# 
# 2.  Pick model
# 
# 3.  Grid search to find best parameters using k-Fold Cross Validation
# 
# 4.  Fit with the best parameters and compute RMSE

## Training/testing
set.seed(123)
tr_te_split <- data_to_model %>% initial_split(prop = 0.7, strata = tempo)

training <- training(tr_te_split)
testing <- testing(tr_te_split)

## SVM RBF
## Tuning
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

# parallel computing
cl <- makeCluster(num_cores)
plan(cluster, workers = cl)

# construct a 5x5 grid of sensible values
# of cost and sigma to check
svm_rbf_grid <- grid_regular(cost(), rbf_sigma(), levels = 5)

# 10 folds for cross validation
folds <- vfold_cv(training, v = 5, strata = tempo)

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

## Predictions
svm_rbf_fit$.predictions

### Predictions vs Observations
testing_rows <- svm_rbf_fit$.predictions[[1]]$.row

svm_rbf_fit$.predictions[[1]] %>%
  select(tempo, .pred, .row) %>%
  as.data.frame() %>%
  cbind(X_train[testing_rows, ]) %>%
  ggplot() +
  aes(tempo, .pred, colour = genre) +
  # aes(colour = genre) +
  geom_point() + 
  geom_smooth(alpha = 0.2) +
  scale_colour_gradientn(colours = RColorBrewer::brewer.pal(4, "RdYlBu"))

## Residuals
# Residuals
svm_rbf_residuals <- svm_rbf_fit$.predictions[[1]] %>%
 mutate(res = tempo - .pred) %>%
 select(res) %>%
 as.vector() %>%
 .$res

  
### Check their correlation with the other variables -->
# Correlation of residuals
cor(data_to_model[testing_rows, ], y = svm_rbf_residuals) %>%
   tibble("corr" = as.vector(.), "feat" = rownames(.)) %>%
   filter(corr > -0.1 & corr < 0.1) %>%
   ggplot() +
   aes(x = corr, y = reorder(feat, corr)) +
   geom_col(position=position_dodge(), width = 0.8) +
   theme(axis.text=element_text(size=5))

# Final Modelling (for Kaggle) ------------------------------------------------------------
### Tuning
# Training for Kaggle
# train on full training dataset
training <- data_to_model

# model
svm_rbf_spec <- svm_rbf(cost = tune(), rbf_sigma = tune()) %>%
  set_mode("regression") %>%
  set_engine("kernlab")

# fitting workflow
svm_rbf_wf <- workflow() %>%
  add_model(svm_rbf_spec) %>%
  add_formula(tempo ~ .)

# parallel computing
cl <- makeCluster(num_cores)
plan(cluster, workers = cl)

# construct a 5x5 grid of sensible values
# of cost and sigma to check
# svm_rbf_grid <- grid_regular(cost(range = c(18, 23), trans = identity_trans()), 
#                              rbf_sigma(range = c(0.0045, 0.007), trans = identity_trans()), levels = 5)
svm_rbf_grid <- grid_regular(cost(), 
                             rbf_sigma(), levels = 5)

# 10 folds for cross validation
folds <- vfold_cv(training, v = 5, strata = tempo)

# fit for each set of folds
# for all parameters in the grid
svm_rbf_res <- svm_rbf_wf %>%
  tune_grid(resamples = folds,
            grid = svm_rbf_grid)

### Compare the parameters
# Compare tuned parameters for Kaggle
svm_rbf_res %>%
  collect_metrics() %>%
  filter(.metric == "rmse") %>%
  mutate(cost = factor(cost)) %>%
  ggplot(aes(rbf_sigma, mean, color = cost)) +
  geom_line(size = 1.5, alpha = 0.6) +
  geom_point(size = 2) +
  # facet_wrap(~ .metric, scales = "free", nrow = 2) +
  scale_x_log10(labels = scales::label_number()) +
  scale_color_viridis_d(option = "plasma", begin = .9, end = 0)

### Lowest RMSE from cross validation
# Lowest RMSE from cross validation
svm_rbf_res %>% 
  collect_metrics() %>%
  filter(.metric == "rmse") %>%
  filter(mean == min(mean)) %>%
  select(cost, rbf_sigma, .metric, mean, std_err)

### Fit with the best parameters

# Final fit for Kaggle
best_svm <- svm_rbf_res %>% select_best("rmse")

svm_rbf_wf_final <- svm_rbf_wf %>% finalize_workflow(best_svm)

svm_rbf_fit <- svm_rbf_wf_final %>% fit(training) 
stopCluster(cl)

###  Save model
# <https://community.rstudio.com/t/saving-a-model-fit-with-tidymodels/114839>
# Save model for Kaggle
current_time <- Sys.time() %>% format("%Y%m%d_%H%M%S")

filename <- "models/svm_rbf" %>% 
  paste(current_time, sep = "_") %>% 
  paste(".rds", sep = "")

svm_rbf_fit %>% 
  readr::write_rds(file =  filename)


# Part B ------------------------------------------------------------
set.seed(12345)
n <- 1000
U <- rnorm(n)
hist(U)

u <- 2
Ua <- c(U, u)
hist(Ua)
abline(v=u)

# Naive CP
# Get (X,Y)
X <- rnorm(n)
Y <- X + U
regData <- data.frame(X,Y)
# run the A
fitlm <- lm(Y~X, data=regData)
# compute Abs(residuals)
eVec <- abs(fitlm$residuals)
hist(eVec)

plot(X,Y)
# add new X
Xnew <- -1.5
abline(fitlm)
abline(v=Xnew, lty="dashed")
# predict new Y
muHat <- predict(fitlm, newdata=data.frame(X=Xnew))
points(Xnew, muHat, pch=19, col="red")
# create Conformal Intervals
C.X <- c(muHat-quantile(eVec, .975), muHat+quantile(eVec, .975))
points(rep(Xnew, 2), C.X, type="l", col="red", lwd=2)


# Conformal Pred
nEval <- 200
yCand <- seq(from=min(Y), to=max(Y), length=nEval)

confPredict <- function(y, Xin){
  nData <- nrow(regData)  
  regData.a <- rbind(regData,c(Xin, y))
  fitlm.a <- lm(Y~X, data=regData.a)
  resOut <- abs(fitlm.a$residuals)
  resOut_new <- resOut[length(resOut)]
  pi.y <- mean(apply(as.matrix(resOut),
                     1,
                     function(x){x<=resOut_new}))
  testResult <- pi.y*(nData+1) <= ceiling(.975*(nData+1))
  return(testResult)
}

Cxa <- range(yCand[sapply(yCand, confPredict, Xin=Xnew)])

plot(X,Y)
abline(fitlm)
abline(v=Xnew, lty="dashed")
points(rep((Xnew+.05), 2), C.X, type="l", col="red", lwd=2)
points(rep(Xnew, 2), Cxa, type="l", col="blue", lwd=3)


# Split CP
splitConfPredict <- function(Xin){
  nData <- nrow(regData)
  regData$index <- 1:nData
  regData$split <- 1
  regData$split[sample(regData$index, floor(nrow(regData)/2), replace=F)] <- 2
  fitlm.spl <- lm(Y~X, data=subset(regData, split==1))
  resOut <- abs(subset(regData, split==2)$Y - predict(fitlm.spl,
                                                      newdata=subset(regData, split==2)))
  kOut <- ceiling(((nData/2)+1)*(.975))
  resUse <- resOut[order(resOut)][kOut]
  Y.hat <- predict(fitlm.spl, newdata=data.frame(X=Xin))
  C.split <- c(Y.hat-resUse,Y.hat+resUse)
  return(C.split)
}

plot(X,Y)
abline(fitlm)
abline(v=Xnew, lty="dashed")
points(rep((Xnew+.05), 2), C.X, type="l", col="red", lwd=2)
points(rep(Xnew, 2), Cxa, type="l", col="blue", lwd=3)
points(rep(Xnew-.05, 2), splitConfPredict(Xnew), type="l", col="green", lwd=3)

# Point 1 ------------------------------------------------------------
## Import model
model_date <- "20220604_164949"

svm_rbf_fit <- paste("models/", model_date, ".rds", sep = "") %>%
  readr::read_rds(.)

svm_rbf_fit

# parallel computing
cl <- makeCluster(num_cores)
plan(cluster, workers = cl)

# Implement Split Conformal Prediction
splitConfPredict <- function(Xin){
  nData <- nrow(regData)
  regData$index <- 1:nData
  regData$split <- 1
  regData$split[sample(regData$index, floor(nrow(regData)/2), replace=F)] <- 2
  # fitlm.spl <- lm(Y~X, data=subset(regData, split==1))
  fitlm.spl <- svm_rbf_fit %>% fit(Y~X, data=subset(regData, split==1)) 
  resOut <- abs(subset(regData, split==2)$Y - predict(fitlm.spl,
                                                      newdata=subset(regData, split==2)))
  kOut <- ceiling(((nData/2)+1)*(.975))
  resUse <- resOut[order(resOut)][kOut]
  Y.hat <- predict(fitlm.spl, newdata=data.frame(X=Xin))
  C.split <- c(Y.hat-resUse,Y.hat+resUse)
  return(C.split)
}

X <- as.matrix(tibble(data_to_model) %>% select(-tempo))
Y <- as.matrix(data_to_model$tempo)
regData <- data_to_model


Xnew <- m_data_to_model %>% select(-tempo) %>% slice(1)
splitConfPredict(Xnew)









