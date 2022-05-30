### SL HW2 Test Mimmo

# Library
library(data.table)
library(caret)
library(ggplot2)
library(kernlab)

### Part 1
train.path <- '/Users/domenicospoto/Desktop/Sapienza/MScDataScience/StatisticalLearning/HW2/data4final_hw/train4final_hw.csv'
test.path <- '/Users/domenicospoto/Desktop/Sapienza/MScDataScience/StatisticalLearning/HW2/data4final_hw/test4final_hw.csv'

train <- fread(train.path)
test <- fread(test.path)
print(dim(train)) # 1561x7042

## Point 1
# randomly pick m = 10 observations from the training set and put them aside
set.seed(123) # set seed for reproducibility
sample.m <- sample(1:nrow(train), 10)
m <- train[sample.m,]
train.new <- train[!sample.m,]

# convert to a matrix and split the X and y in train
X_train <- as.matrix(train.new[,-c('id', 'genre', 'tempo')])
y_train <- as.matrix(train.new[,'tempo'])
dim(X_train)
dim(y_train)

# get only mel col
mel_columns <- 1:6840
time_columns <- 7012:7015
signal_columns <- 7034:7039
X_train <- X_train[, c(mel_columns)]
dim(X_train)

# Kernel PCA
?kpca
K_pca_mat <- data.frame(matrix(ncol = 8, nrow = 100))

#provide column names
colnames(K_pca_mat) <- c('rbfdot', 'polydot', 'vanilladot', 'tanhdot', 'laplacedot', 'besseldot', 'anovadot', 'splinedot')


for(col in 1:ncol(K_pca_mat)){
  print(colnames(K_pca_mat)[col])
  if (colnames(K_pca_mat)[col] == 'polydot'){
    X_kpca <- kpca(X_train,  kernel = colnames(K_pca_mat)[col], kpar = list(degree = 0.01, scale = 0.01, offset = 0.01), features=100)
    K_pca_mat[, colnames(K_pca_mat)[col]] <- X_kpca@eig
  }
  else if (colnames(K_pca_mat)[col] == 'tanhdot'){
    X_kpca <- kpca(X_train,  kernel = colnames(K_pca_mat)[col], kpar = list(scale = 0.01, offset = 0.01), features=100)
    K_pca_mat[, colnames(K_pca_mat)[col]] <- X_kpca@eig
  }
  else if (colnames(K_pca_mat)[col] == 'besseldot'){
    # X_kpca <- kpca(X_train,  kernel = colnames(K_pca_mat)[col], kpar = list(), features=100)
    # K_pca_mat[, colnames(K_pca_mat)[col]] <- X_kpca@eig
  }
  else if (colnames(K_pca_mat)[col] == 'anovadot'){
    X_kpca <- kpca(X_train,  kernel = colnames(K_pca_mat)[col], kpar = list(sigma = 0.01, degree = 0.01), features=100)
    K_pca_mat[, colnames(K_pca_mat)[col]] <- X_kpca@eig
  }
  else if (colnames(K_pca_mat)[col] == 'vanilladot'){
    X_kpca <- kpca(X_train,  kernel = colnames(K_pca_mat)[col], kpar=list(), features=100)
    K_pca_mat[, colnames(K_pca_mat)[col]] <- X_kpca@eig
  }
  else if (colnames(K_pca_mat)[col] == 'laplacedot'){
    # X_kpca <- kpca(X_train,  kernel = colnames(K_pca_mat)[col], kpar=list(sigma = 1), features=100)
    # K_pca_mat[, colnames(K_pca_mat)[col]] <- X_kpca@eig
  }
  else{
    X_kpca <- kpca(X_train,  kernel = colnames(K_pca_mat)[col], kpar = list(sigma = 0.01), features=100)
    K_pca_mat[, colnames(K_pca_mat)[col]] <- X_kpca@eig
  }
}





# Get the principal component vectors
X_pcv <- data.frame(dpcv(X_kpca))
names(X_pcv) <- paste("Comp", 1:ncol(X_pcv), sep = ".")
X_pcv$tempo <- y_train
head(X_pcv)
dim(X_pcv)

colo1 <- RColorBrewer::brewer.pal(3, "PuOr")[c(1,3)]
pairs(X_pcv, lower.panel = NULL, cex = .4, asp = 1, 
      pch = 21, bg = colo1[X_pcv$tempo])








# data split
suppressMessages(require(caret, quietly = T))
id_tr <- createDataPartition(train.new$tempo, p = .7, list = FALSE)
x_tr <- train.new[ id_tr, ]
x_te <- train.new[-id_tr, ]











