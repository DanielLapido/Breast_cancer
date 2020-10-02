##########################################
### Breast cancer Study
#   used for tumor classification 
##########################################

rm(list = ls())


if(!require(highcharter)){
  install.packages("highcharter")
}
if(!require(DMwR)){
  install.packages(DMwR)
}
library(DMwR)
library(highcharter)
library(pROC)
library(deepnet)
library(tidyverse)
library(skimr)
library(forcats)
library(VIM)
library(GGally)
library(MASS)
library(caret)
library(randomForest)
library(gbm)
library(e1071)
library(gridExtra)
library(glmnet)
library(neuralnet)
library(kernlab)
options(digits=4)

### Motivation
#
# Breast cancer is produced when the mamary cells begin to grow without control.
# Cancerous cells normally form a tumor that can be usually observed with a radiography
# or can be palpated as a bulk.

# Breast cancer is the most frequent cancer type
# within women of all around the world. Although the rates are higher in developed countries, it is spreading worldwide.
# It was the most frequent death cause in 11 regions of the world.

# More information is available every year to early diagnose breast cancer,
# which is crucial in the prognosis and chance of survival.
# The accurate classification of tumors is vital in order
# to start a quick treatment for malign ones and to not force patients to go through unnecesary therapy in the benign cases.


# The Breast Cancer dataset contains data on 30 predictors and the objective is to predict
# the probability that somebody has a malign tumor.

# Data can be found in UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)

# Loading and preparing data
cancer = read_csv("cancer.csv")

# A glimpse of the data can be obtained:
glimpse(cancer)


# Variable description

# There are ten features

# a) radius (mean of distances from center to points on the perimeter) 
# b) texture (standard deviation of gray-scale values)
# c) perimeter 
# d) area
# e) smoothness (local variation in radius lengths)
# f) compactness (perimeter^2 / area - 1.0)
# g) concavity (severity of concave portions of the contour)
# h) concave points (number of concave portions of the contour)
# i) symmetry
# j) fractal dimension ("coastline approximation" - 1)
#------------------------------------------------

#Identification number is removed and the target is converted into factors.
cancer = cancer[,-1]
cancer$diagnosis = as.factor(cancer$diagnosis)
levels(cancer$diagnosis)=c("B", "M")

#Change problematic name:
names(cancer) <- make.names(names(cancer))

# Split data into training and testing sets using the caret package
in_train <- createDataPartition(cancer$diagnosis, p = 0.8, list = FALSE)  # 80% for training
training <- cancer[ in_train,]
testing <- cancer[-in_train,]



### KNN ###

ctrl <- trainControl(method = "cv", number = 5,
                     classProbs = TRUE, 
                     verboseIter=T)
ctrl$sampling <- "up"

knn.train = train(diagnosis ~ ., 
                  method = "knn", 
                  data = training,
                  preProcess = c("center", "scale"),
                  tuneGrid = expand.grid(k = c(1:20)),
                  trControl = ctrl)

knnPred = predict(knn.train, testing)
cmknn = confusionMatrix(knnPred,testing$diagnosis)

aux = which.max(knn.train$results[,2])
optimal_k = knn.train$results[aux,1]
sub = paste("Optimal number of k is", optimal_k, "(accuracy :", knn.train$results[aux,2],") in KNN")

hchart(knn.train$results[,1:2], 'line', hcaes(as.integer(k), Accuracy)) %>%
  hc_subtitle(text = sub) %>%
  hc_title(text = "Accuracy With Varying K (KNN)") %>%
  hc_add_theme(hc_theme_google()) %>%
  hc_xAxis(title = list(text = "Number of Neighbors(k)")) %>%
  hc_yAxis(title = list(text = "Accuracy"))

knitr::kable(cmknn[["table"]])
knitr::kable(cmknn[["overall"]])
plot(knn.train, main="Accuracy with varying K")


### RANDOM FOREST ###

rf.train <- train(diagnosis ~., 
                  method = "rf", 
                  data = training,
                  preProcess = c("center", "scale"),
                  ntree = 200,
                  cutoff=c(0.7,0.3),
                  tuneGrid = expand.grid(mtry=c(6,8,10)), 
                  metric = "Kappa",
                  maximize = F,
                  trControl = ctrl)

rfPred = predict(rf.train, testing)
cmrf = confusionMatrix(rfPred,testing$diagnosis)
# Variable importance
rf_imp <- varImp(rf.train, scale = F)
plot(rf_imp, scales = list(y = list(cex = .95)), main="Variable importance")
plot(rf.train$finalModel, main="Optimal number of trees and mtry")

knitr::kable(cmrf[["table"]])
knitr::kable(cmrf[["overall"]])


### SVM ### 

svm.train = train(diagnosis ~., method = "svmRadial", 
                  data = training,
                  preProcess = c("center", "scale"),
                  tuneGrid = expand.grid(C = c(.25, .5, 1),
                                         sigma = c(0.01,.05)),
                  metric = "Kappa",
                  trControl = ctrl)

svmPred = predict(svm.train, testing)
cmsvm = confusionMatrix(svmPred,testing$diagnosis)

knitr::kable(cmsvm[["table"]])
knitr::kable(cmsvm[["overall"]])

### GRADIENT BOOSTING ###

GBM.train <- gbm(ifelse(training$diagnosis=="B",0,1) ~., data=training,
                 distribution= "bernoulli",n.trees=250,shrinkage = 0.01,interaction.depth=2,n.minobsinnode = 8)
threshold = 0.5
gbmProb = predict(GBM.train, newdata=testing, n.trees=250, type="response")
gbmPred = rep("B", nrow(testing))
gbmPred[which(gbmProb > threshold)] = "M"
cmgb = confusionMatrix(factor(gbmPred), testing$diagnosis)
knitr::kable(cmgb[["table"]])
knitr::kable(cmgb[["overall"]])



### NEURAL NETWORK ###

nn.train <- train(diagnosis ~., 
                  method = "nnet",
                  data = training,
                  preProcess = c("center", "scale"),
                  MaxNWts = 1000,
                  maxit = 100,
                  tuneGrid = expand.grid(size=c(2,4,6), decay=c(0.01,0.001)), 
                  metric = "Kappa",
                  maximize = F,
                  trControl = ctrl)

nn_imp <- varImp(nn.train, scale = F)
plot(nn_imp, scales = list(y = list(cex = .95)), main="Variable Importance")
plot(nn.train, main="Hyperparameter tunning")

threshold = 0.5
nnProb = predict(nn.train, newdata=testing, type="prob")
nnPred = rep("B", nrow(testing))
nnPred[which(nnProb[,2] > threshold)] = "M"
cmnn = confusionMatrix(factor(nnPred), testing$diagnosis)

knitr::kable(cmnn[["table"]])
knitr::kable(cmnn[["overall"]])


### DEEP NEURAL NETWORK ###

ctrl$sampling <- "smote"
ctrl$number<-2
# Deep Neural Network
dnn.train <- train(diagnosis ~., 
                   method = "dnn", 
                   data = training,
                   preProcess = c("center", "scale"),
                   numepochs = 20, # number of iterations on the whole training set
                   tuneGrid = expand.grid(layer1 = 1:4,
                                          layer2 = 0:2,#We include layers with zero nodes because if the optimal is zero nodes, that means that we only needed 1 layer
                                          layer3 = 0:2,
                                          hidden_dropout = 0, 
                                          visible_dropout = 0),
                   metric = "Kappa",
                   maximize = F,
                   trControl = ctrl)

threshold = 0.5
dnnProb = predict(dnn.train, newdata=testing, type="prob")
dnnPred = rep("B", nrow(testing))
dnnPred[which(dnnProb[,2] > threshold)] = "M"
cmdnn = confusionMatrix(factor(dnnPred), testing$diagnosis)
plot(dnn.train, main="Hyperparameter tuning")


knitr::kable(cmdnn[["table"]])
knitr::kable(cmdnn[["overall"]])

### Comparison of results

par(mfrow=c(2,3))
fourfoldplot(cmknn$table, color = c("#B22222", "#2E8B57"),
             main=paste("KNN (",round(cmknn$overall[1]*100),"%)",sep=""))
fourfoldplot(cmrf$table, color = c("#B22222", "#2E8B57"),
             main=paste("Random.F (",round(cmrf$overall[1]*100),"%)",sep=""))
fourfoldplot(cmsvm$table, color = c("#B22222", "#2E8B57"),
             main=paste("SVM (",round(cmsvm$overall[1]*100),"%)",sep=""))
fourfoldplot(cmgb$table, color = c("#B22222", "#2E8B57"),
             main=paste("Gradient.B (",round(cmgb$overall[1]*100),"%)",sep=""))
fourfoldplot(cmnn$table, color = c("#B22222", "#2E8B57"),
             main=paste("NN (",round(cmnn$overall[1]*100),"%)",sep=""))

fourfoldplot(cmdnn$table, color = c("#B22222", "#2E8B57"),
             main=paste("Deep NN (",round(cmdnn$overall[1]*100),"%)",sep=""))


Algorithms = c("KNN", "Random Forest", "SVM", "Gradient Boosting", "Neural Networks", "Deep Neural Networks")
Accuracy = c(cmknn$overall[1], cmrf$overall[1],cmsvm$overall[1],cmgb$overall[1],cmnn$overall[1],cmdnn$overall[1])
Kappa=c(cmknn$overall[2], cmrf$overall[2],cmsvm$overall[2],cmgb$overall[2],cmnn$overall[2],cmdnn$overall[2])
True_negatives = c(cmknn$table[1], cmrf$table[1],cmsvm$table[1],cmgb$table[1],cmnn$table[1],cmdnn$table[1])
True_positives = c(cmknn$table[4], cmrf$table[4],cmsvm$table[4],cmgb$table[4],cmnn$table[4],cmdnn$table[4])
False_negatives = c(cmknn$table[3], cmrf$table[3],cmsvm$table[3],cmgb$table[3],cmnn$table[3],cmdnn$table[3])
False_positives = c(cmknn$table[2], cmrf$table[2],cmsvm$table[2],cmgb$table[2],cmnn$table[2],cmdnn$table[2])

summarized=data.frame(Algorithms)
summarized$Accuracy=Accuracy
summarized$Kappa=Kappa
summarized$True_negatives=True_negatives
summarized$True_positives=True_positives
summarized$False_positives=False_positives
summarized$False_negatives=False_negatives
rownames(summarized)=NULL
colnames(summarized)=c("Algorithm","Accuracy","Kappa","True negatives", "True positives", "False positives", "False negatives")

knitr::kable(summarized)


### Cost-sensitive learning

cost.unit <- c(0, 1, 10, 0)
HealthCost <- function(data, lev = NULL, model = NULL) 
{
  y.pred = data$pred 
  y.true = data$obs
  CM = confusionMatrix(y.pred, y.true)$table
  out = sum(as.vector(CM)*cost.unit)/sum(CM)
  names(out) <- c("HealthCost")
  out
}
ctrl <- trainControl(method = "cv", number = 5,
                     classProbs = TRUE, 
                     summaryFunction = HealthCost,
                     verboseIter=T)
ctrl$sampling <- "up"


rf.train2 <- train(diagnosis ~., 
                   method = "rf", 
                   data = training,
                   preProcess = c("center", "scale"),
                   ntree = 200,
                   cutoff=c(0.7,0.3),
                   tuneGrid = expand.grid(mtry=c(6,8,10)), 
                   metric = "HealthCost",
                   maximize = F,
                   trControl = ctrl)





rfPred2 = predict(rf.train2, testing)
cmrf2 = confusionMatrix(rfPred2,testing$diagnosis)
knitr::kable(cmrf2[["table"]])
knitr::kable(cmrf2[["overall"]])
rfcost = HealthCost(data = data.frame(pred  = rfPred2, obs = testing$diagnosis))

#### KNN

knn.train2 = train(diagnosis ~ ., 
                   method = "knn", 
                   data = training,
                   metric= "HealthCost",
                   maximize = F,
                   preProcess = c("center", "scale"),
                   tuneGrid = expand.grid(k = c(1:20)),
                   trControl = ctrl)

knnPred2 = predict(knn.train2, testing)
cmknn2 = confusionMatrix(knnPred2,testing$diagnosis)



knitr::kable(cmknn2[["table"]])
knitr::kable(cmknn2[["overall"]])
knncost = HealthCost(data = data.frame(pred  = knnPred2, obs = testing$diagnosis))


#### SVM

svm.train2 = train(diagnosis ~., method = "svmRadial", 
                   data = training,
                   preProcess = c("center", "scale"),
                   tuneGrid = expand.grid(C = c(.25, .5, 1),
                                          sigma = c(0.01,.05)),
                   metric = "HealthCost",
                   trControl = ctrl)

svmPred2 = predict(svm.train2, testing)
cmsvm2 = confusionMatrix(svmPred2,testing$diagnosis)


knitr::kable(cmsvm2[["table"]])
knitr::kable(cmsvm2[["overall"]])
svmcost = HealthCost(data = data.frame(pred  = svmPred2, obs = testing$diagnosis))


#### Gradient Boosting


knitr::kable(cmgb[["table"]])
knitr::kable(cmgb[["overall"]])
gbmcost = HealthCost(data = data.frame(pred  =gbmPred, obs = testing$diagnosis))


#### Neural networks

nn.train2 <- train(diagnosis ~., 
                   method = "nnet",
                   data = training,
                   preProcess = c("center", "scale"),
                   MaxNWts = 1000,
                   maxit = 100,
                   tuneGrid = expand.grid(size=c(2,4,6), decay=c(0.01,0.001)), 
                   metric = "HealthCost",
                   maximize = F,
                   trControl = ctrl)


threshold = 0.5
nnProb2 = predict(nn.train, newdata=testing, type="prob")
nnPred2 = rep("B", nrow(testing))
nnPred2[which(nnProb2[,2] > threshold)] = "M"
cmnn2 = confusionMatrix(factor(nnPred), testing$diagnosis)
knitr::kable(cmnn2[["table"]])
knitr::kable(cmnn2[["overall"]])
nncost = HealthCost(data = data.frame(pred  = nnPred2, obs = testing$diagnosis))


### Cost sensitive learning results

library(ggplot2)

Algorithm = c("Random Forest","KNN", "SVM", "Gradient Boosting", "Neural Networks")
costs = c(rfcost,knncost,svmcost,gbmcost,nncost)
results=as.data.frame(Algorithm)
results$costs=costs
results=results[order(costs),]

ggplot(data=results)+aes(x=reorder(Algorithm, -costs), y=costs)+geom_col(fill="#f68060")+coord_flip()+xlab("Algorithms")+labs(title="Cost comparison")+
  scale_y_continuous(limits = c(0, 0.236))


### Final model creation

lrFit = train(diagnosis ~ ., 
              method = "glmnet",
              tuneGrid = expand.grid(alpha = seq(0, 1, 0.1), lambda = seq(0, .1, 0.02)),
              metric = "HealthCost",
              data = training,
              preProcess = c("center", "scale"),
              trControl = ctrl)

lrPred = predict(lrFit, testing)
cmlr= confusionMatrix(lrPred, testing$diagnosis)
knitr::kable(cmlr[["table"]])
knitr::kable(cmlr[["overall"]])


mode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}

#For each patient compute the most "voted" category
ensemble.pred = apply(data.frame(knnPred2, svmPred2, gbmPred,lrPred), 1, mode)
cmensemble = confusionMatrix(factor(ensemble.pred), testing$diagnosis)
knitr::kable(cmensemble[["table"]])
knitr::kable(cmensemble[["overall"]])


