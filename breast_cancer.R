##########################################
### Breast cancer Study
#   used for tumor classification 
##########################################

# AUTHOR: DANIEL LAPIDO MARTÍNEZ

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


rm(list = ls())

library(tidyverse)
library(klaR)
library(skimr)
library(mice)
library(VIM)
library(GGally)
library(MASS)
library(glmnet)
library(e1071) 
library(rpart)
library(pROC)
library(class)
library(randomForest)
library(caret)
library(verification)
library(dplyr)
library(tidyr)
library(MASS)
library(tidyverse)
library(ggcorrplot)
library(ggpubr)
library(readr)
library(car)
library("corrplot")
library(gridExtra)
library(kernlab)

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

#-----------------------------------------------------------
# MODEL FOR EXPLAINING THE MAIN PREDICTORS AFFECTING THE OUTPUT


# Missing-values
md.pattern(cancer)
#There are no missing-values

#Outliers
melt.cancer1 = reshape2::melt(cancer[c(1,2:11)],id.var = "diagnosis")
melt.cancer2 = reshape2::melt(cancer[c(1,12:21)],id.var = "diagnosis")
melt.cancer3 = reshape2::melt(cancer[c(1,22:31)],id.var = "diagnosis")

ggplot(data = melt.cancer1, aes(x = variable, y=value)) + 
  geom_boxplot(aes(fill=diagnosis)) + 
  facet_wrap(~variable, scales="free")+theme(strip.text = element_text(size=8))+theme(axis.title=element_blank())
ggplot(data = melt.cancer2, aes(x = variable, y=value)) + 
  geom_boxplot(aes(fill=diagnosis)) + 
  facet_wrap(~variable, scales="free")+theme(strip.text = element_text(size=8))+theme(axis.title=element_blank())
ggplot(data = melt.cancer3, aes(x = variable, y=value)) + 
  geom_boxplot(aes(fill=diagnosis)) + 
  facet_wrap(~variable, scales="free")+theme(strip.text = element_text(size=8))+theme(axis.title=element_blank())


#Proportion of benign and malign
table(cancer$diagnosis)/length(cancer$diagnosis)
freq=summary(cancer$diagnosis)/nrow(cancer)

ggplot()+aes(x=c("Benign", "Malign"),fill="blue", y=freq)+ geom_bar(stat="identity")+xlab("diagnosis")+
  guides(fill=FALSE)+ylab("frecuency")

#density distributions
ggplot(data = melt.cancer1, aes(x = value, fill=diagnosis)) + 
  geom_density(alpha=0.7, size=0.5) + 
  facet_wrap(~variable, scales="free")+theme(strip.text = element_text(size=8))+ theme(axis.text.x = element_text(angle = 90, hjust = 1))

#We can see that some variables are better for classification than others.
# For example radius, perimeter, area and concavity mean classify reasonably well
# whereas smoothness_mean and fractal dimension mean do not look appropiate at all. 

ggplot(data = melt.cancer2, aes(x = value, fill=diagnosis)) + 
  geom_density(alpha=0.7, size=0.5) + 
  facet_wrap(~variable, scales="free")+theme(strip.text = element_text(size=8))+ theme(axis.text.x = element_text(angle = 90, hjust = 1))

#Regarding standard error distributions, we can see that perimeter_se
#and area_se could be useful for classification, while texture_se and
#smoothness_se would be horrible.

ggplot(data = melt.cancer3, aes(x = value, fill=diagnosis)) + 
  geom_density(alpha=0.7, size=0.5) + 
  facet_wrap(~variable, scales="free")+theme(strip.text = element_text(size=8))+ theme(axis.text.x = element_text(angle = 90, hjust = 1))

#Finally, we can see in the plot above that radius, perimeter, area
# and concave_points worst are good for classification.


#We have now an idea of which variables look good for classification,
#however, it is possible that the dataset contains redundant information 
# about the response because some of these predictors may be correlated.
#If that is the case, perhaps we could drop some of the predictors, as 
#having more variables is not necessary better.  
# too many predictors will add unwanted noise to the problem
#and will probably invalid the explanatory purpose of the model. 


#CORRELATED VARIABLES
cancercor = cancer[-1]
corrplot(cor(cancercor), method = "pie", tl.cex=0.7)

correlationMatrix = cor(cancercor)
index=findCorrelation(correlationMatrix, cutoff=0.8)

#The most correlated variables are:
colnames(cancercor)[index]


#There are indeed lots of correlated variables. Which ones do we drop?

#By looking at the previous plots, we select the variables that better classify 
# and drop those correlated to that variable:
reduced_cancer=cancer
drops = c('perimeter_mean','radius_mean','compactness_mean','concave points_mean','radius_se','perimeter_se','radius_worst','perimeter_worst','compactness_worst','concave points_worst','compactness_se','concave points_se','texture_worst','area_worst')
reduced_cancer = reduced_cancer[ , !(names(reduced_cancer) %in% drops)]
levels(reduced_cancer$diagnosis)=c("B", "M")


#---------------------------------------------------

### MODELS FOR PREDICTING THE OUTPUT ###

### Probabilistic-learning models 

# Split dataset in train and test

in_train = createDataPartition(reduced_cancer$diagnosis, p = 0.7, list = FALSE)  # 70% for training

#Sets for the reduced dataset
training = reduced_cancer[ in_train,]
testing = reduced_cancer[-in_train,]

#Sets for the dataset with all variables
trainingfull = cancer[ in_train,]
testingfull = cancer[-in_train,]

#Notice that the observations used for training and testing are the 
#same for both datsets. The only difference is that one contains 
#more variables than the other.

# Logistic regression

#First we use logistic regression for the reduced dataset
logit.model = glm(diagnosis ~ ., family=binomial(link='logit'), data=training)
summary(logit.model)

# make predictions (posterior probabilities)
probability = predict(logit.model,newdata=testing, type='response')
head(probability)
prediction = as.factor(ifelse(probability > 0.5,"M","B"))
head(prediction)

# Performance: confusion matrix
confusionMatrix(prediction, testing$diagnosis)

#We get very good results. However we will perform 5-fold cross validation
# just to be sure about the results

#5-FOLD CROSS VALIDATION
ctrl = trainControl(method = "cv", number = 5,
                    classProbs = TRUE, 
                    verboseIter=T)
logistic_regression = train(diagnosis ~ ., 
              method = "glm",
              metric = "Kappa",
              data = training,
              family = binomial,
              trControl = ctrl)

lrPred = predict(logistic_regression, testing)
confusionMatrix(lrPred, testing$diagnosis)
#The results are as good as before.

#Lets see how good would be the results if we had not removed correlated variables:
logit.modelfull = glm(diagnosis ~ ., family=binomial(link='logit'), data=trainingfull)

probability = predict(logit.modelfull,newdata=testingfull, type='response')
prediction = as.factor(ifelse(probability > 0.5,"M","B"))
confusionMatrix(prediction, testingfull$diagnosis)

# We get a warning that the algorithm did not converge

# For this last case we could use penalized logistic regression:
ctrl = trainControl(method = "cv", number = 5,
                     classProbs = TRUE, 
                     verboseIter=T)
lrFit = train(diagnosis ~ ., 
               method = "glmnet",
               tuneGrid = expand.grid(alpha = seq(0, 1, 0.1), lambda = seq(0, .1, 0.02)),
               metric = "Kappa",
               data = trainingfull,
               preProcess = c("center", "scale"),
               trControl = ctrl)

plot(lrFit, scales = list(x = list(rot = 90)))
lrPred = predict(lrFit, testingfull)
confusionMatrix(lrPred, testingfull$diagnosis)


#We are more concerned about false negatives than false positives
# We might want to change the threshold = 0.5

#It is better to decrease the element (2,1) at the cost of increasing the (1,2)
# However changing the threshold does not give positive results.

### ROC curve

# ROC curve shows true positives vs false positives in relation with different thresholds

lrProb = predict(logistic_regression, testing, type="prob")

plot.roc(testing$diagnosis, lrProb[,2],col="darkblue", print.auc = TRUE,  auc.polygon=TRUE, grid=c(0.1, 0.2),
         grid.col=c("green", "red"), max.auc.polygon=TRUE,
         auc.polygon.col="lightblue", print.thres=TRUE)

### LDA

lda.model = lda(diagnosis ~ ., data=training, prior = c(.8, .2))
probability = predict(lda.model, newdata=testing)$posterior

threshold = 0.55

diag.pred = rep("B", nrow(testing))
diag.pred[which(probability[,2] > threshold)] = "M"

# Produce a confusion matrix
confusionMatrix(factor(diag.pred), testing$diagnosis)

#Accuracy is worse than with logistic regression, but the amount
# of false negatives is small.

# Naive bayes

#It is very fast but suffers a lot from collinearity
grid <- data.frame(usekernel = c(TRUE, FALSE),fL = 0:5,adjust = seq(0, 5, by = 1))
fit_nb <- train(diagnosis~.,data = training, method = "nb", trControl =  ctrl,metric = "Accuracy",importance = TRUE, tuneGrid=grid)
nbPred = predict(fit_nb, testing)
confusionMatrix(nbPred,testing$diagnosis)

plot(fit_nb, scales = list(x = list(rot = 90)))


## KNN

#kNN requires variables to be normalized or scaled, so we center and scale the data.

ctrl = trainControl(method="repeatedcv",repeats = 5)

knnFit = train(diagnosis ~ ., 
                method = "knn", 
                data = training,
                preProcess = c("center", "scale"),
                tuneLength = 3, #This number should be increased
                trControl = ctrl)

knnPred = predict(knnFit, testing)
confusionMatrix(knnPred,testing$diagnosis)

#Performs very well


## SVM

svmFit = train(diagnosis ~., method = "svmRadial", 
                data = training,
                preProcess = c("center", "scale"),
                tuneGrid = expand.grid(C = c(.25, .5, 1),
                                       sigma = c(0.01,.05)), 
                trControl = ctrl)

svmPred = predict(svmFit, testing)
confusionMatrix(svmPred,testing$diagnosis)

#Exceptional performance and no assumptions needed of the data.
