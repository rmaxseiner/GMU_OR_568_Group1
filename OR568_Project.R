## Libraries
library(mlbench)
library(summarytools)
library(corrplot)
library(gridExtra)
library(e1071)
library(caret)
library(AppliedPredictiveModeling)
library(randomForest)
library(rpart)
library(pROC)
library(gbm)
library(ipred)
library(kernlab)
library(nnet)

## Data Explotation
data("PimaIndiansDiabetes")
data("PimaIndiansDiabetes2") #seems to be the same dataset, but with NA instead of 0 for missing values
df <- PimaIndiansDiabetes
df2 <- PimaIndiansDiabetes2
str(PimaIndiansDiabetes)
?PimaIndiansDiabetes
head(df)
head(df2)
tail(df)
tail(df2)
PimaIndiansDiabetes
summary(df)
summary(df2)

#histograms of predictors(without missing values)
windows()
par(mfrow = c(3,3))
for( i in 1:8){
  hist(df2[,i], main = colnames(df2)[i],xlab = colnames(df2)[i], col = 'magenta3')
}
#pressure - some values seem unrealisticly low, 20-50? How should we treat those?
# age - most participants are between 20-30

#relationships between predictors
pairs(PimaIndiansDiabetes)
cor(PimaIndiansDiabetes[1:8]) 
corrplot(cor(PimaIndiansDiabetes[1:8]))
#   no significant correlation between any predictors

#Box plots
windows()
par(mfrow = c(3,3))
for( i in 1:8){
  boxplot(df2[,i], main = colnames(df2)[i],xlab = colnames(df2)[i], col = 'magenta3')
}
#the insulin boxplot shows the most outliers
#blood pressue also has several outliers

#skewness
skewness(PimaIndiansDiabetes$pregnant) #0.898
skewness(PimaIndiansDiabetes$glucose) #0.173
skewness(PimaIndiansDiabetes$pressure) #-1.836
skewness(PimaIndiansDiabetes$triceps) #0.109
skewness(PimaIndiansDiabetes$insulin) #2.263
skewness(PimaIndiansDiabetes$mass) #-0.427
skewness(PimaIndiansDiabetes$pedigree) #1.912
skewness(PimaIndiansDiabetes$age) #1.125

#find near zero var predictors
nearZeroVar(df2) #integer(0)

#count missing values
sapply(df2,function(x) sum(is.na(x)))
#pregnant  glucose pressure  triceps  insulin     mass pedigree      age diabetes 
#     0        5       35      227      374       11        0        0        0 

#addressing missing values: replace with mean
df2$glucose[is.na(df2$glucose)]<-mean(df2$glucose,na.rm=TRUE)
df2$pressure[is.na(df2$pressure)]<-mean(df2$pressure,na.rm=TRUE)
df2$mass[is.na(df2$mass)]<-mean(df2$mass,na.rm=TRUE)
df2$insulin[is.na(df2$insulin)]<-mean(df2$insulin,na.rm=TRUE)
df2$triceps[is.na(df2$triceps)]<-mean(df2$triceps,na.rm=TRUE) 
sapply(df2,function(x) sum(is.na(x))) #check that NAs are removed
#pregnant  glucose pressure  triceps  insulin     mass pedigree      age diabetes 
#      0        0        0        0        0        0        0        0        0

#check changes in summary statistics and relationships
summary(df2)
pairs(df2)
cor(df2[1:8]) 
corrplot(cor(df2[1:8]))
#still no highly correlated predictors

skewness(df2$pregnant) #0.898
skewness(df2$glucose) #0.531
skewness(df2$pressure) #-1.836
skewness(df2$triceps) #0.819
skewness(df2$insulin) #3.007
skewness(df2$mass) #0.596
skewness(df2$pedigree) #1.912
skewness(df2$age) #1.125
#skewness for predictors with more missing value became more centered, with the exception of insulin


#split x and y
x <- df2[,-9]
y <- df2[,9]

#without triceps
x2 <- x[,-4]

## Split training and test set
set.seed(27)
split <- createDataPartition(y = y, times = 1, p = 0.8, list = FALSE)
xTrain <- x2[split, ]
xTest <- x2[-split, ]
yTrain <- y[split]
yTest <- y[-split]
#Check distribution of pos and neg values in train and test sets
summary(yTrain)
summary(yTest)

######################### Logistic Regression ######################### with triceps

## model performed the same with and without triceps

#set control
ctrl <- trainControl(summaryFunction = twoClassSummary,
                     classProbs = TRUE)

set.seed(27)
logTrain <- train(x = xTrain,
                 y = yTrain,
                 method = "glm",
                 metric = "ROC",
                 preProc = c("center", "scale", "pca"),
                 trControl = ctrl)
logTrain
#  ROC        Sens       Spec     
#  0.8338129  0.8709089  0.5810162

summary(logTrain)
#Coefficients:
#              Estimate  Std. Error z value Pr(>|z|)    
#(Intercept)   -0.85545    0.10860  -7.877 3.36e-15 ***
#  PC1         -0.93528    0.08644 -10.821  < 2e-16 ***
#  PC2         -0.05325    0.08297  -0.642  0.52097    
#  PC3          0.33450    0.09467   3.533  0.00041 ***
#  PC4          0.25352    0.10923   2.321  0.02029 *  
#  PC5         -0.03491    0.11920  -0.293  0.76965    
#  PC6          0.82328    0.14165   5.812 6.17e-09 ***
#  PC7         -0.39313    0.15076  -2.608  0.00912 **   

#Try test set
logPred <- predict(logTrain,xTest)
confusionMatrix(data = logPred, reference = yTest)
#Confusion Matrix and Statistics
#         Reference
#Prediction neg pos
#       neg  86  25
#       pos  14  28
#Accuracy : 0.7451         
#95% CI : (0.6684, 0.812)
#No Information Rate : 0.6536         
#P-Value [Acc > NIR] : 0.009661       
#Kappa : 0.4082         
#Mcnemar's Test P-Value : 0.109315       
#            Sensitivity : 0.8600         
#            Specificity : 0.5283         
#         Pos Pred Value : 0.7748         
#         Neg Pred Value : 0.6667         
#             Prevalence : 0.6536         
#         Detection Rate : 0.5621         
#   Detection Prevalence : 0.7255         
#      Balanced Accuracy : 0.6942     

#check variable importance (without PCA first)
varImp(logTrain)
#         Overall
#glucose  100.000
#mass      52.107
#pregnant  42.396
#pedigree  34.112
#age       16.153
#pressure   3.942
#insulin    1.806
#triceps    0.000

# varImp with PCA:
#     Overall
# PC1 100.000
# PC6  52.425
# PC3  30.780
# PC7  21.988
# PC4  19.265
# PC2   3.315
# PC5   0.000
 

######################### Random Forest ######################### without triceps

## performed slightly better without triceps

ctrl <- trainControl(method = "LGOCV",
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE,
                     savePredictions = TRUE)
mtryValues <- c(1:8)
set.seed(27)
rfFit <- train(x = xTrain, 
               y = yTrain,
               method = "rf",
               ntree = 500,
               tuneGrid = data.frame(mtry = mtryValues),
               preProc = c("center", "scale"), ##the model performed slightly better without PCA
               importance = TRUE,
               metric = "ROC",
               trControl = ctrl)
rfFit
# mtry  ROC        Sens    Spec     
# 2     0.8309283  0.8528  0.6045283
#ROC was used to select the optimal model using the largest value.
#The final value used for the model was mtry = 2.

rfPred = predict(rfFit, xTest)
rfPR = postResample(pred=rfPred, obs=yTest)

rfPR
# Accuracy     Kappa 
# 0.7647059 0.4803774


######################### K Nearest Neighbor ######################### with triceps

set.seed(27)
knnTune <- train(xTrain,
                 yTrain,
                 method = "knn",
                 preProc = c("center", "scale", "pca"),
                 tuneGrid = data.frame(.k = 1:25),
                 trControl = trainControl(method = "cv"))
knnTune

#  k   Accuracy   Kappa    
# 21  0.7674775  0.4629227
# Accuracy was used to select the optimal model using the largest value.
# The final value used for the model was k = 21.

knnPred = predict(knnTune, xTest)
knnPR = postResample(pred=knnPred, obs=yTest)

knnPR
# Accuracy     Kappa 
# 0.7647059   0.4612676 



######################### Boosted Trees ######################### without triceps

gbmGrid <- expand.grid(.interaction.depth = seq(1, 7, by = 2),
                       .n.trees = seq(100, 1000, by = 50), 
                       .shrinkage = c(0.01, 0.1),
                       .n.minobsinnode = 10)

gbmTune <- train(x = xTrain, y = yTrain,
                 method = "gbm",
                 tuneGrid = gbmGrid,
                 preProcess = c("center","scale"),##the model performed better without PCA
                 verbose = FALSE, 
                 trControl = trainControl(method = "cv", number = 10,
                                          classProbs = TRUE, summaryFunction = twoClassSummary))

gbmTune

gbmPred = predict(gbmTune,n.trees = 200, newdata = xTest)
gbmPR = postResample(pred=gbmPred, obs=yTest)
gbmPR
# Accuracy     Kappa 
# 0.7581699 0.4488365  

######################### Neural Networks  ######################### without triceps

ctrl <- trainControl(method = "cv", number = 10)

nnetGrid <- expand.grid(.decay = c(0, 0.01, 0.1),
                        .size = c(1:10))
set.seed(27)
nnetTune <- train(xTrain, yTrain,
                  method = "nnet",
                  tuneGrid = nnetGrid,
                  trControl = ctrl,
                  preProc = c("center", "scale"),
                  linout = FALSE,
                  trace = FALSE,
                  MaxNWts = 10*(ncol(xTrain)+ 1) + 10 + 1,
                  maxit = 500)
nnetTune

nnPred = predict(nnetTune, newdata=xTest)
nnPR = postResample(pred=nnPred, obs=yTest)

nnPR
# Accuracy     Kappa 
# 0.7908497 0.5297733 

######## Averaged ######## without tricps

avGrid <- expand.grid(.decay = c(0, 0.01, 0.1),
                      .size = c(1:10),
                      .bag = FALSE)


ctrl <- trainControl(method = "cv", number = 10)

set.seed(27)
avTune <- train(xTrain, yTrain,
                method = "avNNet",
                tuneGrid = avGrid,
                trControl = ctrl,
                preProc = c("center", "scale"),
                linout = FALSE,
                trace = FALSE,
                MaxNWts = 10*(ncol(xTrain)+ 1) + 10 + 1,
                maxit = 500)
avTune


avPred.tuned = predict(avTune, newdata=xTest)
avPR.tuned = postResample(pred=avPred.tuned, obs=yTest)

avPR.tuned
#  Accuracy     Kappa 
#  0.7908497  0.5297733 

#performed the same as the unaveraged nnet
