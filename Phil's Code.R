#Group 1: Diabetes Dataset
#Members: Phil, Ron, Kelly, Jane

#Libraries used 
library(caret) #ML Model buidling package
library(tidyverse) #ggplot and dplyr
library(MASS) #Modern Applied Statistics with S
library(mlbench) #data sets from the UCI repository.
library(summarytools)
library(corrplot) #Correlation plot
library(gridExtra) #Multiple plot in single grip space
library(timeDate) 
library(pROC) #ROC
library(caTools) #AUC
library(rpart.plot) #CART Decision Tree
library(e1071) #imports graphics, grDevices, class, stats, methods, utils


#Pima Indians Diabetes Dataset Found Inside Caret Function
data(PimaIndiansDiabetes)# There are two of them, versions
df <- PimaIndiansDiabetes
df
str(df)

#Summary Statistics
summary(df)

#Confirmation of No Near Zero Variance for Predictor Variables
predictors <- PimaIndiansDiabetes[ , -(9)]
print(nearZeroVar(predictors))

#Check for missing values
#Confirmed No Missing Values
sapply(df, function(x) sum(is.na(x)))

#List Zero Markers: 6 out of 9 Variables have zero markers for Predictor Variables
list( Column = colSums(df==0), 
      Row = sum(rowSums(df==0)))

#Logic Behind 6 Zero Markers
#pregnant- not all woman have a baby, likely 0 is a true value, will keep predictor variable
#glucose- only 5 values are missing, will keep predictor variable, will use numerical mean
#pressure- only 35 values are missing, will keep predictor variable, will use numerical mean  
#triceps- approximately 30% of the data contains 0 values, will keep predictor variable, will use numerical mean
#insulin- almost 50% of the data has 0 values, will keep predictor variable, will use numerical mean
#mass- only 11 values are missing, will keep predictor variable

#Predictor Variables After Review of Summary Statistics and Zero Markers
#1.pregnant
#2.glucose
#3.pressure
#4.mass
#5.pedigree
#6.age
#7.triceps
#8.insulin

#Outcome Variable
#1.diabetes

#Replace All Zeros 
df[df == 0] <- NA
df

#Return Pregnant NA back to 0(zerO)
df$pregnant[is.na(df$pregnant)] <- 0
df

#Replace NA Values with Mean from respective columns: glucose, pressure, mass, insulin & triceps
df$glucose[is.na(df$glucose)]<-mean(df$glucose,na.rm=TRUE) #glucose
df$pressure[is.na(df$pressure)]<-mean(df$pressure,na.rm=TRUE) #pressure
df$mass[is.na(df$mass)]<-mean(df$mass,na.rm=TRUE) #mass
df$insulin[is.na(df$insulin)]<-mean(df$insulin,na.rm=TRUE) #insulin
df$triceps[is.na(df$triceps)]<-mean(df$triceps,na.rm=TRUE) #triceps

#Updated Summary Statistics After replacing NA Values with Mean from respective columns: glucose, pressure, & mass
summary(df)

#Histograms of Diabetes: Predictor Variables
n <-df[,1:8] #Predictors are variables 1-8
par(mfrow = c(3,3)) #Histograms will be 3x3
for (i in 1:ncol(n))
{hist(n[ ,i], xlab = names(n[i]), main = paste(names(n[i]), "Histogram"), col="orange")  
} 

#Correlation Plot of Diabetes: Predictor Variables
x <- cor(df[1:8])
corrplot(x, method="number")  

#Box Plots of Diabetes: Predictor Variables
boxplot(df$pregnant, main = "Pregnant Boxplot", col = "red")
boxplot(df$glucose, main = "Glucose Boxplot", col = "red")
boxplot(df$pressure, main = "Pressure Boxplot", col = "red")
boxplot(df$triceps, main = "Triceps Boxplot", col = "red")
boxplot(df$insulin, main = "Insulin Boxplot", col = "red")
boxplot(df$mass, main = "Mass Boxplot", col = "red")
boxplot(df$pedigree, main = "Pedigree Boxplot", col = "red")
boxplot(df$age, main = "Age Boxplot", col = "red")
 

#Split Training and Test Data, 80/20
set.seed(100)
split <- caret::createDataPartition(y = df$diabetes, times = 1, p = 0.8, list = FALSE)

#Train_data Split, 80%
train_data <- df[split,]

#Test_data Split, 20%
test_data <- df[-split,]

#Summary Statistics
summary(train_data)


##################Training Models########################## 
#Logistic Regression: Training Model
#No Tuning Parameters for Simple Logistic Regression
lr_train_data <- caret::train(diabetes ~., data = train_data,
                          method = "glm",
                          metric = "ROC",
                          tuneLength = 10,
                          trControl = trainControl(method = "cv", number = 10,
                                                   classProbs = T, summaryFunction = twoClassSummary),
                          preProcess = c("center","scale","pca"))

lr_train_data

#Random Forest: Training Model
rf_train_data <- caret::train(diabetes ~., data = train_data,
                             method = "ranger",
                             metric = "ROC",
                             trControl = trainControl(method = "cv", number = 10,
                                                      classProbs = T, summaryFunction = twoClassSummary),
                             preProcess = c("center","scale","pca"))
rf_train_data
plot(rf_train_data)



#K Nearest Neighbor: Training Model
knn_train_data <- caret::train(diabetes ~., data = train_data,
                          method = "knn",
                          metric = "ROC",
                          tuneGrid = expand.grid(.k = c(3:10)),
                          trControl = trainControl(method = "cv", number = 10,
                                                   classProbs = T, summaryFunction = twoClassSummary),
                          preProcess = c("center","scale","pca"))

knn_train_data

plot(knn_train_data) 


#Classification and Regression Trees (CART): Training Model
cart_train_data <- caret::train(diabetes ~., data = train_data,
                            method = "rpart",
                            metric = "ROC",
                            tuneLength = 20,
                            trControl = trainControl(method = "cv", number = 10,
                                                     classProbs = TRUE, summaryFunction = twoClassSummary),
                            preProcess = c("center","scale","pca"))

cart_train_data


#Compare ROC Value by Training Model
allmodels <- list(Logistic_Regression = lr_train_data, Random_Forest = rf_train_data, KNN = knn_train_data)
trainresults <- resamples(allmodels)

#Box Plot: Training Models' ROC Values
#Logistic Regression Performed Best on Training Data
bwplot(trainresults, metric="ROC")


###########################Test Data############################
#Logistic Regression: Testing Data
lrpredict <- predict(lr_train_data, test_data)
#Confusion Matrix Accuracy
lrconfusion <- confusionMatrix(lrpredict, test_data$diabetes, positive="pos")
lrconfusion

#Random Forest: Testing Data
rfpredict <- predict(rf_train_data, test_data)
#Confusion Matrix Accuracy
rfconfusion <- confusionMatrix(rfpredict, test_data$diabetes, positive="pos")
rfconfusion

#K Nearest Neighbor: Testing Data
knnpredict <- predict(knn_train_data, test_data)
#Confusion Matrix Accuracy
knnconfusion <- confusionMatrix(knnpredict, test_data$diabetes, positive="pos")
knnconfusion

#Classification and Regression Trees (CART): Testing Data
cartpredict <- predict(cart_train_data, test_data)
#Confusion Matrix Accuracy
cartconfusion <- confusionMatrix(cartpredict, test_data$diabetes, positive="pos")
cartconfusion


#Comparing Test Results
lrfinal<- c(lrconfusion$byClass['Sensitivity'], lrconfusion$byClass['Specificity'], lrconfusion$byClass['Precision'], 
            lrconfusion$byClass['Recall'], lrconfusion$byClass['F1'])
rffinal <- c(rfconfusion$byClass['Sensitivity'], rfconfusion$byClass['Specificity'], rfconfusion$byClass['Precision'], 
             rfconfusion$byClass['Recall'], rfconfusion$byClass['F1'])

knnfinal <- c(knnconfusion$byClass['Sensitivity'], knnconfusion$byClass['Specificity'], knnconfusion$byClass['Precision'], 
              knnconfusion$byClass['Recall'], knnconfusion$byClass['F1'])

cartfinal <- c(cartconfusion$byClass['Sensitivity'], cartconfusion$byClass['Specificity'], cartconfusion$byClass['Precision'], 
               cartconfusion$byClass['Recall'], cartconfusion$byClass['F1'])


allmodelsfinal <- data.frame(rbind(lrfinal, rffinal, knnfinal, cartfinal))
names(allmodelsfinal) <- c("Sensitivity", "Specificity", "Precision", "Recall", "F1")
allmodelsfinal 



 