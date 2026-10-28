#Classification project in Machine Statistical Learning

#Bank Marketing Classification Project


#------------------BANK MARKETING CLASSIFICATION PROJECT--------------------#
#Goa: predict whether SUBSCRIBED = yes or no


#First load packages
############################################################
#Install these packages once if you haven't already:
#install.packages(c("e1071", "randomForest", "corrplot", "caret", "ROCR", "pdp"))

library(e1071)         #for Naive Bayes
library(randomForest)  #for Random Forest modeling
library(corrplot)      #for correlation matrix visualization
library(caret)         #for confusionMatrix and evaluation metrics
library(ROCR)          #for ROC curves and AUC calculation
library(pdp)           #for Partial Dependence Plots




#2.LOAD DATASET
#Ensure bank.csv is located in the 'data' folder relative to your working directory
bank <- read.csv(file.choose())


#DATA CLEANING & PREPROCESSING

#Basic checks
dim(bank)
names(bank)
str(bank)

#############################
#Check for missing values
#############################

colSums(is.na(bank))
sum(is.na(bank)) #no missing values found

#Create a frequency table for the response variable
#This counts how many clients subscribed "yes" and how many did not subscribe "no"
table(bank$SUBSCRIBED)

#Create a proportion table for the response variable
#This shows the percentage/proportion of clients in each class
prop.table(table(bank$SUBSCRIBED))



################
#data cleaning
################


#we will fix the response first
bank$SUBSCRIBED<-factor(bank$SUBSCRIBED,levels=c("no","yes"))

#fix the variable otherproduct, because it has capital letters
bank$otherproduct<-tolower(bank$otherproduct)



#Check that the fix worked
table(bank$otherproduct)

#turn categorical variables into factors
bank$job<-as.factor(bank$job)
bank$marital<-as.factor(bank$marital)
bank$otherproduct<-as.factor(bank$otherproduct)
bank$education<-as.factor(bank$education)
bank$default<-as.factor(bank$default)
bank$housing<-as.factor(bank$housing)
bank$loan<-as.factor(bank$loan)
bank$contact<-as.factor(bank$contact)
bank$month<-factor(
  bank$month,
  levels=c("mar","apr","may","jun","jul",
           "aug","sep","oct","nov","dec"))
bank$day_of_week<-factor(bank$day_of_week,
                         levels=c("mon","tue","wed","thu","fri"))
bank$poutcome<-as.factor(bank$poutcome)

#turn integer variables into numeric variables
bank$age<-as.numeric(bank$age)
bank$duration<-as.numeric(bank$duration)
bank$campaign<-as.numeric(bank$campaign)
bank$pdays<-as.numeric(bank$pdays)
bank$previous<-as.numeric(bank$previous)


#create a categorical variable based on pdays

bank$newpdays<-ifelse(
  bank$pdays==999,
  "no_previous_contact",
  ifelse(
    bank$pdays<=10,
    "up_to_10_days",
    ifelse(
      bank$pdays>10 & bank$pdays<=21,
      "more_than10",
      "NA")))

bank$newpdays<-factor(bank$newpdays,
                      levels=c("no_previous_contact","up_to_10_days","more_than10","NA"))

#table of the new variable
table(bank$newpdays)
#########################################################


############################################################
# Check and remove duplicated observations
############################################################

duplicated.rows<-duplicated(bank)

number.duplicates<-sum(duplicated.rows)
percentage.duplicates<-mean(duplicated.rows)

number.duplicates
percentage.duplicates

bank[duplicated.rows,]

all.duplicated.rows<-duplicated(bank) |
  duplicated(bank,fromLast=TRUE)

bank[all.duplicated.rows,]

bank<-bank[!duplicated(bank),]

rownames(bank)<-NULL

sum(duplicated(bank))

dim(bank)


#######################
#Exploratory Analysis
#######################


#frequency table of the response
table.res<-table(bank$SUBSCRIBED)
table.res
#percentages of no and yes
prop.table(table.res)

#barplot of the response variable SUBSCRIBED
barplot(table.res,main="SUBSCRIBED variable",xlab="subscription status",
        ylab="number of clients")


#Boxplot for the numeric variables
boxplot(age~SUBSCRIBED,data=bank)
boxplot(duration~SUBSCRIBED,data=bank)
boxplot(campaign~SUBSCRIBED,data=bank)
#boxplot(pdays~SUBSCRIBED,data=bank) #not informative because of 999 special code number
boxplot(previous~SUBSCRIBED,data=bank)
boxplot(emp.var.rate~SUBSCRIBED,data=bank)
boxplot(cons.price.idx~SUBSCRIBED,data=bank)
boxplot(cons.conf.idx~SUBSCRIBED,data=bank)
boxplot(euribor3m~SUBSCRIBED,data=bank)
boxplot(nr.employed~SUBSCRIBED,data=bank)


###########################################
#Contingency tables for factor variables
###########################################

#job
table(bank$job, bank$SUBSCRIBED)
prop.table(table(bank$job, bank$SUBSCRIBED), 1)
#marital
table(bank$marital, bank$SUBSCRIBED)
prop.table(table(bank$marital, bank$SUBSCRIBED), 1)
#otherproduct
table(bank$otherproduct, bank$SUBSCRIBED)
prop.table(table(bank$otherproduct, bank$SUBSCRIBED), 1)
#education
table(bank$education, bank$SUBSCRIBED)
prop.table(table(bank$education, bank$SUBSCRIBED), 1)
#default
table(bank$default, bank$SUBSCRIBED)
prop.table(table(bank$default, bank$SUBSCRIBED), 1)
#housing
table(bank$housing, bank$SUBSCRIBED)
prop.table(table(bank$housing, bank$SUBSCRIBED), 1)
#loan
table(bank$loan, bank$SUBSCRIBED)
prop.table(table(bank$loan, bank$SUBSCRIBED), 1)
#contact
table(bank$contact, bank$SUBSCRIBED)
prop.table(table(bank$contact, bank$SUBSCRIBED), 1)
#month
table(bank$month, bank$SUBSCRIBED)
prop.table(table(bank$month, bank$SUBSCRIBED), 1)
#day_of_week
table(bank$day_of_week, bank$SUBSCRIBED)
prop.table(table(bank$day_of_week, bank$SUBSCRIBED), 1)
#poutcome
table(bank$poutcome, bank$SUBSCRIBED)
prop.table(table(bank$poutcome, bank$SUBSCRIBED), 1)
#newpdays
table(bank$newpdays,bank$SUBSCRIBED)
prop.table(table(bank$newpdays,bank$SUBSCRIBED),1)

#####################################################
#Check the correlation between the numeric variables
#####################################################

#Numeric variables for correlation analysis
cor.num.vars<-c(
  "age",
  "campaign",
  "previous",
  "emp.var.rate",
  "cons.price.idx",
  "cons.conf.idx",
  "euribor3m",
  "nr.employed")

cor.num.vars
datanum<-bank[,cor.num.vars]
summary(datanum)

#checking the correlation between variables
data_matrix<-data.matrix(datanum)
M<-cor(data_matrix)
round(M,2)
corrplot(M,method="number",number.cex = 0.65,tl.cex = 0.7)

#higly correlated variables-we put a threshold at 0.85
cor.limit<-0.85
M<-cor(data_matrix)
high.cor<-which(abs(M)>=cor.limit&abs(M)< 1,arr.ind=TRUE)
high.cor.table <- data.frame(var1 = rownames(M)[high.cor[,1]],
                             var2 = colnames(M)[high.cor[,2]],
                             correlation = M[high.cor])

#remove duplicate pairs
high.cor.table<-high.cor.table[high.cor.table$var1<high.cor.table$var2, ]
high.cor.table

#we will remove highly correlated values
#the variables that will be excluded are emp.var.rate and euribor3m

#Check if the dataset contains outliers
vars<-colnames(datanum)
Outliers<-NULL
for(i in vars){
  max<-quantile(bank[,i],0.75)+IQR(bank[,i])*1.5
  min<-quantile(bank[,i],0.25)-IQR(bank[,i])*1.5
  idx<-which(bank[,i]<min|bank[,i]>max)
  print(paste(i,length(idx),sep=":"))
  Outliers<-c(Outliers,idx)
}

#which clients are potential outliers
Outliers<-unique(Outliers)
length(Outliers)

#histograms for the numeric variables
par(mfrow=c(1,2))
hist(bank$campaign,breaks=30) #right skewed
hist(bank$duration,breaks=30) #right skewed
hist(bank$age,breaks=30)
hist(bank$previous,breaks=30)
hist(bank$emp.var.rate,breaks=30)
hist(bank$cons.price.idx,breaks=30)
hist(bank$cons.conf.idx,breaks=30)
hist(bank$euribor3m,breaks=30)
hist(bank$nr.employed,breaks=30)
par(mfrow=c(1,1))
#########################################
#Create the final dataset for the models
#########################################
#Main realistic problem:
#Predict before the phone call who is likely to subscribe.
#Therefore, we remove duration because it is only known after the call.

bank.before.call<-bank

#Remove duration because it is not known before the phone call.
#This avoids data leakage.
bank.before.call$duration<-NULL

#pdays was replaced by newpdays
bank.before.call$pdays<-NULL

############################################################
#Remove highly correlated economic variables
############################################################

#emp.var.rate, euribor3m and nr.employed are highly correlated.
#We keep nr.employed and remove emp.var.rate and euribor3m
#to reduce redundancy.

bank.before.call$emp.var.rate<-NULL
bank.before.call$euribor3m<-NULL


############################################################
#Modelling dataset
############################################################

bank.model<-bank.before.call
names(bank.model)
str(bank.model)




############################################################
# Final numeric candidate variables
############################################################

num.vars<-c(
  "age",
  "campaign",
  "previous",
  "cons.price.idx",
  "cons.conf.idx",
  "nr.employed")

num.vars
############################################################
# Final mixed candidate variables
############################################################

mixed.vars<-c(
  "age",
  "campaign",
  "previous",
  "cons.price.idx",
  "cons.conf.idx",
  "nr.employed",
  "job",
  "marital",
  "education",
  "default",
  "housing",
  "loan",
  "otherproduct",
  "contact",
  "month",
  "day_of_week",
  "poutcome",
  "newpdays")
mixed.vars

##############################
#Train/Validation/Test split
##############################

set.seed(100)

#Randomly shuffle the row numbers
ind<-sample(1:nrow(bank.model))

############################################################
#Define the number of observations in each dataset
############################################################

#60% for training
x1<-round(0.60*nrow(bank.model))

#20% for validation
x2<-round(0.20*nrow(bank.model))


############################################################
#Select the row numbers for each dataset
############################################################

trainobs<-ind[1:x1]

validobs<-ind[(x1+1):(x1+x2)]

testobs<-ind[(x1+x2+1):nrow(bank.model)]


############################################################
# Create training, validation and test datasets
############################################################

train.set<-bank.model[
  trainobs,
]

valid.set<-bank.model[
  validobs,
]

test.set<-bank.model[
  testobs,
]




#Check dataset dimensions
dim(train.set)
dim(valid.set)
dim(test.set)

#Check dataset proportions
nrow(train.set)/nrow(bank.model)
nrow(valid.set)/nrow(bank.model)
nrow(test.set)/nrow(bank.model)



#Check response proportions
prop.table(table(bank.model$SUBSCRIBED))

prop.table(table(train.set$SUBSCRIBED))

prop.table(table(valid.set$SUBSCRIBED))

prop.table(table(test.set$SUBSCRIBED))

##################################
#Check the sizes of the datasets
##################################

nrow(train.set)
nrow(valid.set)
nrow(test.set)

nrow(train.set)/nrow(bank.model)
nrow(valid.set)/nrow(bank.model)
nrow(test.set)/nrow(bank.model)


###########################################
#Check if yes/no proportions are similar
###########################################

prop.table(table(bank.model$SUBSCRIBED))
prop.table(table(train.set$SUBSCRIBED))
prop.table(table(valid.set$SUBSCRIBED))
prop.table(table(test.set$SUBSCRIBED))
#all proportions are similar

#####################
#Baseline accuracy
#####################

#Since the response is imbalanced, a naive model that always predicts
#the majority class "no" would already have high accuracy.

baseline.accuracy<-max(prop.table(table(test.set$SUBSCRIBED)))
baseline.accuracy
#Therefore, sensitivity, specificity, precision, F1 score, ROC curve and AUC 
#were also taken into consideration for the final models across methods


############################################################
#We will fit a simple Logistic regression model with 
#threshold of 0.5 as a baseline
#LOGISTIC REGRESSION
############################################################

#Fit a Logistic Regression model on the training set
logistic.model<-glm(SUBSCRIBED ~ ., data=train.set, family="binomial")

#Predict probabilities on the test set
logistic.prob<-predict(logistic.model, newdata=test.set, type="response")

#Classify using a basic threshold of 0.5
basic.log.pred<-ifelse(logistic.prob>0.5,"yes","no")
basic.log.pred<-factor(basic.log.pred,levels=levels(test.set$SUBSCRIBED))

#Confusion matrix and evaluation metrics
log.cm<-confusionMatrix(basic.log.pred,test.set$SUBSCRIBED,positive="yes",mode="everything")
log.cm


############################################################
#RANDOM FOREST
############################################################
set.seed(1)
traindata <- train.set
validationset <- valid.set
testdata <- test.set


#the loop to find the optimal number of trees (ntree) and number of variables (mtry)
set.seed(100)
oob<-NULL
for (ntree in c(50,100,200,300)) {
  for (mtry in c(2,3,4)) {
    
    myRF<- randomForest(SUBSCRIBED ~ ., data=traindata, ntree=ntree,mtry=mtry)
    oob<-c(oob,myRF$err.rate[ntree,1])
  }}

results<- matrix(oob,byrow=TRUE,4,3)
colnames(results)<- c(2,3,4)
rownames(results) <- c(50,100,200,300)
results

#the model that is obtained from the loop above:
#200 trees and 3 variables
set.seed(100)
myRF1<- randomForest(SUBSCRIBED~ ., data=traindata, ntree=300,
                     mtry=3, importance=TRUE)
myRF1

plot(myRF1)
myRF1$predicted
myRF1$importance
myRF1$err.rate
myRF1$votes


#to find the best threshold that will maximize the f1 score
predictprob<- predict(myRF1,testdata,type="prob")[,"yes"] 
thresholds<- seq(0.01, 0.99, by=0.01)
f1<- numeric()

set.seed(100)
for(t in thresholds){
  pred<- ifelse(predictprob>t,"yes","no")
  pred<- factor(pred,levels=levels(testdata$SUBSCRIBED))
  
  cm<- confusionMatrix(pred,testdata$SUBSCRIBED,positive="yes")
  
  f1<- c(f1,cm$byClass["F1"])}

bestthres<- thresholds[which.max(f1)]
bestthres

bestf1<- max(f1,na.rm = TRUE)
bestf1

#at this stage, i will compute the f1-score for this model
newf1<- ifelse(predictprob>bestthres,"yes","no")
newf1<- factor(newf1,levels=levels(testdata$SUBSCRIBED))

conf_matrix_f1<- confusionMatrix(
  newf1,
  testdata$SUBSCRIBED,
  positive="yes",
  mode="everything")

conf_matrix_f1$byClass[
  c("Sensitivity","Specificity","Recall","Precision","F1")]


#the next thing i want to do is shuffle the variables and check
#how many and which of them i can remove getting the smallest
#decrease in F1 score possible

#whenever i have a very small decrease in F1 score by removing
#a variable, i will choose to remove the variable, to obtain
#a more parsimonious model

#so first i shuffle the data
str(traindata)

vars <- c(
  "loan",
  "housing",
  "otherproduct",
  "previous",
  "age",
  "job",
  "marital",
  "education",
  "contact",
  "month",
  "day_of_week",
  "campaign",
  "poutcome",
  "cons.conf.idx",
  "nr.employed",
  "newpdays",
  "default",
  "cons.price.idx")

out<-numeric()

set.seed(100)
for(v in vars){
  shuffle<- testdata
  shuffle[[v]]<- sample(shuffle[[v]])
  
  pshuffle<- predict(myRF1,shuffle,type="prob")[,"yes"]
  
  predshuffle<- ifelse(pshuffle>bestthres,"yes","no")
  predshuffle<- factor(
    predshuffle,
    levels=levels(testdata$SUBSCRIBED))
  
  cm<- confusionMatrix(
    predshuffle,
    testdata$SUBSCRIBED,
    positive="yes",
    mode="everything")
  
  out<- c(out,cm$byClass["F1"])
}

outres<-data.frame(
  Variable=vars,
  F1=out)

outres[order(out, decreasing=TRUE),]


#i check the results and then compare to the VarImpPlot:
varImpPlot(myRF1,main="Baseline Variable Importance")


#Partial Dependence plot for the best model 
#for the most significant variables from the VarImpPlot



#for nr.employed
partialPlot(
  myRF1,
  pred.data=traindata,
  x.var="nr.employed",
  which.class="yes",
  main="Partial Dependence Plot for nr.employed",
  xlab="Number of Employees",
  ylab="Partial dependence")
#age
partialPlot(
  myRF1,
  pred.data=traindata,
  x.var="age",
  which.class="yes",
  main="Partial Dependence Plot for age",
  xlab="Age",
  ylab="Partial dependence")


#job
partialPlot(
  myRF1,
  pred.data=traindata,
  x.var="job",
  which.class="yes",
  main="Partial Dependence Plot for job",
  xlab="Job",
  ylab="Partial dependence")


#education
partialPlot(
  myRF1,
  pred.data=traindata,
  x.var="education",
  which.class="yes",
  main="Partial Dependence Plot for education",
  xlab="Education",
  ylab="Partial dependence")


#from both the training and the test datasets
traindata$loan<-NULL
testdata$loan<-NULL
validationset$loan<-NULL

traindata$default<-NULL
testdata$default<-NULL
validationset$default<-NULL


####### MODEL 2 ############

set.seed(100)
myRF2<- randomForest(SUBSCRIBED~ .,
                     data=traindata,
                     ntree=300,
                     mtry=3,
                     importance=TRUE)

myRF2

plot(myRF2)
myRF2$predicted
myRF2$importance
myRF2$err.rate
myRF2$votes

#to find the best threshold that will maximize the f1 score
predictprob2<- predict(
  myRF2,
  testdata,
  type="prob")[,"yes"] 

thresholds2<- seq(0.01, 0.99, by=0.01)
f12<- numeric()

set.seed(100)
for(t in thresholds2){
  pred<- ifelse(predictprob2>t,"yes","no")
  pred<- factor(pred,levels=levels(testdata$SUBSCRIBED))
  cm<- confusionMatrix(pred,testdata$SUBSCRIBED,positive="yes")
  
  f12<- c(f12,ifelse(is.na(cm$byClass["F1"]),0,cm$byClass["F1"]))}

bestthres2<- thresholds2[which.max(f12)]
bestthres2

bestf12<- max(f12,na.rm = TRUE)
bestf12

newf12<- ifelse(predictprob2>bestthres2,"yes","no")

newf12<- factor(newf12,
                levels=levels(testdata$SUBSCRIBED))

conf_matrix_f12<- confusionMatrix(
  newf12,
  testdata$SUBSCRIBED,
  positive="yes",
  mode="everything")

conf_matrix_f12$byClass[c("Sensitivity","Specificity","Recall","Precision","F1")]


table(traindata$SUBSCRIBED)

prop.table(table(traindata$SUBSCRIBED))

table(validationset$SUBSCRIBED)

prop.table(table(validationset$SUBSCRIBED))

table(testdata$SUBSCRIBED)

prop.table(table(testdata$SUBSCRIBED))



######COMPARISON OF THE TWO MODELS AND EVALUATION OF THE
######RESULTS THROUGH THE ROC CURVE AND THE CORRESPONDING AUC

#the curve of the 1st model and the plot

pred1<- prediction(
  predictprob,
  testdata$SUBSCRIBED)

perf1<- performance(
  pred1,
  "tpr",
  "fpr")

plot(
  perf1,
  col="hotpink4",
  lwd=2,
  main="ROC Curves")

abline(0,1,col="gray")


#the curve of the 2nd model and the plot
#overlaying the plot of the 1st model to compare

pred2<- prediction(
  predictprob2,
  testdata$SUBSCRIBED)

perf2<- performance(
  pred2,
  "tpr",
  "fpr")

plot(
  perf2,
  col="cadetblue",
  add=TRUE,
  lwd=2)

legend(
  "bottomright",
  legend=c("Model 1","Model 2"),
  col=c("hotpink4","cadetblue"),
  lwd=2)


#the corresponding areas under the curve
auc1<-performance(pred1,"auc")@y.values[[1]]
auc1

auc2<-performance(pred2,"auc")@y.values[[1]]
auc2

#the corresponding f1-score matrices
conf_matrix_f1$byClass[c("Sensitivity","Specificity","Recall","Precision","F1")]

conf_matrix_f12$byClass[c("Sensitivity","Specificity","Recall","Precision","F1")]

conf_matrix_f1$table
conf_matrix_f12$table

#model 2 did not predict any new yes, but it only detected
#some false yes, so this is why we prefer model 1 again





###################################################
#NAIVE BAYES with RFE variable selection method
#without laplace
###################################################

#Thresholds to examine
thresholds<-seq(0.05,0.50,by=0.01)


#function for fitting and evaluating one variable subset

evaluate.nb.subset<-function(variable.set,evaluation.set,
                             thresholds){
  
#Create the model formula
formula.text<-paste("SUBSCRIBED ~",paste(variable.set,collapse=" + "))
nb.formula<-as.formula(formula.text)
  
#Fit Naive Bayes using the training set
nb.model<-naiveBayes(nb.formula,data=train.set)
  
  
#Predict probabilities
nb.prob<-predict(nb.model,newdata=evaluation.set,type="raw")
nb.prob.yes<-nb.prob[,"yes"]
  
  
#Table for threshold results
threshold.results<-data.frame(threshold=thresholds,accuracy=NA,sensitivity=NA,
                                specificity=NA,precision=NA,F1=NA)
  
#Calculate metrics for every threshold
for(i in 1:length(thresholds)){
  current.threshold<-thresholds[i]
  predicted.class<-ifelse(
  nb.prob.yes>=current.threshold,"yes","no")
  predicted.class<-factor(predicted.class,levels=c("no","yes"))
  actual.class<-factor(evaluation.set$SUBSCRIBED,levels=c("no","yes"))
  cm<-table(Predicted=predicted.class,Actual=actual.class)
  TP<-cm["yes","yes"]
  FP<-cm["yes","no"]
  FN<-cm["no","yes"]
  TN<-cm["no","no"]
    
  #Accuracy
  accuracy<-(TP+TN)/sum(cm)
  
  #Sensitivity
  if((TP+FN)==0){
      sensitivity<-0
    }else{
     sensitivity<-TP/(TP+FN)
    }
    
  #Specificity
  if((TN+FP)==0){
      specificity<-0
    }else{
      specificity<-TN/(TN+FP)
    }
    
  #Precision
  if((TP+FP)==0){
      precision<-0
    }else{
      precision<-TP/(TP+FP)
    }
    
  #F1 score
  if((precision+sensitivity)==0){
      F1<-0
    }else{
      F1<-2*precision*sensitivity/
        (precision+sensitivity)}
    
  #Save the results
  threshold.results$accuracy[i]<-accuracy
  threshold.results$sensitivity[i]<-sensitivity
  threshold.results$specificity[i]<-specificity
  threshold.results$precision[i]<-precision
  threshold.results$F1[i]<-F1
  }
  
  
#Calculate AUC
actual.numeric<-ifelse(evaluation.set$SUBSCRIBED=="yes",1,0)
roc.prediction<-prediction(nb.prob.yes,actual.numeric)
auc.object<-performance(roc.prediction,"auc")
auc.value<-auc.object@y.values[[1]]
threshold.results$AUC<-auc.value
  
  
  
#Select threshold with highest F1
best.threshold.row<-which.max(threshold.results$F1)
best.result<-threshold.results[best.threshold.row,]
  
#Return results
return(list(model=nb.model,probabilities=nb.prob.yes,threshold.results=threshold.results,
            best.result=best.result))
}

#################################
#RFE USING THE TRAINING SET
#################################

#Start with all mixed candidate variables
current.variables<-mixed.vars


#Store every RFE variable subset
rfe.variable.subsets<-list()


#Store the detailed variable-removal results
rfe.removal.results<-list()


#Store the results for every RFE step
rfe.train.results<-data.frame(
  step=integer(),
  number.variables=integer(),
  variables=character(),
  variable.removed=character(),
  threshold=numeric(),
  accuracy=numeric(),
  sensitivity=numeric(),
  specificity=numeric(),
  precision=numeric(),
  F1=numeric(),
  AUC=numeric(),
  stringsAsFactors=FALSE)
step.number<-1



#Begin recursive feature elimination
repeat{
#Store the current variable subset
rfe.variable.subsets[[step.number]]<-current.variables
  
#Fit current subset and evaluate on train.set
  
  current.train.fit<-evaluate.nb.subset(
    variable.set=current.variables,
    evaluation.set=train.set,
    thresholds=thresholds)
  
current.train.result<-current.train.fit$best.result
  
  
  
#Save current training result
rfe.train.results<-rbind(rfe.train.results,
                          data.frame(step=step.number,number.variables=length(current.variables),
                                      variables=paste(current.variables,collapse=", "),
                                      variable.removed=NA_character_,
                                      threshold=current.train.result$threshold,
                                      accuracy=current.train.result$accuracy,
                                      sensitivity=current.train.result$sensitivity,
                                      specificity=current.train.result$specificity,
                                      precision=current.train.result$precision,
                                      F1=current.train.result$F1,
                                      AUC=current.train.result$AUC,
                                      stringsAsFactors=FALSE))
  
#Stop when only one variable remains
if(length(current.variables)==1){
  break
  }
  
#Try removing each variable
removal.results<-data.frame(variable.removed=character(),threshold=numeric(),
    accuracy=numeric(),
    sensitivity=numeric(),
    specificity=numeric(),
    precision=numeric(),
    F1=numeric(),
    AUC=numeric(),
    stringsAsFactors=FALSE)
  for(j in 1:length(current.variables)){
    variable.removed<-current.variables[j]
    candidate.variables<-current.variables[-j]
    
    
    
#Fit candidate subset on train.set
candidate.fit<-evaluate.nb.subset(variable.set=candidate.variables,
                                  evaluation.set=train.set,thresholds=thresholds)
    
candidate.result<-candidate.fit$best.result
    
    
#Store candidate result
removal.results<-rbind(removal.results,
                data.frame(variable.removed=variable.removed,
                                threshold=candidate.result$threshold,
                                accuracy=candidate.result$accuracy,
                                sensitivity=candidate.result$sensitivity,
                                specificity=candidate.result$specificity,
                                precision=candidate.result$precision,
                                F1=candidate.result$F1,
                                AUC=candidate.result$AUC,
                                stringsAsFactors=FALSE))}
  
#Rank possible removals
removal.results<-removal.results[order(removal.results$F1,removal.results$AUC,
                                       decreasing=TRUE),]
  
  
#Save removal results
rfe.removal.results[[step.number]]<-removal.results
  
  
#Remove the least useful variable
variable.to.remove<-removal.results$variable.removed[1]
rfe.train.results$variable.removed[nrow(rfe.train.results)]<-variable.to.remove
current.variables<-current.variables[current.variables!=variable.to.remove]
step.number<-step.number+1
}



#RFE training results
rfe.train.results


#View variable elimination order
rfe.train.results[,c("step","number.variables","variable.removed","F1","AUC")]


#test every rfe subset on the test set
rfe.test.results<-data.frame(
  subset.number=integer(),
  number.variables=integer(),
  variables=character(),
  best.threshold=numeric(),
  accuracy=numeric(),
  sensitivity=numeric(),
  specificity=numeric(),
  precision=numeric(),
  F1=numeric(),
  AUC=numeric(),
  stringsAsFactors=FALSE)


#Store every fitted Naive Bayes model
rfe.test.models<-list()

#Fit every subset on train.set and test on test.set
for(i in 1:length(rfe.variable.subsets)){
  current.subset<-rfe.variable.subsets[[i]]
  current.test.fit<-evaluate.nb.subset(
    variable.set=current.subset,
    evaluation.set=test.set,
    thresholds=thresholds)
  current.test.result<-current.test.fit$best.result
  
#Store the fitted model
rfe.test.models[[i]]<-current.test.fit
  
#Store test metrics
rfe.test.results<-rbind(rfe.test.results,
                          data.frame(subset.number=i,
                                     number.variables=length(current.subset),
                                     variables=paste(current.subset,collapse=", "),
                                     best.threshold=current.test.result$threshold,
                                     accuracy=current.test.result$accuracy,
                                     sensitivity=current.test.result$sensitivity,
                                     specificity=current.test.result$specificity,
                                     precision=current.test.result$precision,
                                     F1=current.test.result$F1,
                                     AUC=current.test.result$AUC,
                                     stringsAsFactors=FALSE))
}


#test results for all subsets
rfe.test.results

#Rank all subsets by test F1
rfe.test.results.ranked<-rfe.test.results[
  order(rfe.test.results$F1,rfe.test.results$AUC,decreasing=TRUE),]
rfe.test.results.ranked



#Plot test F1 against number of variables

plot(
  rfe.test.results$number.variables,
  rfe.test.results$F1,
  type="b",
  xlab="Number of variables",
  ylab="Test F1 score",
  main="Simple Naive Bayes RFE Test Results")

#Select the model with the highest test F1
best.test.F1<-max(
  rfe.test.results$F1)

best.test.rows<-which(
  rfe.test.results$F1==best.test.F1)


#Use AUC to resolve an F1 tie
best.test.row<-best.test.rows[which.max(rfe.test.results$AUC[best.test.rows])]


#Best test result
best.nb.test.result<-rfe.test.results[best.test.row,]
best.nb.test.result


#Selected variables
best.nb.variables<-rfe.variable.subsets[[best.test.row]]
best.nb.variables
length(best.nb.variables)

#Selected threshold
best.nb.threshold<-rfe.test.results$best.threshold[best.test.row]
best.nb.threshold

#Selected simple Naive Bayes model
best.nb.model<-rfe.test.models[[best.test.row]]$model
best.nb.model



############################################################
# FINAL VALIDATION STEP
############################################################


#we check the model that perfomed better on the validation set
#predict classes directly on the unseen validation set
nb.validation<-predict(best.nb.model,newdata=valid.set)

actual.val<-factor(valid.set$SUBSCRIBED,levels=c("no","yes"))

#confusion matrix and evaluation metrics on validation set
validation.cf<-confusionMatrix(nb.validation,actual.val,positive="yes",mode="everything")
validation.cf



