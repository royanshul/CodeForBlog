setwd("C:/Users/Anshul.Roy/Desktop/Personal/Linear Regression Datasets/HousePrice_Prediction_Linear Regresion")
library(corrplot)
library(leaps)
library(ggplot2)

house<-read.csv("kc_house_data.csv",header= TRUE)
str(house) #All are numeric/integer fields.
View(house) #Observe, date column has values with T. We should pick only the date and ignore the values after T.
house$date <- as.character(house$date)
house$date_new <- substr(house$date,1,8)
house$month <- substr(house$date_new,5,6)
house$year <- substr(house$date_new,1,4)
house$day <- substr(house$date_new,7,8)
house <- house[,-c(1,2)]
sum(duplicated(house)) #Find if any duplicate exist.
sum(is.na(house))
lapply(house,function(x){unique(x)})
#Remove Date Variable
house<- house[,-20]
house$day <- as.integer(house$day)
house$year <- as.integer(house$year)
house$month <- as.integer(house$month)
house$waterfront <- as.character(house$waterfront)
house$view <- as.character(house$view)
house$condition <- as.character(house$condition)
house$floors <- as.character(house$floors)
#Dummy Variable creation
dummy <- data.frame(model.matrix(~waterfront-1,house)[,-1])
dummy1 <- data.frame(model.matrix(~view-1,house)[,-1])
dummy2 <- data.frame(model.matrix(~condition-1,house)[,-1])
dummy3 <- data.frame(model.matrix(~floors-1,house)[,-1])

house <- cbind(house,dummy,dummy1,dummy2,dummy3)
house <- house [,-c(6,7,8,9)] #remove the variable converted to dummy
remove(dummy,dummy1,dummy2,dummy3)



#Split the data into train & test
set.seed(100)
#dataframe <-data.frame(scale(dataframe))
train.index <- sample(1:nrow(house),0.7*nrow(house),replace=FALSE)
train <- house [train.index,]
test <- house[-train.index,]

#Model preparation

model_1 <- lm(price~.,data=train)
summary(model_1)
step <- stepAIC(model_1,direction="both")
step
model_2 <- lm(price ~ bedrooms + bathrooms + sqft_living + sqft_lot + 
                grade + sqft_above + yr_built + yr_renovated + zipcode + 
                lat + long + sqft_living15 + sqft_lot15 + year + day + model.matrix..waterfront...1..house.....1. + 
                view1 + view2 + view3 + view4 + condition4 + condition5 + 
                floors2 + floors2.5 + floors3 + floors3.5, data = train)
summary(model_2)
SSE3 <- sum(model_2$residuals^2)
RMSE3 <- sqrt(SSE3/nrow(train))
vif(model_2) #sqft_above is highly correlated. However, its highly significant. Hence, we cannot proceed further and this is our Model.
#Lets verify the created model using different verifying tools.


train_1 <- train[-1] #Remove the price which we need to find
sub.fit = regsubsets(price ~ bedrooms + bathrooms + sqft_living + sqft_lot + 
                       grade + sqft_above + yr_built + yr_renovated + zipcode + 
                       lat + long + sqft_living15 + sqft_lot15 + year + day + model.matrix..waterfront...1..house.....1. + 
                       view1 + view2 + view3 + view4 + condition4 + condition5 + 
                       floors2 + floors2.5 + floors3 + floors3.5, data = train)
best.summary <- summary(sub.fit)
names(best.summary)
which.min(best.summary$rss) #This suggest that Model with 8 Features have smalles RSS.


#Additional Code just for further evaluation. Many evaluation technique exists and evaluation of the model can be done in many ways. 
#The below is one check.

#THe below functions are from leaps and are used to find Mallow's Cp. 
par(mfrow=c(1,2))
plot(best.summary$cp, xlab="number of features", ylab="cp")
plot(sub.fit, scale="Cp")
# In the plot on the left-hand side, the model with three features has the lowest cp. The plot
# on the right-hand side displays those features that provide the lowest Cp. The way to read
# this plot is to select the lowest Cp value at the top of the y axis. Then, move
# to the right and look at the colored blocks corresponding to the x axis. 
# By using the which.min() and which.max() functions, we can identify how cp compares 
# to BIC and the adjusted R-squared.
which.min(best.summary$bic) #8
which.max(best.summary$adjr2) #8 
# Hence all these shows that for 8 Variables, we have best bic, adjusted r-square and cp.
best.fit = lm(price ~ bedrooms + bathrooms + sqft_living  + model.matrix..waterfront...1..house.....1. + view4 +  grade +  yr_built + 
                 lat , data=train)
#Created Model using the best 8 variables.
summary(best.fit)

# With the Eight-feature model, F-statistic and all the t-tests have significant p-values.
# Having passed the first test, we can produce our diagnostic plots:
par(mfrow=c(2,2))
plot(best.fit)
vif(best.fit)
#Vif for sqft_living is high ~4. Lets remove and check.

best.7 = lm(price ~ bedrooms + bathrooms +  waterfront + view +  grade +  yr_built + lat , data=train)
summary(best.7)
par(mfrow=c(2,2))
plot(best.7)
vif(best.7)


# Instead of the plot of fits versus residuals is of concern, we can formally test the assumption 
# of the constant variance of errors in R. This test is known as the Breusch-Pagan (BP) test. F
# or this, we need to load the lmtest package, and run one line of code. The BP test has the null 
# hypotheses that the error variances are zero versus the alternative of not zero.
install.packages("lmtest")
library(lmtest)
bptest(best.7)

#Because p-value is very small, we should continue with  variable model i.e. best.fit.

#Final Model is Model with 8 Variables and Adjusted R-Square of 0.70
#----------------------------------------------------------
#Predict the Model accuracy
Predict <- predict(best.fit,test[,-1]) 
SSE_final = sum((Predict-test$price)^2)
SST = sum((mean(train$price)-test$price)^2)
R2 = 1-(SSE/SST)  #Will give R-Square Value for the test data. We can also check that from Confusion Matrix & Cross-Validation.
(cor(test$price,Predict)) ^2  #82.956% Accuracy.
summary(Predict)

#Note the model performance is only 70% which is Ok but not very good. We can increase the performance
#by binning,outlier treatment, dummy variable creation etc which are part of feature engineering and 
#extraction.

