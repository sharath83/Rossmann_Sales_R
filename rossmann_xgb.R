
# ROSSMANN sales prediction - XGB2
#-------------------------------------------
setwd('/Users/homw/Documents/petp/Rossman/')
rm(list=ls())
library(lubridate)
library(randomForest)
library(caret)
library(readr)
library(xgboost)
library(dplyr)

#Preprocessing function ----------
clean <- function(df){
  #Parsing and conversion of dates to factors
  #fix date
  date <- strptime(df$Date, format = '%Y-%m-%d')
  df$Month <- month(date)
  df$Year <- year(date)
  df$Day <- day(date)
  df$weekday <- weekdays(date)
  df$week <- week(date)
  df$WeeksSinceOp <- (df$Year-2012)*52 + df$week
  #plot(df$Day, df$Sales, cex=0.3, pch =20)
  
  #Fix competition
  max.compdist <- max(df$CompetitionDistance, na.rm=T)
  df[is.na(df$CompetitionDistance),'CompetitionDistance'] <- 3*max.compdist
  df[is.na(df$CompetitionOpenSinceMonth),c('CompetitionOpenSinceMonth',"CompetitionOpenSinceYear")] <- 0
  df$CompMonths <- (df$Year-df$CompetitionOpenSinceYear)*12+df$Month-df$CompetitionOpenSinceMonth
  df <- df[,!(names(df) %in% c('CompetitionOpenSinceMonth',"CompetitionOpenSinceYear"))]
  
  #Fix Promo2
  #No. of weeks since Promo2 is running
  df[is.na(df$Promo2SinceWeek),c('Promo2SinceWeek',"Promo2SinceYear")] <- 0
  df$PromoSinceWeeks <- (df$Year - df$Promo2SinceYear)+df$week-df$Promo2SinceWeek
  
  #If Promo2 is a new campaign in the particular month
  mon <- lapply(df$PromoInterval,function(x) substr(x,1,3))
  mon <- unlist(mon)
  df[(mon ==""),"Promo2New"] <- "none"
  df[(mon=="Jan" & as.factor(df[["Month"]]) %in% c(1,4,7,10) |
        (mon=="Feb" & as.factor(df[["Month"]]) %in% c(2,5,8,11)) |
        (mon=="Mar" & as.factor(df[["Month"]]) %in% c(3,6,9,12))),"Promo2New"] <- "new"
  df[is.na(df$Promo2New),"Promo2New"] <- "old"
  
  #Now remove promo2 and competition original variables
  df <- df[,!(names(df) %in% c('CompetitionOpenSinceMonth',"CompetitionOpenSinceYear",
                               'Promo2SinceWeek',
                               'Promo2SinceYear',
                               'PromoInterval'))]
  #sum(is.na(df))
  #summary(as.factor((df$Promo2New)))
  df$Month <- as.factor(df$Month)
  df$Year <- as.factor(df$Year)
  df$Day <- as.factor(df$Day)
  df$weekday <- as.factor(df$weekday)
  
  df$SchoolHoliday = as.factor(df$SchoolHoliday)
  df$DayOfWeek = as.factor(df$DayOfWeek)
  df$Open = as.factor(df$Open)
  df$Promo = as.factor(df$Promo)
  df$Promo2 = as.factor(df$Promo2)
  df$Promo2New = as.factor(df$Promo2New)
  
  return(df)
}

#------------------ main code -------
#Raw files
train <- read.csv('train.csv', header = T)
test <- read.csv('test.csv', header = T)
stores <- read.csv('store.csv', header = T)
train <- merge(train, stores)
test <- merge(test, stores)
test[is.na(test$Open),'Open'] <- 0 # Consider 622 is close

#Clean train
train <- clean(train)
#Clean test
test <- clean(test)

# #Processed and saved files. Start from here if saved the above files
# train <- read.csv('train_pr.csv', header = T)
# test <- read.csv('test_pr.csv', header = T)

sub <- as.data.frame(test[,2]) # take Id out of test
names(sub)[1] <- "Id"
test <- test[,-2]

train <- train[,!(names(train) %in% c("Date","weekday"))]
test <- test[,!(names(test) %in% c("Date", "weekday"))]


# Make all numeric for convenience
for (f in names(train)) {
  if (class(train[[f]])=="factor") {
    train[[f]] <- as.character(train[[f]])
    levels <- unique(c(train[[f]], test[[f]]))
    train[[f]] <- as.integer(factor(train[[f]], levels=levels))
    test[[f]]  <- as.integer(factor(test[[f]],  levels=levels))
  }
}

train <- as.data.frame(lapply(train, as.numeric))
test <- as.data.frame(lapply(test, as.numeric))
sum(is.na(train))
summary(train)

s <- sample(nrow(train),100000)
dval<-xgb.DMatrix(data=data.matrix(train[s,-c(3,4)]),label=log(train$Sales[s]+1))
dtrain<-xgb.DMatrix(data=data.matrix(train[-s,-c(3,4)]),label=log(train$Sales[-s]+1))
watchlist<-list(val=dval,train=dtrain)
param <- list(  objective           = "reg:linear", 
                booster = "gbtree",
                eta                 = 0.05, # 0.06, #0.01,
                max_depth           = 10, #changed from default of 8
                subsample           = 0.75, # 0.7
                colsample_bytree    = 0.7 # 0.7
                # alpha = 0.0001, 
                # lambda = 1
)
Error<- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  elab <- exp(as.numeric(labels))
  epreds <- exp(as.numeric(preds))
  err <- mean(abs((epreds/elab)-1))
  return(list(metric = "MAPE", value = err))
}

clf <- xgb.train(   params              = param, 
                    data                = dtrain, 
                    nrounds             = 1000, #300, #280
                    verbose             = 1,
                    early.stop.round    = 100,
                    watchlist           = watchlist,
                    maximize            = FALSE,
                    feval               = Error
)
# set.seed(123)
v<- sample(nrow(train),10000)
pred <- predict(clf, data.matrix(train[v,-c(3,4)]))
pred <- exp(pred)-1
rmse <- sqrt(mean((pred-sales[v])^2))
pred[1:10]
train[v[1:10],3]

pred1 <- exp(predict(clf, data.matrix(test))) -1
#pred.test <- predict(clf, data.matrix(test))
sub$Sales <- ifelse(test$Open == 2, 0, pred1)

xgb.best <- read.csv("xgboost_best.csv")
xgb.2 <- read.csv('sub4.csv')
xgb.3 <- read.csv("sub20.csv")
check <- cbind(xgb.best, xgb.2$Sales, xgb.3$Sales, Sales)

test$xgb.best <- xgb.best$Sales
Sales <- ifelse(test$Open == 2, 0, xgb.best$Sales)
Sales <- 0.6*Sales + 0.3*xgb.2$Sales + 0.1*xgb.3$Sales
sub$Sales <- ifelse(test$Open == 2, 0, xgb.best$Sales)

sub$Sales <- Sales
write.csv(sub, file="sub22_randombest.csv", row.names=FALSE)
sum(is.na(sub))

plot(check$Sales, check$`sub$Sales`, cex=0.3, pch=20)

#---------
t <- train %>% 
  group_by(Store, Year, Month) %>% 
  filter(Month %in% c(7,9)) %>%
  summarise(meanS = mean(Sales),
           maxS = max(Sales),
           medianS = median(Sales))

ts <- test %>% 
  group_by(Store, Year, Month) %>% 
  filter(Month %in% c(7,9)) %>%
  summarise(meanS = mean(xgb.best),
            maxS = max(xgb.best),
            medianS = median(xgb.best))

#Train only on 7 and 9 months and the corresponding stores in test
train_79 <- train %>%
  filter(Store %in% unique(test$Store) & Month %in% c(7,9))

#Train xgb model
s <- sample(nrow(train_79),10000)
dval<-xgb.DMatrix(data=data.matrix(train_79[s,-c(3,4)]),label=log(train_79$Sales[s]+1))
dtrain<-xgb.DMatrix(data=data.matrix(train_79[-s,-c(3,4)]),label=log(train_79$Sales[-s]+1))
watchlist<-list(val=dval,train=dtrain)
param <- list(  objective           = "reg:linear", 
                booster             = "gbtree",
                eta                 = 0.05, # 0.06, #0.01,
                max_depth           = 15, #changed from default of 8
                subsample           = 0.7, # 0.7
                colsample_bytree    = 0.7 # 0.7
                # alpha = 0.0001, 
                # lambda = 1
)
Error<- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  elab <- exp(as.numeric(labels))
  epreds <- exp(as.numeric(preds))
  err <- mean(abs((epreds/elab)-1))
  return(list(metric = "MAPE", value = err))
}

clf <- xgb.train(   params              = param, 
                    data                = dtrain, 
                    nrounds             = 500, #300, #280
                    verbose             = 1,
                    early.stop.round    = 100,
                    watchlist           = watchlist,
                    maximize            = FALSE,
                    feval               = Error
)
# set.seed(123)
v<- sample(nrow(train_79),10000)
pred <- predict(clf, data.matrix(train_79[v,-c(3,4)]))
pred <- exp(pred)-1
rmse <- sqrt(mean((pred-sales[v])^2))
pred[1:10]
train_79[v[1:10],3]

pred1 <- exp(predict(clf, data.matrix(test))) -1
#pred.test <- predict(clf, data.matrix(test))
sub$Sales <- ifelse(test$Open == 2, 0, pred1)

xgb.best <- read.csv("xgboost_best.csv")
xgb.2 <- read.csv('sub4.csv')
xgb.3 <- read.csv("sub20.csv")
best <- read.csv("sub21_randensmble.csv")
check <- cbind(xgb.best, xgb.2$Sales, xgb.3$Sales, sub$Sales, best$Sales)
sum(check$`best$Sales`-check$`sub$Sales`)


test$xgb.best <- xgb.best$Sales
Sales <- ifelse(test$Open == 2, 0, xgb.best$Sales)
Sales <- 0.6*Sales + 0.3*xgb.2$Sales + 0.05*xgb.3$Sales+0.05*sub$Sales #0.6*best+0.3*nextbest + 0.1*3rd best is the curreent best
sub$Sales <- ifelse(test$Open == 2, 0, Sales)

sub$Sales <- Sales
write.csv(sub, file="sub26_crazyensemble.csv", row.names=FALSE)

# Random forest using caret
#Divide the data into 3 parts
t <- train

train <- train %>% mutate(logSales = log(Sales+1))
train <- train %>% arrange(Year, Month, Day)

train <- train %>% filter(Sales!=0)
#cut3 <- floor(nrow(train)/3)
trdata <- train %>% filter(Year != 3) #Training base models to generate ensemble data. Select all the records of 2013 and 14
table(train$Year)
# endata <- train %>% filter(Year == 3 & Month %in% c(1,2,3)) #Data for Ensemble building
# tsdata <- train %>% filter(Year == 3 & Month %in% c(4,5,6)) #Testing the ensemble model

tsdata <- train %>% filter(Year == 3) #Testing the ensemble model


label <- "logSales"
predictors <- names(train)[!(names(train) %in% c("Sales","logSales", "Customers","Open"))]

# control <- trainControl(method = "cv",
#                         number = 3)
# #grid <- expand.grid(mtry = c(8,12,15))
# rf <- train(trdata[,predictors],trdata[,label], method='rf', 
#             trControl = control,
#             trace = T, metric = "RMSE")
# 
# tr.small <- train[sample(nrow(train),100000),]
# 
# rf <- randomForest(trdata[,predictors],trdata[,label],ntree = 5,mtry = 5,do.trace = TRUE,replace = TRUE)
# rf

# RANDOM FORESTS using H2O
#intializing h2o process with 2 threads
library(h2o)
local=h2o.init(nthreads=2,max_mem_size='2G',assertion = FALSE)
#train_rf=as.h2o(data.matrix(train))
train_rf=as.h2o(data.matrix(trdata))
#endata_rf <- as.h2o(data.matrix(endata))

rf_h2o=h2o.randomForest(predictors,
                        label,
                        training_frame=train_rf,
                        ntrees=501,
                        mtries=10,      #no of maximum features to be used to split on for each tree
                        max_depth=12,  # maximum depth of each tree, pruning is good to avoid over-fitting
                        nbins_cats = 1115                    
)

#predict on endata and tsdata
#pred.rf <- as.data.frame(h2o.predict(rf_h2o, endata_rf))
pred.rf.ts <- as.data.frame(h2o.predict(rf_h2o,as.h2o(tsdata)))
pred.rf.test <- as.data.frame(h2o.predict(rf_h2o,as.h2o(test)))

#rmpse1 <- sqrt(mean(((exp(pred.rf)+1)/trdata[,"Sales"] - 1)^2))


# XGB with 2000 rounds, 0.025 LR, depth as 12
s <- (nrow(train)-80000):nrow(train) #Splitting temporally
dval<-xgb.DMatrix(data=data.matrix(train[s,predictors]),label=train[s,label])
dtrain<-xgb.DMatrix(data=data.matrix(train[-s,predictors]),label=train[-s,label])
watchlist<-list(val=dval,train=dtrain)
param <- list(  objective           = "reg:linear", 
                booster             = "gbtree",
                eta                 = 0.01, # 0.06, #0.01,
                max_depth           = 15, #changed from default of 8
                subsample           = 0.8, # 0.7
                colsample_bytree    = 0.95 # 0.7
                # alpha = 0.0001, 
                # lambda = 1
)
Error<- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  elab <- expm1(as.numeric(labels))
  epreds <- expm1(as.numeric(preds))
  err <- sqrt(mean((epreds/elab - 1)^2))
  return(list(metric = "RMPSE", value = err))
}

xgb11 <- xgb.train(   params              = param, 
                    data                = dtrain, 
                    nrounds             = 5000, #300, #280
                    verbose             = 1,
                    early.stop.round    = 50,
                    watchlist           = watchlist,
                    maximize            = FALSE,
                    feval               = Error
)
#Predict on endata and tsdata
#pred.xgb1 <- predict(xgb1, data.matrix(endata[,predictors]))
pred.xgb1.ts <- predict(xgb1, data.matrix(tsdata[,predictors]))
pred.xgb1.test <- predict(xgb1, data.matrix(test[,predictors]))
# xgb2 for ensemble
s <- 1:80000 #Splitting temporally
dval<-xgb.DMatrix(data=data.matrix(trdata[s,predictors]),label=trdata[s,label])
dtrain<-xgb.DMatrix(data=data.matrix(trdata[-s,predictors]),label=trdata[-s,label])

param <- list(  objective           = "reg:linear", 
                booster             = "gbtree",
                eta                 = 0.05, # 0.06, #0.01,
                max_depth           = 10, #changed from default of 8
                subsample           = 0.7, # 0.7
                colsample_bytree    = 0.7 # 0.7
                # alpha = 0.0001, 
                # lambda = 1
)
xgb2 <- xgb.train(   params              = param, 
                     data                = dtrain, 
                     nrounds             = 4000, #300, #280
                     verbose             = 1,
                     early.stop.round    = 10,
                     watchlist           = watchlist,
                     maximize            = FALSE,
                     feval               = Error
)
#predict on endata and tsdata
#pred.xgb2 <- predict(xgb2, data.matrix(endata[,predictors]))
pred.xgb2.ts <- predict(xgb2, data.matrix(tsdata[,predictors]))
pred.xgb2.test <- predict(xgb2, data.matrix(test[,predictors]))

#Build a linear model with one round of outliers removed
lm1 <- lm(logSales~., trdata[,c(predictors,label)])
resid <- rstudent(lm1)
length(resid[abs(resid) > 3.5])
tr.lm <- trdata[-which(abs(resid) > 3.5),]
lm1 <- lm(logSales~., tr.lm[,c(predictors,label)])
summary(lm1)
#predict on endata and tsdata
#pred.lm <- predict(lm1, endata[,predictors])
pred.lm.ts <- predict(lm1, tsdata[,predictors])
pred.lm.test <- predict(lm1, test[,predictors])

#Create data for 2nd level model
#endata <- as.data.frame(cbind(endata, pred.rf, pred.xgb1, pred.xgb2, pred.lm))
tsdata <- as.data.frame(cbind(tsdata, pred.rf.ts, pred.xgb1.ts, pred.xgb2.ts, pred.lm.ts))
test <- as.data.frame(cbind(test, pred.rf.test, pred.xgb1.test, pred.xgb2.test, pred.lm.test))
names(test)[19:22] <- names(tsdata)[22:25]

#names(tsdata) <- names(endata)
#fit xgb on endata and predict on test
en.predictors <- names(tsdata)[!(names(endata) %in% c("Sales","logSales", "Customers","Open","pred.lm.ts"))]

s.en <- ((nrow(endata)-1e04):nrow(endata)) #Splitting temporally
dval<-xgb.DMatrix(data=data.matrix(tsdata[s.en,en.predictors]),label=tsdata[s.en,label])
dtrain<-xgb.DMatrix(data=data.matrix(tsdata[-s.en,en.predictors]),label=tsdata[-s.en,label])
watchlist<-list(val=dval,train=dtrain)
param <- list(  objective           = "reg:linear", 
                booster             = "gbtree",
                eta                 = 0.05, # 0.06, #0.01,
                max_depth           = 10, #changed from default of 8
                subsample           = 0.7, # 0.7
                colsample_bytree    = 0.8 # 0.7
                # alpha = 0.0001, 
                # lambda = 1
                )

xgb.en <- xgb.train( params              = param, 
                     data                = dtrain, 
                     nrounds             = 1000, #300, #280
                     verbose             = 1,
                     early.stop.round    = 10,
                     watchlist           = watchlist,
                     maximize            = FALSE,
                     feval               = Error
)

#Now predict on tsdata using 2nd level model
pred.ensemble.ts <- predict(xgb.en, data.matrix(tsdata[,en.predictors]))
mape.en <- mean(abs(tsdata$logSales/pred.ensemble.ts - 1))
mape.xgb1 <- mean(abs(tsdata$logSales/tsdata$predict - 1))

#predict on test using the ensemble model
pred.final <- predict(xgb.en, data.matrix(test[,en.predictors]))
Sales <- expm1(pred.xgb1.test)
#sub$Sales <- Sales
sub$Sales <- ifelse(test$Open == 2, 0, Sales)
crazy <- read.csv("crazycrazyensemble.csv")
check <- cbind(sub,best$Sales,xgb.best$Sales,crazy$Sales)

Sales <- crazy$Sales*0.8 + sub$Sales*0.2
rmpse(check[check$Sales!=0,5],check[check$Sales!=0,3])
rmpse <- function(y, yhat){
  sqrt(mean((yhat/y - 1)^2))
}
xgb.best <- read.csv("xgboost_best.csv")
xgb.2 <- read.csv('sub4.csv')

write.csv(sub, file="sub35_random.csv", row.names=FALSE)
# mean sales model - really cool
train = train[train$Sales>0,]
vars=c('Store','DayOfWeek','Promo')
mdl = train %>% group_by_(.dots=vars) %>% summarise(PredSales=exp(mean(log(Sales)))) %>% ungroup()
pred = test %>% left_join(mdl,by=vars) %>% select(Id,PredSales) %>% rename(Sales=PredSales)
pred$Sales[is.na(pred$Sales)]=0
write.csv(pred, "pred.csv",row.names=F)





