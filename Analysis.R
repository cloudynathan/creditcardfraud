#weighted random forest with 5 fold cross-validation 
library(caret)

df <- read.csv("C:/workspaceR/creditcardfraud/creditcard.csv")
str(df)
df$Class <- as.factor(df$Class)
levels(df$Class) <- c("NotFraud", "Fraud")
str(df$Class)
summary(df$Class)
anyNA(df)
#imbalanced binary classification

set.seed(123)

data_set_size <- floor(nrow(df)*0.80)
index <- sample(1:nrow(df), size = data_set_size)
training <- df[index,]
testing <- df[-index,]

ControlParameters <- trainControl(method="cv",
                                  number=5,
                                  classProbs=TRUE,
                                  savePredictions=TRUE,
                                  verboseIter=TRUE)

rf <- train(Class~.,
            data=training,
            method="ranger",
            class.weights = c(0.1,0.9),
            num.trees=101,
            classification=TRUE,
            importance="impurity",
            trControl=ControlParameters)

predictions <- predict(rf, testing)

t <- table(predictions=predictions, actual=testing$Class)
t

correlationMatrix <- cor(df[,1:30])
importance <- varImp(rf)
print(importance)
plot(importance)


