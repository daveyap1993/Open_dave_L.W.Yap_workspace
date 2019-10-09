##### Prediction
#####logistics regression
# Split the data back into a train set and a test set
train <- full[1:891,]
test <- full[892:1309,]

library(tidyverse)
library(caret)
library(MASS)
library(SDMTools)
library(randomForest)
library(ggthemes)
library(e1071)
library(xgboost)
library(Matrix)


# define training control
train_control<- trainControl(method="cv", number=5)

##################### train the full model 
model<- train(Survived ~Pclass + Sex + Age + SibSp + Parch +Fare + Embarked + Title + 
                Fsize + Child + Mother, data=train, trControl=train_control, method="glm", family=binomial())

# print cv scores
summary(model)

probabilities <- model %>% predict(train)
predicted.classes <- ifelse(probabilities > 0.5, "1", "0")
gt <- train$Survived
mean(gt == predicted.classes) #0.8338945

probabilities <- model %>% predict(test)
predicted.classes.test <- ifelse(probabilities > 0.5, "1", "0")

submission <- data.table(test[c("PassengerId")])
submission[, Survived := predicted.classes.test]
fwrite(submission,file="submission_fulllogit.csv")


#################### train the stepwise model 
model<- train(Survived ~Pclass + Sex + Age + SibSp + Parch +Fare + Embarked + Title + 
                Fsize + Child + Mother, data=train, trControl=train_control, method="glmStepAIC", family=binomial())
           
# print cv scores
summary(model)

probabilities <- model %>% predict(train)
predicted.classes <- ifelse(probabilities > 0.5, "1", "0")
gt <- train$Survived
mean(gt == predicted.classes)

probabilities <- model %>% predict(test)
predicted.classes.test <- ifelse(probabilities > 0.5, "1", "0")

submission <- data.table(test[c("PassengerId")])
submission[, Survived := predicted.classes.test]
fwrite(submission,file="submission_stepwiselogit.csv")


########################################################################################
#####random forest
# Set a random seed and 3 fold
train_control<- trainControl(method="cv", number=3)
set.seed(754)
train$Survived <- as.factor(train$Survived)

mtry <- sqrt(ncol(train))
tunegrid <- expand.grid(.mtry=mtry)

# Build the model (note: not all possible variables are used)
rf_model <- train(Survived ~Pclass + Sex + Age + SibSp + Parch +Fare + Embarked + Title + 
                            Fsize + Child + Mother, data=train, trControl=train_control, method="rf",
                  family=binomial(),matric="Accuracy",tuneGrid=tunegrid,ntree=200)
# print cv scores
summary(rf_model)

probabilities <- rf_model  %>% predict(train)
gt <- train$Survived
mean(gt == probabilities)

#predict
probabilities <- model %>% predict(test)
predicted.classes.test <- ifelse(probabilities > 0.5, "1", "0")

#print result
submission <- data.table(test[c("PassengerId")])
submission[, Survived := predicted.classes.test]
fwrite(submission,file="submission_rf.csv")

#random forest-important feature plot
# Set a random seed
set.seed(754)

# Build the model (note: not all possible variables are used)
rf_model <- randomForest(factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + 
                           Fare + Embarked + Title + Fsize + Child + Mother, data = train, ntree=200)

# Show model error
plot(rf_model, ylim=c(0,0.36))
legend('topright', colnames(rf_model$err.rate), col=1:3, fill=1:3)

# Get importance
importance    <- importance(rf_model)
varImportance <- data.frame(Variables = row.names(importance), 
                            Importance = round(importance[ ,'MeanDecreaseGini'],2))

# Create a rank variable based on importance
rankImportance <- varImportance %>%
  mutate(Rank = paste0('#',dense_rank(desc(Importance))))

# Use ggplot2 to visualize the relative importance of variables
ggplot(rankImportance, aes(x = reorder(Variables, Importance), 
                           y = Importance, fill = Importance)) +
  geom_bar(stat='identity') + 
  geom_text(aes(x = Variables, y = 0.5, label = Rank),
            hjust=0, vjust=0.55, size = 4, colour = 'red') +
  labs(x = 'Variables') +
  coord_flip() + 
  theme_few()+
  theme(text = element_text(size=20))

################################################################################################################
###xgboost
sparse_matrix <- sparse.model.matrix(Survived ~ Pclass + Sex + Age + SibSp + 
                                       Parch +Fare + Embarked + Title + 
                                       Fsize + Child + Mother, data = train)[,-1]
output_vector = as.numeric(train$Survived)-1
#build model
bst <- xgboost(data = sparse_matrix, label = output_vector, max_depth = 5, gamma=0.2,
               eta = 0.2, nthread = 2, nrounds = 200,objective = "binary:logistic", early_stopping_rounds = 5)
probabilities <- predict(bst, sparse_matrix)
predicted.classes.test <- ifelse(probabilities > 0.5, "1", "0")
gt <- train$Survived
mean(gt == predicted.classes.test)

test$Survived <- 1
sparse_matrix_test <- sparse.model.matrix(Survived ~ Pclass + Sex + Age + SibSp + 
                                       Parch +Fare + Embarked + Title + 
                                       Fsize + Child + Mother, data = test)[,-1]
probabilities <- bst %>% predict(sparse_matrix_test)
predicted.classes.test <- ifelse(probabilities > 0.5, "1", "0")

#print result
submission <- data.table(test[c("PassengerId")])
submission[, Survived := predicted.classes.test]
fwrite(submission,file="submission_xgboost.csv")

#important feature
importanceRaw <- xgb.importance(feature_names = colnames(sparse_matrix), 
                                model = bst, data = sparse_matrix, 
                                label = output_vector)

# Cleaning for better display
importanceClean <- importanceRaw[,`:=`(Cover=NULL, Frequency=NULL)]
head(importanceClean)
#plot
ggplot(importanceClean, aes(x = reorder(Feature,Gain), 
                           y = Gain)) +
  geom_bar(stat='identity') + 
  coord_flip()+
  theme_few()+
  theme(text = element_text(size=20))
