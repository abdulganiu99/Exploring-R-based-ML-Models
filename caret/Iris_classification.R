#INSTALLATION OF ML LIBRARIES
install.packages("caret")
library(caret)
library(dplyr)

#IMPORTING THE DATA
data(iris)

#BASIC EDA
summary(iris)
str(iris)

#CHECK DATA FOR MISSING VALUES
sum(is.na(iris))
colSums(is.na(iris))

#prePROCESSING DATA
#SPLIT THE DATA INTO TESTING AND TRAINING SETS
set.seed(123)
trainIndex <- createDataPartition(iris$Species, p = 0.8, list=FALSE)
trainData <- iris[trainIndex, ]
testData <- iris[-trainIndex, ]

#STANDARDIZE THE NUMERICAL FEATURES:
preProcValues <- preProcess(trainData[, -5], method = c("center", "scale"))
trainTransformed <- predict(preProcValues, trainData[, -5])
testTransformed <- predict(preProcValues, testData[, -5])

#ENCODING CATEGORICAL VARIABLES
dummies <- dummyVars(Species ~ ., data = trainData)
trainTransformedDummies <- predict(dummies, newdata = trainData)
testTransformedDummies <- predict(dummies, newdata = testData)
#END OF prePROCESSING DATA

#TRAINING A MODEL
set.seed(123)
model <- train(Species ~ ., data = trainData, 
               method = "rf", trControl = trainControl(method = "cv", number = 10))
print(model)

#TUNING HYPERPARAMETERS
tuneGrid <- expand.grid(mtry = c(1, 2, 3))
set.seed(123)
modelTuned <- train(Species ~ ., data = trainData, 
                    method = "rf", trControl = trainControl(method = "cv", number = 10),
                    tuneGrid = tuneGrid)
print(modelTuned)

#EVALUATING THE MODELS PERFORMANCE ON TEST DATA
predictions <- predict(modelTuned, newdata = testData)
confMatrix <- confusionMatrix(predictions, testData$Species)
print(confMatrix)


library(ggplot2)
library(reshape2)
# Manually input confusion matrix values
conf_matrix_df <- data.frame(
  Actual = rep(c("Setosa", "Versicolor", "Virginica"), each = 3),
  Predicted = rep(c("Setosa", "Versicolor", "Virginica"), times = 3),
  Count = c(10, 0, 0, 0, 10, 2, 0, 0, 8)  # Fill in actual confusion matrix values
)

# Convert to matrix format
conf_matrix_melted <- melt(conf_matrix_df)
ggplot(conf_matrix_df, aes(x = Predicted, y = Actual, fill = Count)) +
  geom_tile(color = "black") +
  scale_fill_gradient(low = "white", high = "blue") +
  geom_text(aes(label = Count), color = "black", size = 5) +
  labs(title = "Confusion Matrix Heatmap", x = "Predicted Label", y = "Actual Label") +
  theme_minimal()






