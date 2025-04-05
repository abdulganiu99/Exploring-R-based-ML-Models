#RUN THIS ONCE TO INSTALL NECCESARY PACKAGES
required <- c("caret", "ggplot2", "reshape2", "MLmetrics", "e1071", "xgboost", "nnet", "kernlab")
new <- required[!(required %in% installed.packages())]
if(length(new)) install.packages(new)

library(caret)
library(ggplot2)
library(reshape2)
library(MLmetrics)

#IMPORTING THE DATA
data(iris)

#BASIC EDA
summary(iris)
str(iris)
sum(is.na(iris))
colSums(is.na(iris))

ggplot(iris, aes(x=Petal.Length, y= Petal.Width, color=Species))+
  geom_point(size = 3, alpha = 0.7)+
  theme_minimal()+
  labs(title = "Petal Length vs Petal Width by species")

ggplot(iris, aes(x=Sepal.Length, y= Sepal.Width, color=Species))+
  geom_point(size = 3, alpha = 0.7)+
  theme_minimal()+
  labs(title = "Sepal Length vs Sepal Width by species")
#the sepals have overlapping features for species(versicolor & virginica)

#prePROCESSING DATA
#SPLIT THE DATA INTO TESTING AND TRAINING SETS
set.seed(123)
trainIndex <- createDataPartition(iris$Species, p = 0.8, list=FALSE)
trainData <- iris[trainIndex, ]
testData <- iris[-trainIndex, ]

#FEATURE ENGINEERING
trainData$PetalRatio <- trainData$Petal.Length / trainData$Petal.Width
trainData$PetalArea <- trainData$Petal.Length * trainData$Petal.Width
testData$PetalRatio <- testData$Petal.Length / testData$Petal.Width
testData$PetalArea <- testData$Petal.Length * testData$Petal.Width
trainData$SepalPetalRatio <- trainData$Sepal.Length / trainData$Petal.Length
testData$SepalPetalRatio <- testData$Sepal.Length / testData$Petal.Length

#non-linear transformations
trainData$PetalInteraction <- (trainData$Petal.Length * trainData$Petal.Width)^2
testData$PetalInteraction <- (testData$Petal.Length * testData$Petal.Width)^2
#END OF prePROCESSING DATA

#MODEL TO TEST GENERALIZATION
model_rf_general <- train(
  Species ~ ., data = trainData, method = "rf", 
               trControl = trainControl(method = "cv", number = 10)
)
print(model_rf_general)
predictions <- predict(model_rf_general, newdata= testData)
confusionMatrix(predictions, testData$Species)

#A PEAK AT THE MISCLASSIFIED SAMPLES
misclassified <- testData[predictions != testData$Species, ]
print(misclassified)

#MODEL TO TEST MEMORIZATION(Train on all data)
model_memorize <- train(Species ~ ., data = iris, method = "rf", 
                        trControl = trainControl(method = "none"))
predictions <- predict(model_memorize, newdata= iris)
confusionMatrix(predictions, iris$Species)


#EVALUATING THE MODELS PERFORMANCE ON TEST DATA
predictions <- predict(model_rf_general, newdata = testData)
predictions <- factor(predictions, levels = levels(testData$Species))  # align levels
cm <- confusionMatrix(predictions, testData$Species)

cm_table <- as.data.frame(cm$table) 
ggplot(cm_table, aes(x = Prediction, y = Reference, fill = Freq)) +
  geom_tile(color = "black") +
  scale_fill_gradient(low = "white", high = "blue") +
  geom_text(aes(label = Freq), color = "black", size = 5) +
  labs(title = "Confusion Matrix Heatmap - Random Forest", x = "Predicted Label", y = "Actual Label") +
  theme_minimal()

# Plotting variable importance
varImpPlot <- varImp(model_rf_general)
plot(varImpPlot, main = "Variable Importance - Random Forest")

saveRDS(model_rf_general, "model_rf_general.rds")
# To load it later: loaded_model <- readRDS("tuned_rf_model.rds")

#CHECKING WHERE THE MODEL IS STRUGGLING TO PREDICT
multi_class_f1 <- function(actual, predicted) {
  if (!require(MLmetrics)) {
    install.packages("MLmetrics")
    library(MLmetrics)
  }
  
  classes <- levels(actual)
  
  f1_scores <- c()
  
  for (cls in classes) {
    # One-vs-all setup
    binary_true <- ifelse(actual == cls, 1, 0)
    binary_pred <- ifelse(predicted == cls, 1, 0)
    
    f1 <- F1_Score(binary_pred, binary_true)
    f1_scores <- c(f1_scores, f1)
    
    cat("F1 Score for", cls, ":", round(f1, 3), "\n")
  }
  
  # Optionally return a named vector of scores
  names(f1_scores) <- classes
  return(f1_scores)
}
f1_results <- multi_class_f1(testData$Species, predictions)



#======================================================================
#MODEL COMPARISM WITH OTHERS
#======================================================================
#INSTALLING NECCESARY LIBRARIES
install.packages(c("e1071","xgboost","nnet", "kernlab"))
ctrl <- trainControl(method = "cv", number = 10)
# Train SVM
model_svm <- train(Species ~ ., data = trainData, method = "svmRadial", trControl = ctrl)

# Train KNN
model_knn <- train(Species ~ ., data = trainData, method = "knn", trControl = ctrl)

# Train XGBoost
model_xgb <- train(Species ~ ., data = trainData, method = "xgbTree", trControl = ctrl)  

models_list <- list(RandomForest = model_rf_general,
                    SVM = model_svm,
                    KNN = model_knn,
                    XGBoost = model_xgb)

results <- resamples(models_list)
summary(results)
dotplot(results, metric = "Accuracy", main = "Model Accuracy Comparison")
bwplot(results, layout = c(1, 2))  # Boxplots for Accuracy & Kappa


