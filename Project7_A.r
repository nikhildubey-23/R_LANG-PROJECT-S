# Load necessary libraries
# caret: Provides tools for data partitioning and model evaluation
# randomForest: Implements the Random Forest algorithm for classification
# dplyr: Offers data manipulation functions
if (!require(caret)) install.packages("caret", dependencies=TRUE)
if (!require(randomForest)) install.packages("randomForest", dependencies=TRUE)
if (!require(dplyr)) install.packages("dplyr", dependencies=TRUE)

library(caret)          # Load the caret library
library(randomForest)   # Load the randomForest library
library(dplyr)          # Load the dplyr library for data manipulation

# Load the dataset
# Please replace the file path with the actual path to your dataset
data <- read.csv("path/to/creditcard.csv")

# Overview of the dataset
# Display the first few rows of the dataset
print(head(data))

# Summary statistics of the dataset
# This will show the distribution of features and the target variable (Class)
print(summary(data))

# Check for missing values in the dataset
# This step ensures there are no NA values which could affect model training
print(colSums(is.na(data)))

# Preprocessing
# Normalize the Amount feature to ensure all features contribute equally to the model
# Scaling transforms the Amount to have a mean of 0 and a standard deviation of 1
data$Amount <- scale(data$Amount)

# Split the data into training and testing sets
# We will use 80% of the data for training and 20% for testing the model's performance
set.seed(123)  # Setting a seed for reproducibility so we get the same split every time
trainIndex <- createDataPartition(data$Class, p = 0.8, 
                                  list = FALSE, 
                                  times = 1)
# Create training and test datasets based on the indices generated
train_data <- data[trainIndex, ]  # Training data containing 80% of samples
test_data <- data[-trainIndex, ]   # Testing data containing the remaining 20%

# Train a Random Forest model
# The target variable is Class (0 for legitimate transactions, 1 for fraudulent)
# We use 100 trees in the Random Forest for better stability and accuracy
rf_model <- randomForest(Class ~ ., data = train_data, 
                         ntree = 100, importance = TRUE)

# Print model summary to check the structure and variable importance
print(rf_model)

# Make predictions on the test set using the trained model
predictions <- predict(rf_model, test_data)

# Generate a confusion matrix to evaluate the model's performance
# The confusion matrix shows true positives, true negatives, false positives, and false negatives
confusion_matrix <- confusionMatrix(predictions, test_data$Class)
print(confusion_matrix)

# Calculate additional evaluation metrics for performance assessment
# Precision: Measures the accuracy of positive predictions
# Recall: Measures the ability of the model to find all relevant cases (fraudulent transactions)
# F1 Score: Harmonic mean of precision and recall, providing a balance between the two
precision <- confusion_matrix$byClass["Precision"]
recall <- confusion_matrix$byClass["Recall"]
f1_score <- 2 * (precision * recall) / (precision + recall)

# Print the additional metrics for model performance
cat("Precision:", precision, "\n")  # Display precision
cat("Recall:", recall, "\n")         # Display recall
cat("F1 Score:", f1_score, "\n")     # Display F1 Score

# Variable importance plot to visualize which features are most important
# This helps in understanding which factors contribute most to predicting fraud
varImpPlot(rf_model)
