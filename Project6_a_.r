# Load necessary libraries
# recommenderlab is a package used for building and evaluating recommendation systems in R
if(!require(recommenderlab)) install.packages("recommenderlab", dependencies=TRUE)
library(recommenderlab)

# Load the MovieLense dataset
# This dataset contains user ratings for movies and is included with the recommenderlab package
data("MovieLense")

# Print an overview of the dataset
# This will display basic statistics about the dataset, such as the number of users, items, and ratings
print(summary(MovieLense))

# Convert the dataset into a binary rating matrix
# Movies rated >= 3 are considered "watched" (1), and those rated < 3 are considered "not watched" (0)
# Binarizing helps us work with simpler data (watched vs not watched)
binary_ratings <- binarize(MovieLense, minRating = 3)

# Split the data into training and test sets (80% training, 20% testing)
# We use the training set to build the model and the test set to evaluate it
set.seed(123)  # Set seed for reproducibility
train_indices <- sample(x = 1:nrow(binary_ratings), size = 0.8 * nrow(binary_ratings))  # 80% of the data for training
train_data <- binary_ratings[train_indices,]  # Training data
test_data <- binary_ratings[-train_indices,]  # Test data

# Build a recommendation model using Collaborative Filtering
# We use User-Based Collaborative Filtering (UBCF), which finds similar users to make recommendations
# UBCF recommends items to a user based on what similar users have liked/watched
recommender_model <- Recommender(train_data, method = "UBCF")

# Predict recommendations for users in the test set
# We generate 5 movie recommendations per user from the test set
predictions <- predict(recommender_model, test_data, n = 5)

# Display the recommendations for the first user in the test set
# We convert the prediction object to a list so it's easier to view the recommendations
print(as(predictions, "list"))

# Evaluation of the recommender system
# To check how well our model performs, we use 5-fold cross-validation
# In cross-validation, we divide the data into 5 parts and test the model on each part

# Define the evaluation scheme using cross-validation
# 'given = 5' means that we will use 5 movies as input to predict the remaining ones for each user
# 'goodRating = 3' sets the threshold for considering a rating "good" (>=3 means the user liked the movie)
eval_scheme <- evaluationScheme(data = binary_ratings, method = "cross-validation", 
                                train = 0.8, given = 5, goodRating = 3, k = 5)

# Define the models we want to evaluate
# We are comparing User-Based (UBCF) and Item-Based (IBCF) Collaborative Filtering
models_to_evaluate <- list(
  UBCF = list(name = "UBCF", param = list()),  # User-based CF
  IBCF = list(name = "IBCF", param = list())   # Item-based CF
)

# Perform the evaluation of both models
# The evaluate function will compute precision and recall for the top N recommendations (1, 3, 5 movies)
results <- evaluate(eval_scheme, method = models_to_evaluate, type = "topNList", n = c(1, 3, 5))

# Display the average precision and recall across the 5-fold cross-validation
print(avg(results))

# Plot the evaluation results
# This will help visualize the performance of User-based vs Item-based Collaborative Filtering
plot(results, annotate = 1:2, legend="topleft")
