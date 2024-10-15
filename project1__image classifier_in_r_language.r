# Install and load required libraries
#install.packages("tensorflow")
#install.packages("imager")  # For image processing

library(tensorflow)
library(imager)

# Step 1: Load CIFAR-10 Dataset directly using TensorFlow
cifar10 <- tf$keras$datasets$cifar10$load_data()
x_train <- cifar10[[1]][[1]]
y_train <- cifar10[[1]][[2]]
x_test <- cifar10[[2]][[1]]
y_test <- cifar10[[2]][[2]]

# Step 2: Preprocess the Data
# Normalize pixel values to the range [0,1]
x_train <- x_train / 255.0
x_test <- x_test / 255.0

# Ensure labels are integers
y_train <- as.integer(as.vector(y_train))
y_test <- as.integer(as.vector(y_test))

# Step 3: Convert labels to one-hot encoding
num_classes <- 10
y_train <- tf$one_hot(y_train, num_classes)
y_test <- tf$one_hot(y_test, num_classes)

# Step 4: Build the CNN Model
model <- tf$keras$models$Sequential(list(
  # 1st Convolutional Layer with input_shape
  tf$keras$layers$Conv2D(filters = 32, kernel_size = c(3, 3), activation = 'relu', input_shape = c(32, 32, 3)),
  tf$keras$layers$MaxPooling2D(pool_size = c(2, 2)),
  
  # 2nd Convolutional Layer
  tf$keras$layers$Conv2D(filters = 64, kernel_size = c(3, 3), activation = 'relu'),
  tf$keras$layers$MaxPooling2D(pool_size = c(2, 2)),
  
  # 3rd Convolutional Layer
  tf$keras$layers$Conv2D(filters = 128, kernel_size = c(3, 3), activation = 'relu'),
  tf$keras$layers$MaxPooling2D(pool_size = c(2, 2)),
  
  # Flatten the feature maps
  tf$keras$layers$Flatten(),
  
  # Fully connected layer
  tf$keras$layers$Dense(units = 512, activation = 'relu'),
  
  # Output layer with softmax activation for multi-class classification
  tf$keras$layers$Dense(units = num_classes, activation = 'softmax')
))

# Step 5: Compile the Model
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = tf$keras$optimizers$Adam(),
  metrics = c('accuracy')
)

# Step 6: Train the Model
history <- model %>% fit(
  x_train, y_train,
  epochs = 20,
  batch_size = 32,
  validation_split = 0.2
)

# Step 7: Evaluate the Model
score <- model %>% evaluate(x_test, y_test, verbose = 0)
cat('Test loss:', score[[1]], '\n')
cat('Test accuracy:', score[[2]], '\n')

# Step 8: Save the Model
model %>% save_model_hdf5("cifar10_animal_classifier.h5")

# Step 9: Preprocess and Predict New Data
preprocess_image <- function(image_path) {
  img <- load.image(image_path)
  img <- resize(img, size_x = 32, size_y = 32)
  img_array <- as.array(img)
  img_array <- img_array / 255
  img_array <- array(img_array, dim = c(1, 32, 32, 3))
  return(img_array)
}

predict_image <- function(image_path) {
  img_array <- preprocess_image(image_path)
  prediction <- model %>% predict(img_array)
  predicted_class_index <- which.max(prediction) - 1
  return(predicted_class_index)
}

# Example: Predict the class of a new image
image_path <- "C:/Users/HARDWARE LAB 30/Desktop/R_project_5thsem/download.jpg"
predicted_class_index <- predict_image(image_path)

cifar10_labels <- c("airplane", "automobile", "bird", "cat", "deer", "dog", 
                    "frog", "horse", "ship", "truck")

cat("Predicted label:", cifar10_labels[predicted_class_index + 1], "\n")
