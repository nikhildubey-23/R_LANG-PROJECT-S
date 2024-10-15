install.packages("tuneR")    # For reading and processing audio files
install.packages("seewave")   # For extracting audio features
install.packages("keras")     # For building neural networks

# Load necessary libraries
library(keras)
library(tuneR)
library(seewave)

# Load the trained model
model <- load_model_hdf5("song_recognition_model.h5")

# Function to preprocess and predict lyrics from a new audio file
predict_lyrics <- function(audio_file) {
  # Load the audio file
  audio <- readWave(audio_file)
  
  # Extract MFCC features
  mfccs <- seewave::mfcc(audio@left, f = audio@samp.rate, nco = 13, plot = FALSE)
  
  # Convert to data frame
  mfccs_df <- as.data.frame(mfccs)
  
  # Predict using the trained model
  predictions <- model %>% predict(as.matrix(mfccs_df))
  
  # Get the predicted class (index of max probability)
  predicted_index <- which.max(predictions)
  
  # Map the index back to the actual lyrics (adjust this based on your labels)
  return(predicted_index)  # You may want to create a mapping to actual lyrics
}

# Example usage
predicted_lyrics <- predict_lyrics("path/to/new_song.wav")
print(predicted_lyrics)
