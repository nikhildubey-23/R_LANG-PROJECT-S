# Install necessary packages (uncomment if needed)
# install.packages("keras")
# install.packages("tensorflow")

library(keras)
library(tensorflow)

# Install Keras with TensorFlow backend (run only if not installed)
# install_keras()

# Load the dataset (you can replace this with actual data)
source_sentences <- c("Hello", "How are you?", "Good morning")
target_sentences <- c("Hola", "¿Cómo estás?", "Buenos días")

# Tokenizer for the source and target sentences
tokenizer <- text_tokenizer(num_words = 10000)
fit_text_tokenizer(tokenizer, c(source_sentences, target_sentences))

# Convert sentences to sequences
source_sequences <- texts_to_sequences(tokenizer, source_sentences)
target_sequences <- texts_to_sequences(tokenizer, target_sentences)

# Find the maximum length of sequences in the source and target languages
max_len_source <- max(sapply(source_sequences, length))
max_len_target <- max(sapply(target_sequences, length))

# Pad sequences to ensure uniform length
source_sequences <- pad_sequences(source_sequences, maxlen = max_len_source)
target_sequences <- pad_sequences(target_sequences, maxlen = max_len_target)

# Define model parameters
input_dim <- length(tokenizer$word_index) + 1  # Vocabulary size
embedding_dim <- 256
latent_dim <- 512

# Encoder
encoder_inputs <- layer_input(shape = c(max_len_source))
encoder_embedding <- layer_embedding(input_dim = input_dim, output_dim = embedding_dim)(encoder_inputs)
encoder_lstm <- layer_lstm(units = latent_dim, return_sequences = FALSE, return_state = TRUE)
encoder_outputs, state_h, state_c <- encoder_lstm(encoder_embedding)

# Decoder
decoder_inputs <- layer_input(shape = c(max_len_target))
decoder_embedding <- layer_embedding(input_dim = input_dim, output_dim = embedding_dim)(decoder_inputs)
decoder_lstm <- layer_lstm(units = latent_dim, return_sequences = TRUE, return_state = TRUE)
decoder_outputs, _, _ <- decoder_lstm(decoder_embedding, initial_state = list(state_h, state_c))

# Attention mechanism
attention <- layer_attention()([decoder_outputs, encoder_outputs])
decoder_combined_context <- layer_concatenate()([decoder_outputs, attention])

# Dense layer for predicting the next word
decoder_dense <- layer_dense(units = input_dim, activation = 'softmax')
output <- decoder_dense(decoder_combined_context)

# Define the model
model <- keras_model(inputs = c(encoder_inputs, decoder_inputs), outputs = output)

# Compile the model
model %>% compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = 'accuracy')

# View model summary
summary(model)

# Train the model (dummy data used here, replace with actual dataset)
history <- model %>% fit(
  list(source_sequences, target_sequences),
  target_sequences,
  epochs = 20,            # Number of training iterations
  batch_size = 64,        # Batch size
  validation_split = 0.2  # Use 20% of the data for validation
)

# Save the model after training
save_model_hdf5(model, "translation_model.h5")

# To load the model in the future, use the following:
# loaded_model <- load_model_hdf5("translation_model.h5")

# Test the model on a new sentence (dummy example)
new_source_sentence <- texts_to_sequences(tokenizer, "How is the weather?")
padded_new_source <- pad_sequences(new_source_sentence, maxlen = max_len_source)
prediction <- model %>% predict(list(padded_new_source))

# Convert predicted sequence back to words (using the tokenizer)
predicted_words <- tokenizer$index_word[prediction]
cat("Translated sentence:", paste(predicted_words, collapse = " "))
