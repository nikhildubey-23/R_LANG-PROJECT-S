# run_translation_english_french.R
source("translation_model_english_french.R")

# Function to translate English text to French
translate_to_french <- function(text) {
  loaded_model <- load_translation_model() # nolint: object_usage_linter.
  tokenizer <- loaded_model$tokenizer
  model <- loaded_model$model
  
  # Tokenize the input text
  inputs <- tokenizer$encode(text, return_tensors = "pt")

  # Generate the translation
  translated_tokens <- model$generate(inputs)
  
  # Decode the tokens back to French text
  translated_text <- tokenizer$decode(translated_tokens[[1]], skip_special_tokens = TRUE)
  
  return(translated_text)
}

# Example: Translating a part of a book
book_excerpt <- "Once upon a time, there was a little girl who lived in a village."
french_translation <- translate_to_french(book_excerpt)
cat("French Translation: ", french_translation, "\n")
