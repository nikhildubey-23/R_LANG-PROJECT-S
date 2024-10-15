# run_translation.R
source("translation_model.R")

# Function to translate English text to Spanish
translate_to_spanish <- function(text) {
  loaded_model <- load_translation_model() # nolint
  tokenizer <- loaded_model$tokenizer
  model <- loaded_model$model
  
  # Tokenize the input text
  inputs <- tokenizer$encode(text, return_tensors = "pt")

  # Generate the translation
  translated_tokens <- model$generate(inputs)
  
  # Decode the tokens back to Spanish text
  translated_text <- tokenizer$decode(translated_tokens[[1]], skip_special_tokens = TRUE)
  
  return(translated_text)
}

# Example: Translating a news article
news_article <- "The global economy is recovering from the pandemic."
spanish_translation <- translate_to_spanish(news_article)
cat("Spanish Translation: ", spanish_translation, "\n")
