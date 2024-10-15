library(reticulate)
transformers <- import('transformers')

# Function to load the MarianMT model for English to French translation
load_translation_model <- function() {
  model_name <- "Helsinki-NLP/opus-mt-en-fr"  # Model for English to French translation
  tokenizer <- transformers$MarianTokenizer$from_pretrained(model_name)
  model <- transformers$MarianMTModel$from_pretrained(model_name)
  list(model = model, tokenizer = tokenizer)
}
