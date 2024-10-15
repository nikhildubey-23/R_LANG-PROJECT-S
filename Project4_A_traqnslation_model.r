library(reticulate)
transformers <- import('transformers')

# Load the MarianMT pre-trained model for English to Spanish translation
load_translation_model <- function() {
  model_name <- "Helsinki-NLP/opus-mt-en-es"
  tokenizer <- transformers$MarianTokenizer$from_pretrained(model_name)
  model <- transformers$MarianMTModel$from_pretrained(model_name)
  list(model = model, tokenizer = tokenizer)
}
