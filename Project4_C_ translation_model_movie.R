library(reticulate)
transformers <- import('transformers')

# Load MarianMT model for English to another language (e.g., English to French)
load_translation_model <- function(model_name = "Helsinki-NLP/opus-mt-en-fr") {
  tokenizer <- transformers$MarianTokenizer$from_pretrained(model_name)
  model <- transformers$MarianMTModel$from_pretrained(model_name)
  list(model = model, tokenizer = tokenizer)
}
