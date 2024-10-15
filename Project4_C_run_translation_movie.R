library(stringr)
source("translation_model_movie.R")

# Function to load and translate subtitle lines from English to French
translate_subtitles <- function(subtitle_file, output_file) {
  loaded_model <- load_translation_model() # nolint: object_usage_linter.
  tokenizer <- loaded_model$tokenizer
  model <- loaded_model$model
  
  # Read subtitle file (.srt)
  lines <- readLines(subtitle_file)
  
  translated_lines <- sapply(lines, function(line) {
    if (str_detect(line, "^[0-9]+$") || str_detect(line, "^\\d{2}:\\d{2}:\\d{2}")) {
      # Skip subtitle number and timecodes
      return(line)
    } else if (nchar(line) > 0) {
      # Translate only subtitle text
      inputs <- tokenizer$encode(line, return_tensors = "pt")
      translated_tokens <- model$generate(inputs)
      translated_text <- tokenizer$decode(translated_tokens[[1]], skip_special_tokens = TRUE)
      return(translated_text)
    } else {
      return(line)
    }
  })
  
  # Save translated subtitles back to a new file
  writeLines(translated_lines, output_file)
}

# Example of translating a subtitle file
translate_subtitles("movie_subtitle.srt", "translated_movie_subtitle.srt")
