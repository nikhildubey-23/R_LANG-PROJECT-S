# Install and load the necessary packages
install.packages("httr")
install.packages("jsonlite")

library(httr)
library(jsonlite)

# Set your OpenAI API key (replace this with your actual key)
api_key <- "cREATE YOUR oWN API KEY"

# Function to send a request to OpenAI API and get the answer
get_openai_answer <- function(question) {
  url <- "https://api.openai.com/v1/chat/completions"  # Updated to chat completions endpoint
  
  headers <- add_headers(
    Authorization = paste("Bearer", api_key),
    `Content-Type` = "application/json"
  )
  
  body <- list(
    model = "gpt-3.5-turbo",  # Updated model
    messages = list(
      list(role = "system", content = "You are a helpful assistant.AND YOU CAN PROVIDE THE ANSWER ABOUT THE art of painting"),
      list(role = "user", content = question)
    ),
    max_tokens = 200  # You can adjust the response length (in tokens)
  )
  
  body_json <- toJSON(body, auto_unbox = TRUE)
  
  response <- POST(url, headers, body = body_json)
  
  if (status_code(response) == 200) {
    result <- content(response, "text")
    parsed_result <- fromJSON(result)
    return(parsed_result$choices[[1]]$message$content)  # Updated for chat response format
  } else {
    return(paste("Error:", status_code(response), content(response, "text")))
  }
}

# Prompt the user to enter their question
user_question <- readline(prompt = "Ask a question about U.S. history: ")

# Get the answer from the OpenAI API based on user input
answer <- get_openai_answer(user_question)

# Print the answer
cat("Answer:", answer)
