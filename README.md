# Triluxo Tech Assignment
[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-%23000.svg?style=flat&logo=flask&logoColor=white)](https://flask.palletsprojects.com/en/3.0.x/)
[![Langchain](https://img.shields.io/badge/Langchain-000000?style=flat&logo=langchain&logoColor=white)](https://www.langchain.com/)
[![Google Gemini](https://img.shields.io/badge/Google%20Gemini-%234285F4.svg?style=flat&logo=google-gemini&logoColor=white)](https://ai.google.dev/)

This project implements a custom chatbot using Langchain, Google Gemini AI, and Flask. It extracts course data from Brainlox, creates embeddings for semantic search, and exposes a RESTful API for conversational interactions.

## Overview

The chatbot works by:

1.  **Fetching Data:** It loads course data from a specified Brainlox URL using Langchain's `WebBaseLoader`.
2.  **Parsing Content:** It parses the HTML and extracts course titles and descriptions.
3.  **Splitting Text:** Long texts are split into smaller chunks for better processing.
4.  **Creating Embeddings:** It converts the text chunks into numerical embeddings and stores them in a Chroma vector database.
5.  **Serving the API:** A Flask RESTful API is set up to handle user queries, retrieve context, and generate responses using Google Gemini.
6.  **Managing Sessions:** It manages user sessions to provide context-aware responses.
7.  **Rate Limiting and Caching:** Applies rate limiting to the API endpoint to prevent abuse and implements caching to improve response times.

## Example I/O

### Input API Request
    ```bash
    curl -X POST \
      -H "Content-Type: application/json" \
      -d '{"message": "What courses are available?"}' \
      http://0.0.0.0:5000/chat
    ```
### Output

![Screenshot 2025-01-29 105358](https://github.com/user-attachments/assets/00ecb87c-483e-4a84-9195-f39331952c9c)


## Prerequisites

Before you start, make sure you have the following:

-   **Python 3.8+**
-   **A Google Gemini API Key:** Get one from the [Google AI Studio](https://aistudio.google.com/).
-   **Environment Variables Set Up:** Create a `.env` file in the project root, and add your API key and other configurations. Example `.env` (replace placeholders with your actual values):

    ```env
    GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
    DATA_URL="https://brainlox.com/courses/category/technical"
    CHUNK_SIZE=1000
    CHUNK_OVERLAP=200
    EMBEDDINGS_MODEL="sentence-transformers/all-MiniLM-L6-v2"
    TEMPERATURE=1
    TOP_P=0.95
    TOP_K=40
    MAX_OUTPUT_TOKENS=8192
    RESPONSE_MIME_TYPE="text/plain"
    MODEL_NAME="gemini-2.0-flash-exp"
    SIMILARITY_SEARCH_K=3
    FLASK_DEBUG=True
    ```

    *   **`GEMINI_API_KEY`**: **REQUIRED:** Your Google Gemini API key.
    *   **`DATA_URL`**: URL of the source data (default: `https://brainlox.com/courses/category/technical`).
    *   **`CHUNK_SIZE`**: Size of text chunks (default: `1000`).
    *   **`CHUNK_OVERLAP`**: Overlap between chunks (default: `200`).
    *  **`EMBEDDINGS_MODEL`**: Hugging Face embeddings model (default: `sentence-transformers/all-MiniLM-L6-v2`).
    *   **`TEMPERATURE`**: Sampling temperature for the model (default: `1`).
    *   **`TOP_P`**: Top-p sampling parameter (default: `0.95`).
    *   **`TOP_K`**: Top-k sampling parameter (default: `40`).
    *   **`MAX_OUTPUT_TOKENS`**: Maximum tokens in a response (default: `8192`).
    *   **`RESPONSE_MIME_TYPE`**: Response type, used to return a plain text response (default: text/plain).
    *   **`MODEL_NAME`**: Name of the Gemini model (default: `gemini-2.0-flash-exp`).
    *  **`SIMILARITY_SEARCH_K`**: The number of documents to retrieve from vector DB (default: `3`).
     *  **`FLASK_DEBUG`**: Enable Flask debug mode (default: `True`).

## Setup and Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/Kaos599/Triluxo-Technologies-Assignment.git
    ```

2.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3. **Set up .env file**
Create a `.env` file in the project root, and add all environment variables specified above, make sure to replace all the placeholders with your correct values.

## Running the Application

1.  **Run the Flask app:**

    ```bash
    python app.py
    ```

    The application will start on `http://0.0.0.0:5000`.

## API Usage

The application provides the following API endpoints:

### 1. `/chat` (POST)

Send messages to the chatbot and receive responses.

*   **Method:** `POST`
*   **Headers:**
    *   `Content-Type: application/json`
*   **Request Body:**

    ```json
    {
        "message": "Your query here",
        "session_id": "Optional session id"
    }
    ```

    *   `message`: (required, string) The user's message.
    *   `session_id`: (optional, string) Session ID for ongoing conversations.
*   **Response Body:**

    ```json
    {
      "response": "Chatbot's response message",
      "session_id": "The active session ID",
      "sources": ["list of urls of the data sources used to generate the response"]
    }
    ```

    *   `response`: (string) The chatbot's response.
    *   `session_id`: (string) Session ID of the current session.
    * `sources`:(list of strings) A list of URL source where the data was found to generate the answer.

*   **Example `curl` Request:**

    ```bash
    curl -X POST \
      -H "Content-Type: application/json" \
      -d '{"message": "What courses are available?"}' \
      http://0.0.0.0:5000/chat
    ```

### 2. `/health` (GET)

Check the status of the application.

*   **Method:** `GET`
*   **Response Body:**

    ```json
    {
        "status": "ok",
        "model": "gemini-2.0-flash-exp"
    }
    ```

    *   `status`: (string) The status, which should be "ok".
    *   `model`: (string) The name of the model in use.

*   **Example `curl` Request:**

    ```bash
     curl http://0.0.0.0:5000/health
    ```

## Important Notes

-   The vector database is saved locally in the `chroma_db` directory. It will only be recreated if the `REFRESH_VECTOR_STORE` environment variable is set to `"true"` (not in the example env provided).
-   The data loading is enhanced to parse HTML and extract course information. The parsing logic in the `load_data` function might need updates if the target website changes.
-   **Rate Limiting:** The `/chat` endpoint is rate-limited to 10 requests per minute.
-   **Error Handling:** Detailed logs are provided.
-   **Session Management:** Sessions are stored in memory, not persistently.
-   **Caching:** Responses are cached for 5 minutes to improve response times.
