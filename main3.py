import os
import logging
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from flask import Flask, request, jsonify
from flask_restful import Api, Resource
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_caching import Cache


load_dotenv()


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


generation_config = {
    "temperature": float(os.getenv("TEMPERATURE", 1)),
    "top_p": float(os.getenv("TOP_P", 0.95)),
    "top_k": int(os.getenv("TOP_K", 40)),
    "max_output_tokens": int(os.getenv("MAX_OUTPUT_TOKENS", 8192)),
    "response_mime_type": os.getenv("RESPONSE_MIME_TYPE", "text/plain"),
}

model = genai.GenerativeModel(
    model_name=os.getenv("MODEL_NAME", "gemini-2.0-flash-exp"),
    generation_config=generation_config,
)


chat_session = model.start_chat(history=[])


def load_data(url):
    try:
        logger.info("Loading data from URL...")
        loader = WebBaseLoader(url)
        data = loader.load()
        logger.info("Data loaded successfully.")
        return data
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def split_documents(data):
    try:
        logger.info("Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(os.getenv("CHUNK_SIZE", 1000)),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 200)),
        )
        docs = text_splitter.split_documents(data)
        logger.info(f"Documents split into {len(docs)} chunks.")
        return docs
    except Exception as e:
        logger.error(f"Error splitting documents: {e}")
        raise


def create_vector_store(docs):
    try:
        logger.info("Creating embeddings and vector store...")
        embeddings = HuggingFaceEmbeddings(model_name=os.getenv("EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
        vector_store = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_db")
        vector_store.persist()
        logger.info("Vector store created successfully.")
        return vector_store
    except Exception as e:
        logger.error(f"Error creating vector store: {e}")
        raise


url = os.getenv("DATA_URL", "https://brainlox.com/courses/category/technical")
data = load_data(url)
docs = split_documents(data)
vector_store = create_vector_store(docs)


app = Flask(__name__)
api = Api(app)


limiter = Limiter(get_remote_address, app=app, default_limits=["5 per minute"])


cache = Cache(app, config={"CACHE_TYPE": "SimpleCache"})


chat_history = []


class Chatbot(Resource):
    @limiter.limit("5 per minute")
    @cache.cached(timeout=60)  
    def post(self):
        try:
            data = request.get_json()
            user_input = data.get("message")

            
            if not user_input or not isinstance(user_input, str):
                logger.warning("Invalid input received.")
                return jsonify({"error": "Invalid input. 'message' must be a non-empty string."}), 400

            logger.info(f"Received user input: {user_input}")

            
            relevant_docs = vector_store.similarity_search(user_input, k=int(os.getenv("SIMILARITY_SEARCH_K", 3)))
            context = " ".join([doc.page_content for doc in relevant_docs])

            
            structured_input = f"Context: {context}\n\nQuestion: {user_input}"
            response = chat_session.send_message(structured_input)
            bot_response = response.text

            
            chat_history.append((user_input, bot_response))

            logger.info(f"Generated response: {bot_response}")
            return jsonify({"response": bot_response})

        except Exception as e:
            logger.error(f"Error processing request: {e}")
            return jsonify({"error": "An internal server error occurred."}), 500


api.add_resource(Chatbot, "/chat")


if __name__ == "__main__":
    app.run(debug=os.getenv("FLASK_DEBUG", "False").lower() == "true")