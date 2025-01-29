import os
import logging
import uuid
import json
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from flask import Flask, request, jsonify
from flask_restful import Api, Resource
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_caching import Cache
from bs4 import BeautifulSoup
from langchain_core.documents import Document
import asyncio
# from multiprocessing import Process, Queue # No longer needed with browser-use
from pydantic import BaseModel, Field
from typing import List

# browser-use imports
from browser_use import Agent, Controller  # Import Agent and Controller from browser_use

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Model configuration
generation_config = {
    "temperature": float(os.getenv("TEMPERATURE", 0.7)),
    "top_p": float(os.getenv("TOP_P", 0.95)),
    "top_k": int(os.getenv("TOP_K", 40)),
    "max_output_tokens": int(os.getenv("MAX_OUTPUT_TOKENS", 8192)),
}

model = genai.GenerativeModel(
    model_name=os.getenv("MODEL_NAME", "gemini-2.0-flash-exp"),
    generation_config=generation_config,
)

# Flask setup
app = Flask(__name__)
api = Api(app)
limiter = Limiter(get_remote_address, app=app, default_limits=["10 per minute"])
cache = Cache(app, config={"CACHE_TYPE": "SimpleCache", "CACHE_DEFAULT_TIMEOUT": 300})
sessions = {}

# Pydantic models
class Course(BaseModel):
    title: str = Field(..., description="Title of the course")
    description: str = Field(..., description="Detailed description of the course")

class CourseList(BaseModel):
    courses: List[Course] = Field(..., description="List of available courses")

# Browser automation functions - No longer needed, functionality moved to execute_browser_operation
# def run_browser_task(task, output_model, queue):
#     """Execute browser automation in separate process"""
#     try:
#         controller = Controller(output_model=output_model)
#         agent = Agent(
#             task=task,
#             llm=model,
#             max_actions_per_step=4,
#             controller=controller
#         )
#         result = asyncio.run(agent.run(max_steps=25))
#         queue.put(result.final_result() if result else None)
#     except Exception as e:
#         logger.error(f"Browser task error: {e}")
#         queue.put(None)

# Data processing pipeline
def load_data(url):
    """Load and process website content"""
    try:
        loader = WebBaseLoader(url)
        raw_docs = loader.load()
        logger.info(f"Loaded raw documents from {url}: {len(raw_docs)}") # Log raw docs loaded

        processed_docs = []
        for doc in raw_docs:
            soup = BeautifulSoup(doc.page_content, 'html.parser')
            courses = soup.find_all('div', class_='course-card')
            logger.info(f"Found {len(courses)} course cards on {url}") # Log course cards found

            for course in courses:
                title = course.find('h3').get_text(strip=True)
                description = course.find('div', class_='description').get_text(strip=True)
                processed_docs.append(Document(
                    page_content=f"Course: {title}\nDescription: {description}",
                    metadata={"source": url, "title": title}
                ))
        logger.info(f"Processed documents from {url}: {len(processed_docs)}") # Log processed docs
        return processed_docs
    except Exception as e:
        logger.error(f"Data loading failed for {url}: {e}")
        raise

def create_vector_store(docs):
    """Create/load vector database"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(os.getenv("CHUNK_SIZE", 768)),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 128)),
        separators=["\n\n", "\n", ". ", "! ", "? ", " "]
    )
    split_docs = text_splitter.split_documents(docs)

    return Chroma.from_documents(
        documents=split_docs,
        embedding=HuggingFaceEmbeddings(
            model_name=os.getenv("EMBEDDINGS_MODEL", "sentence-transformers/all-mpnet-base-v2")
        ),
        persist_directory="./chroma_db"
    )

# Initialize data pipeline
try:
    urls = os.getenv("DATA_URLS", "https://brainlox.com/courses/category/technical").split(',')
    all_docs = [load_data(url.strip()) for url in urls]
    logger.info(f"All Docs (before flattening): {[[len(doc_list) for doc_list in all_docs]]}") # Log lengths of doc lists in all_docs
    flat_docs = [doc for sublist in all_docs for doc in sublist]
    logger.info(f"Flattened Docs (length): {len(flat_docs)}") # Log length of flattened docs
    vector_store = create_vector_store(flat_docs)
except Exception as e:
    logger.error(f"Data pipeline initialization failed: {e}")
    exit(1)

# API Resources
class AIChatAgent(Resource):
    @limiter.limit("10/minute")
    def post(self):
        """AI agent endpoint with dynamic browser integration"""
        data = request.get_json()
        user_input = data.get("message", "").strip()
        session_id = data.get("session_id")

        # Validate input
        if not user_input or len(user_input) > 1000:
            return jsonify({"error": "Invalid message format"}), 400

        # Session management
        if not session_id or session_id not in sessions:
            session_id = str(uuid.uuid4())
            sessions[session_id] = {
                "history": [],
                "chat": model.start_chat(history=[])
            }
        session = sessions[session_id]

        try:
            # Context retrieval
            context_docs = vector_store.similarity_search(user_input, k=3)
            context = "\n".join([f"Source: {doc.metadata['source']}\n{doc.page_content}"
                               for doc in context_docs])

            # Dynamic browser task decision
            browser_data = None
            if self.needs_browser_assistance(user_input, context):
                browser_data = self.execute_browser_operation(user_input)
                if browser_data:
                    context += f"\nLive Data:\n{browser_data.json()}"

            # Generate response
            prompt = self.build_agent_prompt(user_input, context, session['history'])
            response = session["chat"].send_message(prompt)

            # Update session
            session["history"].append((user_input, response.text))

            return jsonify({
                "response": response.text,
                "session_id": session_id,
                "fresh_data": browser_data is not None
            })

        except Exception as e:
            logger.error(f"Chat processing error: {e}")
            return jsonify({"error": "Processing failed"}), 500

    def needs_browser_assistance(self, query, context):
        """Determine if browser automation is needed"""
        decision_prompt = f"""Analyze if this query requires live website data:
        Query: {query}
        Existing Context: {context}

        Respond with JSON: {{"needs_browser": bool, "reason": str}}"""

        try:
            response = model.generate_content(decision_prompt)
            return json.loads(response.text).get("needs_browser", False)
        except Exception as e:
            logger.error(f"Browser decision error: {e}")
            return False

    def execute_browser_operation(self, query):
        """Execute dynamic browser automation using browser-use"""
        try:
            task_prompt = f"""Generate browser automation instructions for:
            Query: {query}
            Required Format: 'Navigate to [URL] and extract [elements]'"""

            instructions = model.generate_content(task_prompt).text
            if "brainlox.com" not in instructions:
                raise ValueError("Invalid automation target")

            controller = Controller() # Initialize browser-use Controller
            agent = Agent(
                task=instructions,
                llm=model, # Use the same Gemini model
                max_actions_per_step=4,
                controller=controller
            )

            result = asyncio.run(agent.run(max_steps=25)) # Run browser-use agent
            if result and result.final_result(): # Check if result and final_result exist
                # Assuming final_result returns a Pydantic model or something serializable.
                # If it returns raw text or different format, adjust accordingly.
                return result.final_result()
            else:
                return None

        except Exception as e:
            logger.error(f"Browser execution failed: {e}")
            return None

    def build_agent_prompt(self, query, context, history):
        """Construct AI agent prompt"""
        return f"""You are an AI Education Advisor. Synthesize information from:

        Context:
        {context}

        Conversation History:
        {history[-3:] if len(history) > 3 else history}

        Query: {query}

        Analysis Steps:
        1. Identify key information needs
        2. Combine static knowledge and live data
        3. Consider user's apparent goals
        4. Formulate comprehensive response

        Final Answer:"""

class HealthCheck(Resource):
    def get(self):
        return jsonify({
            "status": "active",
            "model": model.model_name,
            "vector_store": "chroma",
            "sessions": len(sessions)
        })

# Register endpoints
api.add_resource(AIChatAgent, "/chat")
api.add_resource(HealthCheck, "/health")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=os.getenv("DEBUG", "false").lower() == "true")