import os
import logging
import uuid
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
from browser_use import Agent, Controller
from multiprocessing import Process, Queue
from pydantic import BaseModel, Field
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


try:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    
except Exception as e:
    logger.error(f"Configuration error: {e}")
    exit(1)



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

intent_model = genai.GenerativeModel( 
    model_name=os.getenv("RESPONSE_MODEL_NAME", "gemini-2.0-flash-exp"),
    generation_config=generation_config,
)


def get_limiter_key():
    """Custom rate limiter key function using session ID when available"""
    try:
        if request.is_json:
            session_id = request.get_json().get("session_id")
            if session_id:
                return session_id
    except Exception:
        pass
    return get_remote_address()

app = Flask(__name__)
api = Api(app)
limiter = Limiter(get_limiter_key, app=app, default_limits=["10 per minute"]) 
cache = Cache(app, config={"CACHE_TYPE": "SimpleCache", "CACHE_DEFAULT_TIMEOUT": 300})


sessions = {}

def load_data(url):
    """Enhanced data loader with HTML content extraction"""
    try:
        logger.info(f"Loading data from: {url}")
        loader = WebBaseLoader(url)
        raw_docs = loader.load()
        processed_docs = []

        for doc in raw_docs:
            try:
                soup = BeautifulSoup(doc.page_content, 'html.parser')

                courses = soup.find_all('div', class_='course-card')
                for course in courses:
                    try:
                        title = course.find('h3').get_text(strip=True)
                        description = course.find('div', class_='description').get_text(strip=True)
                        content = f"Course Title: {title}\nDescription: {description}"
                        processed_docs.append(Document(
                            page_content=content,
                            metadata={"source": url, "title": title}
                        ))
                    except Exception as e:
                        logger.warning(f"Error processing course: {e}")
                        continue
            except Exception as e:
                  logger.error(f"Error parsing content {e}")
                  continue
        logger.info(f"Extracted {len(processed_docs)} courses from {url}")
        return processed_docs
    except Exception as e:
        logger.error(f"Data loading failed: {e}")
        raise

def split_documents(data):
    """Optimal text splitting for technical content"""
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(os.getenv("CHUNK_SIZE", 768)),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 128)),
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )
        return text_splitter.split_documents(data)
    except Exception as e:
        logger.error(f"Error splitting documents: {e}")
        raise

def create_vector_store(docs):
    """Vector store with persistence management"""
    persist_dir = "./chroma_db"
    try:
         embeddings = HuggingFaceEmbeddings(
            model_name=os.getenv("EMBEDDINGS_MODEL", "sentence-transformers/all-mpnet-base-v2")
        )
         if os.path.exists(persist_dir) and not os.getenv("REFRESH_VECTOR_STORE", "false").lower() == "true":
            logger.info("Loading existing vector store")
            return Chroma(persist_directory=persist_dir, embedding_function=embeddings)

         logger.info("Creating new vector store")
         vector_store = Chroma.from_documents(
             documents=docs,
            embedding=embeddings,
             persist_directory=persist_dir
        )
         vector_store.persist()
         return vector_store
    except Exception as e:
        logger.error(f"Error creating or loading vector store: {e}")
        raise

class Course(BaseModel):
    title: str
    description: str

class CourseList(BaseModel):
    courses: List[Course]


def run_browser_task(task, output_model, queue):
    """Function to run browser task in a separate process."""
    api_key = os.getenv("GEMINI_API_KEY")
    llm = ChatGoogleGenerativeAI(model=os.getenv("BROWSER_LLM_MODEL", "gemini-2.0-flash-exp"), api_key=api_key)
    controller = Controller(output_model=output_model)
    agent = Agent(
        task=task,
        llm=llm,
        max_actions_per_step=4,
        controller=controller
    )
    try:
        result = asyncio.run(agent.run(max_steps=25))
        if result.final_result():
           queue.put(result.final_result())
        else:
            queue.put(None)
    except Exception as e:
        logger.error(f"Error running browser task: {e}")
        queue.put(None)


try:
    urls = os.getenv("DATA_URLS", "https://brainlox.com/courses/category/technical").split(',')
    all_docs = []
    for url in urls:
        all_docs.extend(load_data(url.strip()))
    split_docs = split_documents(all_docs)
    vector_store = create_vector_store(split_docs)
except Exception as e:
    logger.error(f"Failed to initialize data pipeline: {e}")
    exit(1)

class ChatBot(Resource):
    @limiter.limit("10/minute") 
    def post(self):
        """Enhanced chat endpoint with session management and dynamic browser tasking - Authentication Removed"""

        
        
        
        

        data = request.get_json()
        user_input = data.get("message", "").strip()
        session_id = data.get("session_id")

        if not user_input or len(user_input) > 1000:
            return jsonify({"error": "Invalid message format"}), 400


        if not session_id or session_id not in sessions:
            session_id = str(uuid.uuid4())
            sessions[session_id] = {
                "chat": intent_model.start_chat(history=[]), 
                "history": []
            }

        session = sessions[session_id]

        try:
            
            intent_prompt = f"""Determine if the user query requires using a web browser to find up-to-date information about technical courses on 'https://brainlox.com/courses/category/technical'.

            User Query: {user_input}

            Constraints:
            - Focus on queries about course information, specifically from brainlox.com.
            - If the query is about listing courses, finding specific course details, or exploring course categories, it likely requires browsing.
            - If the query is general knowledge, unrelated to courses, or can be answered from existing knowledge, it does not require browsing.
            - If browsing is needed, generate a concise browser_task string using 'extract_content' to get relevant course details (title and description). Aim to extract a CourseList. Be specific about the website.
            - If no browser task is needed, respond with "NO_BROWSER_TASK".

            Response (Browser Task or NO_BROWSER_TASK):
            """

            intent_response = model.send_message(intent_prompt) 
            browser_task_string = intent_response.text.strip()
            logger.info(f"Intent Model Response: {browser_task_string}")


            if browser_task_string.lower() != "no_browser_task":
                
                queue = Queue()
                process = Process(target=run_browser_task, args=(browser_task_string, CourseList, queue))

                process.start()
                process.join(timeout=120)
                if process.is_alive():
                    process.terminate()
                    process.join()
                    return jsonify({"error": "Browser task timed out."}), 504

                browser_result = queue.get()

                if browser_result:
                    bot_response = f"Course Information from Browser:\n{browser_result}"
                else:
                     bot_response = "Could not get the course information from the browser."

                session["history"].append((user_input, bot_response))
                return jsonify({
                 "response": bot_response,
                  "session_id": session_id,
                   "sources": ["browser"] 
                    })


            else: 
                relevant_docs = vector_store.similarity_search(user_input, k=3)
                context = "\n".join([f"Source: {doc.metadata['source']}\n{doc.page_content}"
                                   for doc in relevant_docs])

                prompt = f"""You are a technical course advisor. Use this context to answer the question:
                {context}

                Current conversation:
                {session['history']}

                Question: {user_input}
                Helpful Answer:"""

                response = session["chat"].send_message(prompt) 
                bot_response = response.text

                session["history"].append((user_input, bot_response))

                return jsonify({
                    "response": bot_response,
                    "session_id": session_id,
                    "sources": list({doc.metadata["source"] for doc in relevant_docs})
                })

        except Exception as e:
            logger.error(f"Chat error: {e}")
            return jsonify({"error": "Processing error"}), 500

class HealthCheck(Resource):
    def get(self):
        return jsonify({"status": "ok", "model": model.model_name})

api.add_resource(ChatBot, "/chat")
api.add_resource(HealthCheck, "/health")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=os.getenv("DEBUG", "false").lower() == "true")