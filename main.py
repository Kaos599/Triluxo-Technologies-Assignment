import os
import google.generativeai as genai
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from flask import Flask, request, jsonify
from flask_restful import Api, Resource

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",
    generation_config=generation_config,
)

chat_session = model.start_chat(history=[])


url = "https://brainlox.com/courses/category/technical"
loader = WebBaseLoader(url)
data = loader.load()


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(data)


app = Flask(__name__)
api = Api(app)


chat_history = []

class Chatbot(Resource):
    def post(self):
        data = request.get_json()
        user_input = data.get("message")
        
        if not user_input:
            return jsonify({"error": "No message provided"}), 400
        
        relevant_docs = vector_store.similarity_search(user_input, k=3)
        context = " ".join([doc.page_content for doc in relevant_docs])
        
        structured_input = f"Context: {context}\n\nQuestion: {user_input}"
        response = chat_session.send_message(structured_input)
        bot_response = response.text
        chat_history.append((user_input, bot_response))
        
        return jsonify({"response": bot_response})


api.add_resource(Chatbot, "/chat")

if __name__ == "__main__":
    app.run(debug=True)