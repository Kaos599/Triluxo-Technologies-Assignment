import os
from flask import Flask, request
from flask_restful import Resource, Api
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import GoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
api = Api(app)

# Set the GOOGLE_API_KEY environment variable
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
if not os.environ["GOOGLE_API_KEY"]:
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")

# URL of the website to scrape
URL = "https://brainlox.com/courses/category/technical"


class Chatbot(Resource):
    def __init__(self):

        # Load data
        loader = WebBaseLoader(URL)
        data = loader.load()

        # Split data into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(data)

        # Create embeddings
        embeddings = OpenAIEmbeddings()

        # Create a vector store (Chroma)
        persist_directory = "chroma_db"
        self.vectordb = Chroma.from_documents(
            documents=texts, embedding=embeddings, persist_directory=persist_directory
        )

        # persist the vector db
        self.vectordb.persist()

        # create retrieval chain
        self.qa = RetrievalQA.from_chain_type(
             llm=GoogleGenerativeAI(model="gemini-pro"), chain_type="stuff", retriever=self.vectordb.as_retriever()
        )

    def post(self):
        data = request.get_json()
        query = data.get("query")
        if not query:
            return {"error": "No query provided"}, 400

        try:
            response = self.qa({"query": query})
            return {"response": response["result"]}

        except Exception as e:
            return {"error": f"An error occurred: {str(e)}"}, 500


api.add_resource(Chatbot, "/chat")


if __name__ == "__main__":
    app.run(debug=True)