import os
from flask import Flask, request
from flask_restful import Resource, Api
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.llms import GoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
api = Api(app)


os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
if not os.environ["GOOGLE_API_KEY"]:
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")


URL = "https://brainlox.com/courses/category/technical"


class Chatbot(Resource):
    def __init__(self):

        
        loader = WebBaseLoader(URL)
        data = loader.load()

        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(data)

        
        embeddings = OpenAIEmbeddings()

        
        persist_directory = "chroma_db"
        self.vectordb = Chroma.from_documents(
            documents=texts, embedding=embeddings, persist_directory=persist_directory
        )

        
        self.vectordb.persist()

        
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