import os
import google.generativeai as genai
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from flask import Flask, request, jsonify
from flask_restful import Api, Resource