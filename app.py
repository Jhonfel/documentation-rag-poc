from flask import Flask, send_from_directory

from langchain_community.document_loaders import TextLoader # Text loader
from langchain.text_splitter import CharacterTextSplitter # Text splitter
from langchain_community.embeddings import OllamaEmbeddings # Ollama embeddings
from langchain.prompts import ChatPromptTemplate # Chat prompt template
from langchain_community.chat_models import ChatOllama # ChatOllma chat model
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser # Output parser
from langchain_community.vectorstores import Weaviate # Vector database
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter

app = Flask(__name__)





# serve index.html
@app.route('/',  methods=["GET",'POST'])
def serve_index():
    return send_from_directory('rag-frontend/dist', 'index.html')

# serve static files
@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('rag-frontend/dist', path)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
