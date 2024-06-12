from flask import Flask, send_from_directory, request, jsonify
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
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI

app = Flask(__name__)

# loading file
loader = TextLoader('storage/knowledge.txt')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
text_splitter = RecursiveCharacterTextSplitter(
chunk_size=1000, chunk_overlap=0, separators=[" ", ",", "\n"]
)
chunks = text_splitter.split_documents(documents)
vectorstore = Chroma.from_documents(
    documents=chunks,
    collection_name="ollama_embeds",
    embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
)
retriever = vectorstore.as_retriever()


model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.1)


rag_template = """You are an assistant for specific documentation for sagemaker query tasks. 
   Use the following pieces of retrieved context to answer the question. 
   If you don't know the answer, just say that you don't know. 
   Question: {question} 
   Documentation: {context} 
   Answer:
   """
rag_prompt = ChatPromptTemplate.from_template(rag_template)
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | rag_prompt
    | model
    | StrOutputParser()
)


# serve index.html
@app.route('/',  methods=["GET",'POST'])
def serve_index():
    return send_from_directory('rag-frontend/dist', 'index.html')

# serve static files
@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('rag-frontend/dist', path)


@app.route('/ask', methods=['POST'])
def ask_question():
    # Retrieve the JSON body of the request
    data = request.get_json()
    
    # Check if the JSON contains the key 'question'
    if 'question' not in data:
        return jsonify({'error': 'No question provided'}), 400
    
    # Extract the question from the received JSON
    question = data['question']
    
    try:
        # Use rag_chain to obtain an answer based on the question
        response = rag_chain.invoke(question)
        # Return the response in JSON format
        return jsonify({'response': response}), 200
    except Exception as e:
        # Handle general errors
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
