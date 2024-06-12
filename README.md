# documentation-rag-poc
Agentic Retrieval Augmented Generation (A-RAG) for documentation answering

<img width="679" alt="image" src="https://github.com/Jhonfel/documentation-rag-agentic-workflow/assets/6430163/bc25391d-4f96-406f-b109-65476487a4df">


## Overview
Project Clementine aims to streamline the process of navigating extensive documentation for developers at Company X. This tool reduces the time spent searching for information and increases overall productivity. The Proof of Concept (POC) developed in collaboration with Loka focuses on enhancing developer experience by providing quick and accurate answers to common documentation-related queries.

## Features
- **Intelligent Query Handling**: Utilizes natural language processing to understand and respond to queries like "What is SageMaker?".
- **Advanced Navigation Assistance**: Directs developers to specific parts of documentation relevant to their queries.
- **Real-Time Updates**: Incorporates mechanisms to handle updates in documentation to provide current information.
- **Scalable and Secure**: Cloud-compatible design ensures scalability and secure handling of sensitive documentation.

## Architecture
### C4 Model Diagrams
Below are links to C4 Model diagrams that detail the various layers of the system architecture:
- **Context Diagram**: Overview of the system's interaction with external entities.
<img width="475" alt="image" src="https://github.com/Jhonfel/documentation-rag-agentic-workflow/assets/6430163/da4f4586-a1db-4bbf-bc62-7aa33721da9e">

- **Container Diagram**: Breaks down the system into its major containers showing the deployment of Flask services and React frontend.
- **Component Diagram**: Details the components involved in the retrieval and generation of answers.
- **Code Diagram**: Provides insights into the structure of the Flask and React codebases.

### Retrieval-Augmented Generation (RAG) Workflow
Employs a RAG workflow to enhance both information retrieval and answer accuracy:
- **Document Retrieval**: Leveraging OpenAI embeddings for dynamic documentation fetching.
- **Answer Generation**: Utilizes LangChain to generate context-aware responses based on retrieved documents.

## Technology Stack
- **AWS Fargate**: For cloud hosting and deployment.
- **Flask**: Serves as the backend framework to manage API requests and responses.
- **React**: Powers the frontend, providing a responsive user interface.
- **LangChain**: Integrates with language models to facilitate the RAG workflow.
- **OpenAI Embeddings**: Used to improve the relevance and precision of document retrieval.

## Setup and Installation
Instructions on setting up the project locally:
1. Clone the repository:
