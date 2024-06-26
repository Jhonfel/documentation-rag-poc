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

![image](https://github.com/Jhonfel/documentation-rag-poc/assets/6430163/f8b4a12e-024d-4931-9851-f8c7d2f13445)


### C4 Model Diagrams
Below are links to C4 Model diagrams that detail the various layers of the system architecture:
- **Context Diagram**: Overview of the system's interaction with external entities.
<img width="475" alt="image" src="https://github.com/Jhonfel/documentation-rag-agentic-workflow/assets/6430163/da4f4586-a1db-4bbf-bc62-7aa33721da9e">

- **Container Diagram**: Breaks down the system into its major containers showing the deployment of Flask services and React frontend.
- **Component Diagram**: Details the components involved in the retrieval and generation of answers.
- **Code Diagram**: Provides insights into the structure of the Flask and React codebases.

### Retrieval-Augmented Generation (RAG) Workflow
![image](https://github.com/Jhonfel/documentation-rag-poc/assets/6430163/18546bcb-d487-4d8b-80de-13ac937d11e3)

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
   ```bash
   git clone https://github.com/Jhonfel/documentation-rag-poc.git
   cd documentation-rag-poc
   ```
2. Build the project using the build script:
   ```bash
   ./build.sh
   ```
3. Start the application with the required environment variable for the OpenAI API key:
   ```bash
   OPENAI_API_KEY=your_openai_api_key ./start.sh
   ```

## Deployment in Production
Instructions for deploying the project in production using AWS Fargate:
1. Build your Docker image and tag it appropriately:
   ```bash
   docker build -t your-registry/your-image-name:tag .
   ```
2. Push the Docker image to your container registry (e.g., AWS ECR):
   ```bash
   docker push your-registry/your-image-name:tag
   ```
3. Create a secret in AWS to store the OpenAI API key:
   ```bash
   aws secretsmanager create-secret --name OpenAIKey --secret-string "{"OPENAI_API_KEY":"your_openai_api_key"}"
   ```
4. Deploy your Docker container on AWS Fargate:
   - Ensure that your task definition references the secret for the OpenAI API key.
   - Configure your Fargate service to use the pushed Docker image.

   You can configure these settings through the AWS Management Console or by using the AWS CLI.
