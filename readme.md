# 📚 Enterprise Document Assistant

A conversational **PDF-based Retrieval-Augmented Generation (RAG)** application that allows users to upload PDF documents and ask questions about their contents.

The application combines **FastAPI, LangChain, ChromaDB, Hugging Face embeddings, and Groq LLMs** to provide context-aware answers from uploaded documents.

## 🚀 Features

- 📄 Upload and process PDF documents directly from the web interface
- 🔍 Semantic document retrieval using vector embeddings
- 🧠 Conversational question answering using chat history
- 🔄 Contextual question rewriting for follow-up questions
- 📚 MMR-based retrieval using ChromaDB
- 🤖 Groq-powered LLM responses
- 💬 Browser-side conversation history
- 📝 Markdown rendering for AI responses
- ➗ Mathematical expression rendering using MathJax
- ⚡ FastAPI backend with REST endpoints
- 🌐 Simple HTML, CSS, and JavaScript frontend

## 🏗️ Architecture

```text
                         User
                           │
                           ▼
                ┌─────────────────────┐
                │    Web Interface    │
                │  HTML / CSS / JS    │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │       FastAPI       │
                │                     │
                │    /query           │
                │    /upload_pdf      │
                └──────────┬──────────┘
                           │
             ┌─────────────┴─────────────┐
             │                           │
             ▼                           ▼
      PDF Ingestion                Query Processing
             │                           │
             ▼                           ▼
      Text Extraction           Question Rewriting
             │                           │
             ▼                           ▼
       Text Chunking                 ChromaDB
             │                    Vector Retrieval
             ▼                           │
   Hugging Face Embeddings               │
             │                           ▼
             └───────────────►     Retrieved Context
                                         │
                                         ▼
                                    Groq LLM
                                         │
                                         ▼
                                      Answer