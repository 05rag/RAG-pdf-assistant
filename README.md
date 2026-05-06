# RAG PDF Assistant

An AI-powered document intelligence system built using Retrieval-Augmented Generation (RAG) that enables semantic search, contextual question answering, citation-based responses, and document summarization across multiple PDFs.

---

## Features

- Multi-PDF upload and indexing
- Semantic search using vector embeddings
- Context-aware conversational querying
- Citation-based responses with page references
- Research paper summarization
- Streamlit-based interactive UI
- Retrieval using ChromaDB vector store
- LLM-powered response generation using Groq

---

## Tech Stack

- Python
- Streamlit
- LangChain
- ChromaDB
- PyMuPDF
- Sentence Transformers
- Groq API

---

## Project Workflow

1. Upload PDF documents
2. Extract and clean text using PyMuPDF
3. Split documents into semantic chunks
4. Generate vector embeddings
5. Store embeddings in ChromaDB
6. Retrieve relevant chunks using semantic similarity
7. Generate grounded answers using LLMs

---

## Example Questions

- What is scaled dot-product attention?
- How does the Transformer differ from RNNs?
- Summarize the research paper.
- What are the key contributions of the paper?

---

## Run Locally

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/RAG-pdf-assistant.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Add your API key in `.env`:

```env
GROQ_API_KEY=your_api_key
```

Run the app:

```bash
streamlit run app.py
```

---

## Future Improvements

- Hybrid search (semantic + keyword)
- Better retrieval reranking
- OCR support for scanned PDFs
- Deployment support
- Conversation memory enhancement

