import os
import streamlit as st
import pandas as pd
from src.engine import EmbeddingManager, VectorStore, RAGRetriever
from src.processor import process_multiple_uploads, process_uploaded_files, split_documents
from src.pipeline import AdvancedRAGPipeline
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import uuid
import re

load_dotenv()
os.getenv("GROQ_API_KEY")

if "session_id" not in st.session_state:
    st.session_state.session_id = f"user_{uuid.uuid4().hex[:12]}"

if "processed" not in st.session_state:
    st.session_state.processed = False

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "embed_mgr" not in st.session_state:
    st.session_state.embed_mgr = None

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = set()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

#upload your file
uploaded_files = st.file_uploader(
    "Upload your PDF", accept_multiple_files=True, type="pdf"
)

@st.cache_resource
def get_vector_store(session_id):
    return VectorStore(collection_name=session_id) 

#processing
if uploaded_files:
    new_files = [f for f in uploaded_files if f.name not in st.session_state.indexed_files]

    if len(new_files) > 0:
        if st.button(f"Analyse PDF"):
            with st.spinner("Reading and indexing your documents..."):

                documents = process_multiple_uploads(new_files)
                if not documents:
                    st.error(" No text could be extracted. The PDF might be an image or encrypted.")
                    st.stop()
                
                chunks = split_documents(documents)

                if st.session_state.embed_mgr is None:
                    st.session_state.embed_mgr = EmbeddingManager()
                if st.session_state.vector_store is None:
                    st.session_state.vector_store = get_vector_store(st.session_state.session_id)

                def clean_text(text):
                    # Remove extra whitespaces, tabs, and newlines
                    text = re.sub(r'[ \t]+', ' ', text) 
                    return text.strip()
                
                cleaned_texts = []
                for chunk in chunks:
                    cleaned_content = clean_text(chunk.page_content)
                    chunk.page_content = cleaned_content  # This is the vital step!
                    cleaned_texts.append(cleaned_content)

                embeddings = st.session_state.embed_mgr.generate_embeddings(cleaned_texts)
                st.session_state.vector_store.add_documents(chunks, embeddings)

                for f in new_files:
                    st.session_state.indexed_files.add(f.name)

                st.session_state.processed = True
            st.success("Analysis Complete!")
            st.rerun()
    else:
        st.info("All uploaded files are already indexed. You can start chatting!")


with st.sidebar:
    st.title("Saved Conversations")
    if st.button("Clear History"):
        st.session_state.chat_history = []
        st.rerun()
    
    st.divider()
    # Display historical questions as a list of "logs"
    for i, chat in enumerate(reversed(st.session_state.chat_history)):
        with st.expander(f"Q: {chat['question'][:30]}..."):
            st.write(f"**A:** {chat['answer']}")
            if chat.get('sources'):
                st.caption(f"Sources: {', '.join(chat['sources'])}")


if st.session_state.processed:

    all_files = list(st.session_state.indexed_files)
    col1, col2 = st.columns([1, 3])
    
    with col1:
        # Add "All Files" as an option for cross-document questions
        selected_file = st.selectbox(
            "Focus on:", 
            options=["All Documents"] + all_files
        )
    
    with col2:
        query = st.chat_input("Enter your query")

    if query:
        filter_filename = None if selected_file == "All Documents" else selected_file
        with st.chat_message("user"):
            st.write(query)

        llm = ChatGroq(
            model = 'llama-3.1-8b-instant',
            temperature = 0,
            max_tokens= 1024
        )
        my_retriever = RAGRetriever(
            vector_store=st.session_state.vector_store, 
            embedding_manager=st.session_state.embed_mgr
        )
        my_pipeline = AdvancedRAGPipeline(retriever = my_retriever, llm = llm)
        
        with st.spinner(f"Searching in {selected_file}..."):
            results = my_pipeline.query(
                query, 
                source_file=filter_filename, 
                top_k=5)

        st.markdown("Assistant Response")
        st.write(results['answer'])

        with st.expander(" View References & Sources"):
            for idx, src in enumerate(results['sources']):
                st.info(f"Source {idx+1}: {src['source']} (Page {src['page']})")
                st.caption(f"Similarity Score: {src['score']:.4f}")
                st.write(src['preview'])
        
        source_list = [src['source'] for src in results.get('sources', [])]
        st.session_state.chat_history.append({
            "question": query,
            "answer": results['answer'],
            "sources": list(set(source_list))
        })

