import os
import tempfile
from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re

def process_multiple_uploads(uploaded_files):
    all_docs=[]
    for uploaded_file in uploaded_files:
        docs = process_uploaded_files(uploaded_file)
        all_docs.extend(docs)
    return all_docs

def process_uploaded_files(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    try:
        loader = PyMuPDFLoader(tmp_path)
        documents = loader.load()

        if not documents or all(doc.page_content.strip() == "" for doc in documents):
            print(f"No extractable text in {uploaded_file.name}")
            return []

        #Adding source information to metadata
        for doc in documents:
            doc.metadata["source"] = uploaded_file.name
            doc.metadata["page"] = doc.metadata.get("page", 0)
            
    finally:
        os.remove(tmp_path)

    return documents


#Text Splitting get into chunks
def split_documents(documents, chunk_size=1000, chunk_overlap = 200):
    """Split the documents into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        length_function = len,
        separators = ["\n\n",   # sections
                    "\n•",
                    "\n-",
                    "\n",
                    ". ",
                    " " ]                          
    )

    split_docs = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(split_docs)} chunks")

    #Show example of a chunk
    if split_docs:
        print(f"\nExample chunk:")
        print(f"content: {split_docs[0].page_content[:500]}...")
        print(f"metadata: {split_docs[0].metadata}")
    return split_docs



