import os
import tempfile
from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re
import numpy as np
import streamlit as st

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


# ---------- Semantic chunking ----------

_SENTENCE_BOUNDARY = re.compile(
    r'(?<=[.!?])\s+(?=[A-Z0-9(\[•])'           # end of a sentence
    r'|\n\s*\n'                                # blank line
    r'|\n(?=\s*(?:[•●▪◦\-\*]|\d+[.)])\s)'      # bullet / numbered line
)


def _split_sentences(text):
    sentences = []
    for part in _SENTENCE_BOUNDARY.split(text):
        part = re.sub(r"\s+", " ", part).strip()
        if part:
            sentences.append(part)
    return sentences


def _semantic_chunks(sentences, model, breakpoint_percentile,
                     buffer_size, min_chars, max_chars):
    """Group consecutive sentences, splitting where the topic shifts."""
    if not sentences:
        return []
    if len(sentences) < 3:
        return [" ".join(sentences)]

    # Embed each sentence with its neighbours for more stable similarity
    windows = [
        " ".join(sentences[max(0, i - buffer_size): i + buffer_size + 1])
        for i in range(len(sentences))
    ]
    embeddings = model.encode(
        windows, normalize_embeddings=True,
        show_progress_bar=False, batch_size=64
    )

    # Cosine distance between each sentence and the next one
    distances = 1 - np.sum(embeddings[:-1] * embeddings[1:], axis=1)
    threshold = np.percentile(distances, breakpoint_percentile)
    breakpoints = {i for i, d in enumerate(distances) if d > threshold}

    chunks, current, size = [], [], 0
    for i, sentence in enumerate(sentences):
        # hard cap on chunk size
        if current and size + len(sentence) > max_chars:
            chunks.append(" ".join(current))
            current, size = [], 0

        current.append(sentence)
        size += len(sentence) + 1

        # topic shift -> close the chunk (unless it's still too small)
        if i in breakpoints and size >= min_chars:
            chunks.append(" ".join(current))
            current, size = [], 0

    if current:
        tail = " ".join(current)
        if chunks and len(tail) < min_chars // 2 and len(chunks[-1]) + len(tail) <= max_chars:
            chunks[-1] += " " + tail      # merge a tiny leftover into the previous chunk
        else:
            chunks.append(tail)
    return chunks


def split_documents(documents, embedding_model,
                    breakpoint_percentile=85, buffer_size=1,
                    min_chunk_chars=300, max_chunk_chars=1500):
    """Split documents into semantically coherent chunks.

    embedding_model: a loaded SentenceTransformer (e.g. EmbeddingManager().model)
    """
    # Safety net for a single "sentence" that is longer than max_chunk_chars
    fallback = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_chars, chunk_overlap=0
    )

    split_docs = []
    for doc in documents:
        sentences = _split_sentences(doc.page_content)
        chunks = _semantic_chunks(
            sentences, embedding_model, breakpoint_percentile,
            buffer_size, min_chunk_chars, max_chunk_chars
        )
        for chunk in chunks:
            pieces = fallback.split_text(chunk) if len(chunk) > max_chunk_chars else [chunk]
            for piece in pieces:
                metadata = dict(doc.metadata)
                metadata["chunk_method"] = "semantic"
                split_docs.append(Document(page_content=piece, metadata=metadata))

    print(f"Split {len(documents)} pages into {len(split_docs)} semantic chunks")
    if split_docs:
        print(f"\nExample chunk:\ncontent: {split_docs[0].page_content[:500]}...")
        print(f"metadata: {split_docs[0].metadata}")
    return split_docs

