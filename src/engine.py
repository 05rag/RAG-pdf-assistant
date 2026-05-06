import os
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from typing import List ,Dict, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import uuid

class EmbeddingManager:

    """Handles document embedding generation using Sentence Transformer"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):

        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the sentence transformer model"""
        try:
            print(f"Loading embedding model: {self.model_name}")

            self.model = SentenceTransformer(self.model_name)

            print(f"Model loaded successfully. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")

        except Exception as e:
            print(f"Error loading model: {self.model_name}:{e}")
            raise

    
    def generate_embeddings(self, texts:List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts
        
        Args:
        texts: List of strings to embed
        
        Returns:
        Numpy array of shape (num_texts, embedding_dim)"""

        if not self.model:
            raise ValueError("Model not loaded")
        
        print(f"Generating embeddings for {len(texts)} texts...")

        embeddings = self.model.encode(texts, show_progress_bar = True)
        
        print(f"Generate embeddings with shape: {embeddings.shape}")
        return embeddings



#Vector Store 
class VectorStore:
    """Manages document embedding in a ChromaDB vector store"""

    def __init__(self, collection_name: str = "pdf_documents", persist_directory: str = "../data/vector_store"):
        """ Initialize the vector store

        Args:
        Collection_name : Name of the ChromaDB collection
        persist_directory: DIrectory to persist the vector store
        """

        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_store() #to initialize vector store

    def _initialize_store(self):
        """Initialize ChromaDB client and collection"""

        try:
            # Create persistent ChromaDB client
            os.makedirs(self.persist_directory, exist_ok=True)
            #persist_directory = creates client with refrence to the chromadb vector store
            self.client= chromadb.PersistentClient(path= self.persist_directory)

            #Get or create collection
            #collection = place in the vector store where we store our vector

            self.collection = self.client.get_or_create_collection(
                name = self.collection_name,
                metadata= {"hnsw:space": "cosine","description": "PDF document embeddings for RAG"}
            ) 

            print(f"Vector store initialized using cosine similarity. Collection: {self.collection_name}")
            print(f"Existing documents in collection: {self.collection.count()}")

        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise
    

    def add_documents(self, documents: List[Any], embeddings:np.ndarray) :
        """ Add documents and their embedding to the vector store"""
    
        
        if len(documents) != len(embeddings):
            raise ValueError("No of embedding should match the document")
        
        print(f"Adding {len(documents)} documents to the vector store...")

        #Preparing data for ChromaDB
        ids =[]
        metadatas = []
        documents_text =[]
        embeddings_list = []

        for i, (doc, embedding) in enumerate (zip(documents,embeddings)): #creating a tuple

            #generate unique id
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)

            #Prepare metadata
            metadata = dict(doc.metadata)
            metadata['doc_index'] = i
            metadata['content_length'] = len(doc.page_content)
            metadatas.append(metadata)

            #document content
            documents_text.append(doc.page_content)

            #embedding
            embeddings_list.append(embedding.tolist())

        #Add to collection
        try:
            self.collection.add(
                ids = ids,
                metadatas = metadatas,
                documents = documents_text,
                embeddings = embeddings_list
            )

            print(f"Successfully added {len(documents)} documents to vector store")
            print(f"Total documents in collection : {self.collection.count()}")

        except Exception as e:
            print(f"Error adding document to vector store: {e}")
            raise



class RAGRetriever():
    """Handels query based retrieval from the vector store"""
    
    def __init__(self, vector_store: VectorStore, embedding_manager : EmbeddingManager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query: str, top_k: int =8, score_threshold : float = 0.3, source_file : str=None ) -> List[Dict[str, Any]]:
        
        print(f"Retrieving documents for the query '{query}'")
        print(f"Top K: {top_k}, store threshold: {score_threshold}")

        #Generating query embedding
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]

        search_filter = {"source" : source_file} if source_file else None

        #Search in vector store
        try:
            results = self.vector_store.collection.query(
                query_embeddings = [query_embedding.tolist()],
                n_results = top_k ,
                where= search_filter
            )
            
            print("RAW RESULTS:", results)

            #process results
            retrieved_docs = []

            if results['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                ids = results['ids'][0]

                for i, (doc_id, document, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances)):
                    #Convert distance to similarity score (ChromaDB uses cosine distance)
                    similarity_score = 1 - distance

                    if similarity_score >= score_threshold:
                        retrieved_docs.append({
                            'id' : doc_id,
                            'content': document,
                            'metadata' :metadata,
                            'similarity_score': similarity_score,
                            'distance' : distance,
                            'rank': i+1
                        })

                print(f"Retrieved {len(retrieved_docs)} documents (after filtering)")

            else:
                print("No documents found")

            return retrieved_docs

        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []

                

   