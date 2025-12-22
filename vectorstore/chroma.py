import chromadb
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List, Dict, Any
import os
import logging

logger = logging.getLogger(__name__)

class VectorStoreManager:
    def __init__(self, persist_directory: str = "data/chroma", collection_name: str = "documind_collection"):
        """
        Initializes the Vector Store with ChromaDB and HuggingFace Embeddings.
        """
        # Ensure directory exists
        os.makedirs(persist_directory, exist_ok=True)
        
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        logger.info(f"Initializing VectorStore in {persist_directory}")
        
        # Initialize Embeddings (running locally via sentence-transformers)
        # using 'all-MiniLM-L6-v2' as per PRD
        self.embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Initialize ChromaDB
        self.vectordb = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embedding_function,
            collection_name=self.collection_name
        )
        
    def add_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Adds document chunks to the vector store.
        """
        if not chunks:
            return

        texts = [c['text'] for c in chunks]
        
        # Prepare metadata - flatten if needed or ensure types are supported
        # Chroma supports int, float, str, bool
        metadatas = []
        ids = []
        
        for c in chunks:
            ids.append(c['id'])
            # Filter metadata to simple types
            meta = {
                "doc_id": c.get("doc_id", ""),
                "page_number": c.get("page_number", 0),
                "chunk_index": c.get("chunk_index", 0),
                **{k: v for k, v in c.get("metadata", {}).items() if isinstance(v, (str, int, float, bool))}
            }
            metadatas.append(meta)
            
        try:
            self.vectordb.add_texts(texts=texts, metadatas=metadatas, ids=ids)
            logger.info(f"Added {len(chunks)} chunks to vector store.")
        except Exception as e:
            logger.error(f"Error adding chunks to vector store: {e}")
            raise e

    def similarity_search(self, query: str, k: int = 5) -> List[Any]:
        """
        Performs semantic search.
        """
        return self.vectordb.similarity_search(query, k=k)

    def retriever(self):
        """
        Returns a retriever object for LangChain chains.
        """
        return self.vectordb.as_retriever()
