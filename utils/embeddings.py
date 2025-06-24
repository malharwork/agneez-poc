# utils/embeddings.py
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class EmbeddingModel:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the embedding model.
        
        Args:
            model_name: Name of the sentence-transformers model to use
        """
        self.model = SentenceTransformer(model_name)
        logger.info(f"Initialized embedding model: {model_name}")
    
    def embed_texts(self, texts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Embed a list of text documents.
        
        Args:
            texts: List of dictionaries containing text and metadata
            
        Returns:
            List of dictionaries with embeddings added
        """
        try:
            text_strings = [doc['text'] for doc in texts]
            embeddings = self.model.encode(text_strings)
            
            # Add embeddings to documents
            for i, doc in enumerate(texts):
                doc['embedding'] = embeddings[i].tolist()
            
            logger.info(f"Embedded {len(texts)} texts")
            return texts
        except Exception as e:
            logger.error(f"Error embedding texts: {str(e)}")
            return texts
    
    def embed_query(self, query: str) -> List[float]:
        """
        Embed a query string.
        
        Args:
            query: Query text
            
        Returns:
            Query embedding as list of floats
        """
        try:
            embedding = self.model.encode(query)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error embedding query: {str(e)}")
            return []