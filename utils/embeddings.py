from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import logging
import numpy as np

logger = logging.getLogger(__name__)

class EmbeddingModel:
    def __init__(self, model_name: str = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'):
        """
        Initialize the embedding model.
        
        Args:
            model_name: Name of the sentence-transformers model to use
                       Default: 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2' (768 dimensions)
        """
        try:
            self.model = SentenceTransformer(model_name)
            self.embedding_dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"Initialized embedding model: {model_name} with {self.embedding_dimension} dimensions")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {str(e)}")
            raise
    
    def embed_texts(self, texts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Embed a list of text documents.
        
        Args:
            texts: List of dictionaries containing 'text' key and metadata
            
        Returns:
            List of dictionaries with embeddings added
        """
        try:
            # Extract text strings
            text_strings = []
            valid_indices = []
            
            for i, doc in enumerate(texts):
                if 'text' in doc and doc['text']:
                    text_strings.append(doc['text'])
                    valid_indices.append(i)
                else:
                    logger.warning(f"Document at index {i} has no 'text' field or empty text")
            
            if not text_strings:
                logger.error("No valid texts to embed")
                return texts
            
            # Generate embeddings
            embeddings = self.model.encode(text_strings, show_progress_bar=False)
            
            # Add embeddings to documents
            embedding_idx = 0
            for i, doc in enumerate(texts):
                if i in valid_indices:
                    doc['embedding'] = embeddings[embedding_idx].tolist()
                    embedding_idx += 1
                else:
                    # For invalid documents, create a zero embedding
                    doc['embedding'] = [0.0] * self.embedding_dimension
            
            logger.info(f"Successfully embedded {len(text_strings)} texts")
            return texts
            
        except Exception as e:
            logger.error(f"Error embedding texts: {str(e)}")
            # Return texts without embeddings on error
            return texts
    
    def embed_query(self, query: str) -> List[float]:
        """
        Embed a single query string.
        
        Args:
            query: Query text
            
        Returns:
            Query embedding as list of floats
        """
        try:
            if not query or not query.strip():
                logger.warning("Empty query provided")
                return [0.0] * self.embedding_dimension
            
            embedding = self.model.encode(query, show_progress_bar=False)
            return embedding.tolist()
            
        except Exception as e:
            logger.error(f"Error embedding query: {str(e)}")
            # Return zero vector on error
            return [0.0] * self.embedding_dimension
    
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Embed multiple texts in batches (useful for large datasets).
        
        Args:
            texts: List of text strings
            batch_size: Number of texts to process at once
            
        Returns:
            List of embeddings
        """
        try:
            if not texts:
                return []
            
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = self.model.encode(batch, show_progress_bar=False)
                all_embeddings.extend(batch_embeddings.tolist())
            
            logger.info(f"Embedded {len(texts)} texts in batches of {batch_size}")
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Error in batch embedding: {str(e)}")
            return []
    
    def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity score
        """
        try:
            # Convert to numpy arrays
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Compute cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error computing similarity: {str(e)}")
            return 0.0