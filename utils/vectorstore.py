import os
import logging
from typing import List, Dict, Any, Optional
from pinecone import Pinecone, ServerlessSpec

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, api_key: str, environment: str, index_name: str):
        """
        Initialize Pinecone vector store.
        
        Args:
            api_key: Pinecone API key
            environment: Pinecone environment
            index_name: Name of the Pinecone index
        """
        self.api_key = api_key
        self.environment = environment
        self.index_name = index_name
        
        self.pc = Pinecone(api_key=api_key)
        
        index_exists = index_name in self.pc.list_indexes().names()
        
        region_parts = environment.split('-')
        cloud = "aws"  # Default
        if len(region_parts) >= 3:
            if "gcp" in region_parts:
                cloud = "gcp"
            elif "aws" in region_parts:
                cloud = "aws"
            elif "azure" in region_parts:
                cloud = "azure"
                        
        region = environment.replace("-gcp", "").replace("-aws", "").replace("-azure", "")
        
        if index_exists:
            try:
                # Try a test query to see if dimensions match
                test_query = [0.0] * 384  # Create a test vector with 384 dimensions
                self.pc.Index(index_name).query(vector=test_query, top_k=1, include_metadata=True)
            except Exception as e:
                # If error mentions dimension mismatch, delete and recreate index
                if "dimension" in str(e).lower():
                    logger.warning(f"Dimension mismatch in index {index_name}, recreating with correct dimensions")
                    self.pc.delete_index(index_name)
                    index_exists = False
        
        if not index_exists:
            self.pc.create_index(
                name=index_name,
                dimension=384,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud=cloud,
                    region=region
                )
            )
            logger.info(f"Created new Pinecone index: {index_name}")
        
        self.index = self.pc.Index(index_name)
        logger.info(f"Connected to Pinecone index: {index_name}")
       
    def add_documents(self, documents: List[Dict[str, Any]], namespace: str):
        """
        Add documents to vector store.
        
        Args:
            documents: List of document dictionaries with embeddings and metadata
            namespace: Namespace to store vectors in (e.g., chapter name or number)
        """
        try:
            vectors = []
            for i, doc in enumerate(documents):
                vector = {
                    "id": f"{namespace}-{i}",
                    "values": doc["embedding"],
                    "metadata": {k: v for k, v in doc.items() if k != "embedding"}
                }
                vectors.append(vector)
            
            # Upsert vectors in batches
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i+batch_size]
                self.index.upsert(vectors=batch, namespace=namespace)
            
            logger.info(f"Added {len(vectors)} vectors to namespace '{namespace}'")
            return True
        except Exception as e:
            logger.error(f"Error adding documents to Pinecone: {str(e)}")
            return False
    
    def similarity_search(self, query_embedding: List[float], namespace: str, top_k: int = 5):
        """
        Search for similar texts in a specific namespace.
        
        Args:
            query_embedding: Query embedding vector
            namespace: Namespace to search in (e.g., chapter name or number)
            top_k: Number of results to return
            
        Returns:
            List of matching documents with their metadata
        """
        try:
            results = self.index.query(
                namespace=namespace,
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            # Format results
            matches = []
            for match in results.matches:
                item = match.metadata
                item['score'] = match.score
                matches.append(item)
            
            return matches
        except Exception as e:
            logger.error(f"Error searching Pinecone: {str(e)}")
            return []
    
    def list_namespaces(self):
        """
        List all namespaces in the index.
        
        Returns:
            List of namespace names
        """
        try:
            stats = self.index.describe_index_stats()
            namespaces = list(stats.namespaces.keys())
            logger.info(f"Found {len(namespaces)} namespaces: {namespaces}")
            return namespaces
        except Exception as e:
            logger.error(f"Error listing namespaces: {str(e)}")
            return []
    
    def delete_namespace(self, namespace: str):
        """
        Delete a namespace and all its vectors.
        
        Args:
            namespace: Namespace to delete
        """
        try:
            self.index.delete(delete_all=True, namespace=namespace)
            logger.info(f"Deleted namespace '{namespace}'")
            return True
        except Exception as e:
            logger.error(f"Error deleting namespace: {str(e)}")
            return False