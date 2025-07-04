import os
import logging
from typing import List, Dict, Any, Optional
from pinecone import Pinecone, ServerlessSpec
import time

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, api_key: str, environment: str, index_name: str, dimension: int = 384):
        """
        Initialize Pinecone vector store.
        
        Args:
            api_key: Pinecone API key
            environment: Pinecone environment (e.g., "us-east-1-aws", "gcp-starter")
            index_name: Name of the Pinecone index
            dimension: Dimension of embeddings (default: 384 for all-MiniLM-L6-v2)
        """
        self.api_key = api_key
        self.environment = environment
        self.index_name = index_name
        self.dimension = dimension
        
        try:
            self.pc = Pinecone(api_key=api_key)
            
            existing_indexes = self.pc.list_indexes().names()
            
            if index_name in existing_indexes:
                logger.info(f"Found existing index: {index_name}")
                
                index_info = self.pc.describe_index(index_name)
                existing_dimension = index_info.dimension
                
                if existing_dimension != dimension:
                    logger.warning(f"Index dimension mismatch. Expected: {dimension}, Found: {existing_dimension}")
                    logger.warning("Deleting and recreating index with correct dimensions...")
                    self.pc.delete_index(index_name)
                    time.sleep(5)
                    self._create_index()
            else:
                self._create_index()
            
            self.index = self.pc.Index(index_name)
            logger.info(f"Connected to Pinecone index: {index_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {str(e)}")
            raise
    
    def _create_index(self):
        """Create a new Pinecone index."""
        try:
            env_parts = self.environment.lower().split('-')
            
            if 'gcp' in self.environment.lower():
                cloud = 'gcp'
                region = self.environment.replace('-gcp', '').replace('gcp-', '')
            elif 'aws' in self.environment.lower():
                cloud = 'aws'
                region = self.environment.replace('-aws', '').replace('aws-', '')
            elif 'azure' in self.environment.lower():
                cloud = 'azure'
                region = self.environment.replace('-azure', '').replace('azure-', '')
            else:
                if 'starter' in self.environment.lower():
                    cloud = 'gcp'
                    region = 'us-central1'
                else:
                    cloud = 'aws'
                    region = self.environment
            
            logger.info(f"Creating index with cloud={cloud}, region={region}")
            
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud=cloud,
                    region=region
                )
            )
            time.sleep(5)
            logger.info(f"Created new Pinecone index: {self.index_name}")
            
        except Exception as e:
            logger.error(f"Error creating index: {str(e)}")
            raise
    
    def add_documents(self, documents: List[Dict[str, Any]], namespace: str, batch_size: int = 100):
        """
        Add documents to vector store.
        
        Args:
            documents: List of document dictionaries with embeddings and metadata
            namespace: Namespace to store vectors in
            batch_size: Number of vectors to upsert at once
        """
        try:
            vectors = []
            
            for i, doc in enumerate(documents):
                if 'embedding' not in doc:
                    logger.warning(f"Document {i} missing embedding, skipping")
                    continue
                
                vector_id = f"{namespace}_{i}_{hash(doc.get('text', ''))}"
                
                metadata = {k: v for k, v in doc.items() if k != 'embedding'}
                
                for key, value in metadata.items():
                    if isinstance(value, (list, dict)):
                        metadata[key] = str(value)
                
                vector = {
                    "id": vector_id,
                    "values": doc["embedding"],
                    "metadata": metadata
                }
                vectors.append(vector)
            
            if not vectors:
                logger.warning("No valid vectors to add")
                return False
            
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(vectors=batch, namespace=namespace)
                logger.info(f"Upserted batch {i//batch_size + 1}/{(len(vectors)-1)//batch_size + 1}")
            
            logger.info(f"Successfully added {len(vectors)} vectors to namespace '{namespace}'")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents to Pinecone: {str(e)}")
            return False
    
    def similarity_search(self, query_embedding: List[float], namespace: str, 
                         top_k: int = 5, filter: Optional[Dict] = None):
        """
        Search for similar texts in a specific namespace with metadata filtering.
        
        Args:
            query_embedding: Query embedding vector
            namespace: Namespace to search in
            top_k: Number of results to return
            filter: Optional metadata filter dictionary
            
        Returns:
            List of matching documents with their metadata and scores
        """
        try:
            query_params = {
                "namespace": namespace,
                "vector": query_embedding,
                "top_k": top_k,
                "include_metadata": True
            }
            
            if filter:
                query_params["filter"] = filter
                logger.info(f"Searching with filter: {filter}")
            
            results = self.index.query(**query_params)
            
            matches = []
            for match in results.matches:
                if match.metadata:
                    item = dict(match.metadata)
                    item['score'] = match.score
                    item['id'] = match.id
                    matches.append(item)
            
            logger.info(f"Found {len(matches)} matches in namespace '{namespace}' with filter {filter}")
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
    
    def get_namespace_stats(self, namespace: str = None):
        """
        Get statistics for a specific namespace or all namespaces.
        
        Args:
            namespace: Specific namespace to get stats for (None for all)
            
        Returns:
            Dictionary with namespace statistics
        """
        try:
            stats = self.index.describe_index_stats()
            
            if namespace:
                if namespace in stats.namespaces:
                    return {
                        "namespace": namespace,
                        "vector_count": stats.namespaces[namespace].vector_count
                    }
                else:
                    return {"namespace": namespace, "vector_count": 0}
            else:
                return {
                    "total_vector_count": stats.total_vector_count,
                    "dimension": stats.dimension,
                    "namespaces": {
                        ns: {"vector_count": info.vector_count}
                        for ns, info in stats.namespaces.items()
                    }
                }
                
        except Exception as e:
            logger.error(f"Error getting namespace stats: {str(e)}")
            return {}
    
    def delete_namespace(self, namespace: str):
        """
        Delete all vectors in a namespace.
        
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
    
    def update_metadata(self, vector_id: str, metadata: Dict[str, Any], namespace: str):
        """
        Update metadata for a specific vector.
        
        Args:
            vector_id: ID of the vector to update
            metadata: New metadata
            namespace: Namespace containing the vector
        """
        try:
            fetch_result = self.index.fetch(ids=[vector_id], namespace=namespace)
            
            if vector_id not in fetch_result.vectors:
                logger.warning(f"Vector {vector_id} not found in namespace {namespace}")
                return False
            
            existing_vector = fetch_result.vectors[vector_id]
            
            self.index.upsert(
                vectors=[{
                    "id": vector_id,
                    "values": existing_vector.values,
                    "metadata": metadata
                }],
                namespace=namespace
            )
            
            logger.info(f"Updated metadata for vector {vector_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating metadata: {str(e)}")
            return False