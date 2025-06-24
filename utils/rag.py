import os
import anthropic
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class RAG:
    def __init__(self, anthropic_api_key: str, vector_store, embedding_model):
        """
        Initialize the RAG system.
        
        Args:
            anthropic_api_key: API key for Anthropic Claude
            vector_store: VectorStore instance
            embedding_model: EmbeddingModel instance
        """
        self.anthropic_api_key = anthropic_api_key
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.client = anthropic.Anthropic(
            api_key=anthropic_api_key
        )
        
        logger.info(f"Initialized RAG with model: claude-3-7-sonnet-20250219")
    
    def process_pdf_to_pinecone(self, pdf_path: str, pdf_loader):
        """
        Process a PDF file and add its contents to Pinecone with improved error handling.
        Each chapter is stored in a separate namespace.
        
        Args:
            pdf_path: Path to the PDF file
            pdf_loader: Module with PDF loading functions
        """
        try:
            chapters = pdf_loader.get_chapters_from_contents(pdf_path)
            processed_chapters = 0
            failed_chapters = []

            for chapter in chapters:
                chapter_number = chapter.get('chapter_number')
                chapter_title = chapter.get('title', 'Untitled')
                namespace = f"chapter_{chapter_number}"
                
                try:
                    chunks = pdf_loader.create_chapter_chunks(pdf_path, chapter)
                    
                    if not chunks or len(chunks) == 0:
                        logger.warning(f"Standard chunking failed for Chapter {chapter_number}: {chapter_title}. Trying alternative extraction...")
                        
                        #Try extracting with page range if available
                        if 'page_range' in chapter and chapter['page_range']:
                            start_page, end_page = chapter['page_range'].split('-')
                            start_page, end_page = int(start_page), int(end_page)
                            
                            # Extract using page range
                            full_text = pdf_loader.extract_text_from_page_range(pdf_path, start_page, end_page)
                            
                            if full_text and len(full_text.strip()) > 50:  # Only proceed if we got meaningful text
                                # Create chunks manually
                                chunks = []
                                # Simple chunk size of ~500 chars with 50 char overlap
                                chunk_size = 500
                                overlap = 50
                                
                                for i in range(0, len(full_text), chunk_size - overlap):
                                    chunk_text = full_text[i:i + chunk_size]
                                    if len(chunk_text.strip()) > 100:  # Only add meaningful chunks
                                        chunks.append({
                                            'text': chunk_text,
                                            'chapter_number': chapter_number,
                                            'chapter_title': chapter_title,
                                            'page_range': chapter.get('page_range', 'unknown'),
                                            'unit': chapter.get('unit', 'unknown')
                                        })
                                
                                logger.info(f"Created {len(chunks)} chunks for Chapter {chapter_number} using alternative extraction")
                    
                    # If still no chunks, try OCR as last resort
                    if not chunks or len(chunks) == 0:
                        logger.warning(f"Alternative extraction failed for Chapter {chapter_number}. Consider using OCR for this chapter.")
                        failed_chapters.append(f"Chapter {chapter_number}: {chapter_title}")
                        continue
                    
                    # Embed chunks
                    embedded_chunks = self.embedding_model.embed_texts(chunks)
                    
                    # Check embedded chunks
                    if not embedded_chunks or 'embedding' not in embedded_chunks[0]:
                        logger.error(f"Failed to embed chunks for chapter {chapter_number}")
                        failed_chapters.append(f"Chapter {chapter_number}: {chapter_title}")
                        continue
                    
                    # Add chunks to Pinecone
                    self.vector_store.add_documents(embedded_chunks, namespace)
                    
                    logger.info(f"Processed chapter {chapter_number}: {chapter_title}")
                    processed_chapters += 1
                    
                except Exception as e:
                    logger.error(f"Error processing chapter {chapter_number}: {str(e)}")
                    failed_chapters.append(f"Chapter {chapter_number}: {chapter_title}")
            
            # Log summary
            logger.info(f"Successfully processed {processed_chapters} out of {len(chapters)} chapters")
            if failed_chapters:
                logger.warning(f"Failed to process {len(failed_chapters)} chapters: {', '.join(failed_chapters)}")
            
            return processed_chapters, failed_chapters
        
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            return 0, []
    
    def answer_question(self, question: str, namespace: str, top_k: int = 3):
        """
        Answer a question using RAG.
        
        Args:
            question: User's question
            namespace: Namespace to search in (e.g., chapter_1)
            top_k: Number of documents to retrieve
            
        Returns:
            Answer and source information
        """
        try:
            query_embedding = self.embedding_model.embed_query(question)
            
            results = self.vector_store.similarity_search(query_embedding, namespace, top_k)
            
            if not results:
                return {
                    "answer": "I couldn't find relevant information to answer your question.",
                    "source": "No sources found"
                }
            
            context = "\n\n".join([result.get('text', '') for result in results])
            
            chapter_number = results[0].get('chapter_number')
            chapter_title = results[0].get('chapter_title')
            unit = results[0].get('unit')
            page_range = results[0].get('page_range')
            
            source = f"Chapter {chapter_number}: {chapter_title} ({unit}, Pages {page_range})"
            
            response = self._call_claude(question, context)
            
            return {
                "answer": response,
                "source": source
            }
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            return {
                "answer": "I encountered an error while generating a response. Please try again.",
                "source": "Error"
            }
    
    def answer_from_page_range(self, question: str, start_page: int, end_page: int, pdf_path: str, pdf_loader):
        """
        Answer a question based on a specific page range.
        
        Args:
            question: User's question
            start_page: Start page number
            end_page: End page number
            pdf_path: Path to the PDF file
            pdf_loader: Module with PDF loading functions
            
        Returns:
            Answer and source information
        """
        try:
            # Extract text from page range
            text = pdf_loader.extract_text_from_page_range(pdf_path, start_page, end_page)
            
            # Generate response using Claude
            response = self._call_claude(question, text)
            
            return {
                "answer": response,
                "source": f"Pages {start_page}-{end_page}"
            }
        except Exception as e:
            logger.error(f"Error answering from page range: {str(e)}")
            return {
                "answer": "I encountered an error while generating a response. Please try again.",
                "source": f"Pages {start_page}-{end_page}"
            }
    
    def _call_claude(self, question: str, context: str) -> str:
        """
        Call the Claude API to generate a response.
        
        Args:
            question: User's question
            context: Context from relevant chunks
            
        Returns:
            Claude's response
        """
        try:
            system_prompt = """You are a helpful learning assistant for a textbook. 
            Answer questions based ONLY on the provided context. 
            If the context doesn't contain relevant information, say so clearly.
            Do not make up information or use external knowledge.
            Be clear, concise, and helpful in your explanations.
            Use examples from the context if applicable.
            Format your answers in an easy-to-understand way.
            """
            
            messages = [
                {
                    "role": "user",
                    "content": f"""Context:
                    {context}
                    
                    Question: {question}
                    
                    Please provide an answer based only on the information in the context.
                    """
                }
            ]
            
            response = self.client.messages.create(
                model="claude-3-7-sonnet-20250219",  
                system=system_prompt,
                messages=messages,
                max_tokens=500,
                temperature=0.3
            )
            
            return response.content[0].text
        
        except Exception as e:
            logger.error(f"Error calling Claude: {str(e)}")
            return f"I'm sorry, I encountered an error while generating a response. Please try again."
    
    # Additional methods from RAGSystem class for backward compatibility
    
    def retrieve_relevant_chunks(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: User query
            top_k: Number of chunks to retrieve
            
        Returns:
            List of relevant chunks
        """
        query_embedding = self.embedding_model.embed_query(query)
        results = self.vector_store.similarity_search(query_embedding, top_k=top_k)
        
        return results
    
    def format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Format retrieved chunks as context for the LLM.
        
        Args:
            chunks: List of relevant chunks
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for chunk in chunks:
            # Add the text without labeling it as a separate document
            context_parts.append(chunk['text'])
        
        # Join all context parts with a space
        return "\n\n".join(context_parts)
    
    def generate_response(self, query: str, chat_history: List[Dict[str, str]] = None, 
                          context_identifier: str = None) -> Dict[str, Any]:
        """
        Generate a response using RAG.
        
        Args:
            query: User query
            chat_history: List of previous messages
            context_identifier: String identifier for the current context (chapter or page range)
            
        Returns:
            Response from Claude with sources
        """
        # Retrieve relevant chunks
        relevant_chunks = self.retrieve_relevant_chunks(query)
        
        if not relevant_chunks:
            return {
                "answer": f"I couldn't find any relevant information to answer your question in {context_identifier}. Please try a different question or select a different section of the document.",
                "sources": []
            }
        
        # Format context
        context = self.format_context(relevant_chunks)
        
        # Prepare system prompt with context and instructions
        system_prompt = f"""You are a helpful assistant that accurately answers questions based on the provided document content.
        
DOCUMENT CONTENT:
{context}

INSTRUCTIONS:
1. Answer the user's question based ONLY on the information in the DOCUMENT CONTENT provided above.
2. If the answer is not in the document or if the information is insufficient to provide a complete answer, respond with:
   "I don't have enough information in {context_identifier} to answer this question. This topic might be covered in other parts of the document. Would you like to try a different section?"
3. If the user asks a question that is completely unrelated to the document content, respond with:
   "Your question appears to be outside the scope of the content in {context_identifier}. Would you like to try a different question related to this section or select a different part of the document?"
4. Keep answers concise but complete.
5. Do not reference "Document 1" or "Document 2" etc. in your answers. Treat all the information as coming from a single coherent source.
6. Do not include phrases like "According to the document" or "The document states" in your answers. Simply provide the information directly.
7. Do not make up information or use knowledge outside of the provided document content.
8. IMPORTANT: Only answer based on the DOCUMENT CONTENT. Never speculate, guess, or draw from general knowledge not contained within the DOCUMENT CONTENT.
"""

        # Initialize messages with system prompt
        messages = [
            {"role": "assistant", "content": "How can I help you with your questions about the document?"}
        ]
        
        # Add chat history if provided
        if chat_history:
            for message in chat_history:
                messages.append(message)
        
        # Add user query
        messages.append({"role": "user", "content": query})
        
        try:
            # Call Claude API with error handling
            response = self.client.messages.create(
                model="claude-3-7-sonnet-20250219",
                system=system_prompt,
                messages=messages,
                max_tokens=1000,
                temperature=0.0
            )
            
            # Prepare sources information
            sources = []
            for chunk in relevant_chunks:
                source_info = f"{chunk.get('chapter_title', 'Section')} (Pages {chunk.get('page_range', 'unknown')})"
                
                if source_info not in sources:
                    sources.append(source_info)
            
            # Return response with sources
            return {
                "answer": response.content[0].text,
                "source": source_info  # For backward compatibility with old method
            }
        
        except Exception as e:
            # Log the error and return a helpful message
            logger.error(f"Error from Claude API: {str(e)}")
            
            return {
                "answer": f"I encountered an error while generating a response. Error details: {str(e)}",
                "source": "Error"  # For backward compatibility
            }