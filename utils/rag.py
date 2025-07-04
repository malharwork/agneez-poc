import anthropic
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class EducationalRAG:
    def __init__(self, anthropic_api_key: str, vector_stores: Dict[str, Any], embedding_model):
        """Initialize the Educational RAG system with multiple vector stores."""
        self.anthropic_api_key = anthropic_api_key
        self.vector_stores = vector_stores
        self.embedding_model = embedding_model
        self.client = anthropic.Anthropic(api_key=anthropic_api_key)
        
        # Topic to index mapping
        self.topic_index_map = {
            'quadratic_equations': ('math_index', 'algebra_quadratic_equations'),
            'digestive_system': ('science_index', 'biology_digestive_system')
        }
    
    def answer_educational_question(self, question: str, topic: str, metadata_filter: Dict, level: str = None):
        """Answer a question with comprehensive metadata filtering."""
        try:
            # Get the appropriate index and namespace
            if topic not in self.topic_index_map:
                return {
                    "answer": f"Topic {topic} not found in the system.",
                    "error": "Invalid topic"
                }
            
            index_name, namespace = self.topic_index_map[topic]
            vector_store = self.vector_stores[index_name]
            
            # Embed the question
            query_embedding = self.embedding_model.embed_query(question)
            
            # Search with metadata filtering
            results = vector_store.similarity_search(
                query_embedding,
                namespace=namespace,
                top_k=5,
                filter=metadata_filter
            )
            
            if not results:
                # Try with relaxed filters
                relaxed_filter = {
                    'board': metadata_filter.get('board'),
                    'grade': metadata_filter.get('grade')
                }
                results = vector_store.similarity_search(
                    query_embedding,
                    namespace=namespace,
                    top_k=5,
                    filter=relaxed_filter
                )
            
            if not results:
                return {
                    "answer": f"I couldn't find relevant content for your query with the specified filters.",
                    "metadata_filter": metadata_filter,
                    "suggestions": self._get_alternative_suggestions(topic, metadata_filter)
                }
            
            # Build context from results
            context_parts = []
            content_metadata = []
            
            for result in results:
                context_parts.append(result.get('text', ''))
                content_metadata.append({
                    'content_id': result.get('content_id'),
                    'subtopic': result.get('subtopic'),
                    'sub_method': result.get('sub_method'),
                    'method_tags': result.get('method_tags', []),
                    'difficulty_level': result.get('difficulty_level'),
                    'content_type': result.get('content_type')
                })
            
            context = "\n\n".join(context_parts)
            
            # Generate response
            response = self._generate_educational_response(
                question, context, metadata_filter, content_metadata
            )
            
            return {
                "answer": response,
                "content_metadata": content_metadata,
                "filter_applied": metadata_filter,
                "topic": topic,
                "results_count": len(results)
            }
            
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            return {
                "answer": "I encountered an error while generating a response. Please try again.",
                "error": str(e)
            }
    
    def get_adaptive_content(self, topic: str, metadata_filter: Dict):
        """Get content adapted to student's current level."""
        try:
            index_name, namespace = self.topic_index_map[topic]
            vector_store = self.vector_stores[index_name]
            
            # Create a general query embedding for the topic
            query = f"Practice problems for {topic} at difficulty level {metadata_filter.get('difficulty_level', 3)}"
            query_embedding = self.embedding_model.embed_query(query)
            
            # Search for appropriate content
            results = vector_store.similarity_search(
                query_embedding,
                namespace=namespace,
                top_k=10,
                filter=metadata_filter
            )
            
            # Sort by difficulty and adaptation weight
            sorted_results = sorted(
                results,
                key=lambda x: (x.get('difficulty_level', 3), -x.get('adaptation_weight', 1.0))
            )
            
            # Select diverse content types
            selected_content = []
            content_types_seen = set()
            
            for result in sorted_results:
                content_type = result.get('content_type')
                if content_type not in content_types_seen or len(selected_content) < 5:
                    selected_content.append(result)
                    content_types_seen.add(content_type)
            
            return {
                "adaptive_content": selected_content[:5],
                "difficulty_range": metadata_filter.get('difficulty_level'),
                "content_types": list(content_types_seen)
            }
            
        except Exception as e:
            logger.error(f"Error getting adaptive content: {str(e)}")
            return {"error": str(e)}
    
    def generate_learning_path(self, topic: str, grade: int, board: str, 
                             current_subtopic: str = None, mastery_level: float = 0.5):
        """Generate a personalized learning path based on current progress."""
        try:
            # Define learning progression for topics
            learning_progressions = {
                'quadratic_equations': {
                    'sequence': [
                        'patterns_introduction',
                        'factorization_method',
                        'completing_square',
                        'formula_method',
                        'applications'
                    ],
                    'prerequisites': {
                        'patterns_introduction': [],
                        'factorization_method': ['patterns_introduction', 'basic_algebra'],
                        'completing_square': ['factorization_method', 'algebraic_manipulation'],
                        'formula_method': ['completing_square', 'square_roots'],
                        'applications': ['formula_method', 'word_problems']
                    }
                },
                'digestive_system': {
                    'sequence': [
                        'anatomy_structure',
                        'digestion_process',
                        'enzymes_secretions',
                        'absorption_transport',
                        'disorders_health'
                    ],
                    'prerequisites': {
                        'anatomy_structure': [],
                        'digestion_process': ['anatomy_structure'],
                        'enzymes_secretions': ['digestion_process', 'basic_chemistry'],
                        'absorption_transport': ['anatomy_structure', 'cell_biology'],
                        'disorders_health': ['digestion_process', 'absorption_transport']
                    }
                }
            }
            
            progression = learning_progressions.get(topic, {})
            sequence = progression.get('sequence', [])
            prerequisites = progression.get('prerequisites', {})
            
            # Find current position
            current_index = 0
            if current_subtopic and current_subtopic in sequence:
                current_index = sequence.index(current_subtopic)
            
            # Determine next steps based on mastery
            learning_path = {
                'current_subtopic': current_subtopic or sequence[0],
                'current_grade': grade,
                'board': board,
                'mastery_level': mastery_level,
                'next_steps': []
            }
            
            if mastery_level < 0.7:
                # Need more practice at current level
                learning_path['recommendation'] = 'Continue practicing current topic'
                learning_path['next_steps'] = [
                    {
                        'subtopic': current_subtopic or sequence[0],
                        'focus': 'practice_problems',
                        'difficulty_adjustment': -0.5
                    }
                ]
            else:
                # Ready to progress
                if current_index < len(sequence) - 1:
                    next_subtopic = sequence[current_index + 1]
                    learning_path['recommendation'] = f'Progress to {next_subtopic}'
                    learning_path['next_steps'] = [
                        {
                            'subtopic': next_subtopic,
                            'focus': 'introduction',
                            'prerequisites_to_review': prerequisites.get(next_subtopic, [])
                        }
                    ]
                else:
                    learning_path['recommendation'] = 'Explore advanced applications'
                    learning_path['next_steps'] = [
                        {
                            'subtopic': 'applications',
                            'focus': 'advanced_problems',
                            'difficulty_adjustment': +0.5
                        }
                    ]
            
            # Add remediation if needed
            if mastery_level < 0.4:
                prereqs = prerequisites.get(current_subtopic, [])
                if prereqs:
                    learning_path['remediation'] = prereqs
            
            return learning_path
            
        except Exception as e:
            logger.error(f"Error generating learning path: {str(e)}")
            return {"error": str(e)}
    
    def update_performance_metrics(self, content_id: str, success: bool, 
                                 error_type: str = None, time_taken: int = 0):
        """Update performance metrics for a specific content."""
        try:
            # This would typically update the vector store metadata
            # For now, we'll log the update
            logger.info(f"Performance update for {content_id}: success={success}, error={error_type}, time={time_taken}")
            
            # In a production system, you would:
            # 1. Fetch the current vector
            # 2. Update its performance metadata
            # 3. Re-index with updated metadata
            
            return {"status": "logged"}
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {str(e)}")
            return {"error": str(e)}
    
    def search_by_prerequisites(self, topic: str, prerequisite_concepts: List[str], 
                               grade: int, board: str):
        """Find content that builds on specific prerequisites."""
        try:
            index_name, namespace = self.topic_index_map[topic]
            vector_store = self.vector_stores[index_name]
            
            # Create query based on prerequisites
            query = f"Content requiring {', '.join(prerequisite_concepts)}"
            query_embedding = self.embedding_model.embed_query(query)
            
            # Search with prerequisite filter
            metadata_filter = {
                'grade': grade,
                'board': board,
                'prerequisite_concepts': {"$in": prerequisite_concepts}
            }
            
            results = vector_store.similarity_search(
                query_embedding,
                namespace=namespace,
                top_k=10,
                filter=metadata_filter
            )
            
            return {
                "content_with_prerequisites": results,
                "prerequisites_searched": prerequisite_concepts
            }
            
        except Exception as e:
            logger.error(f"Error searching by prerequisites: {str(e)}")
            return {"error": str(e)}
    
    def get_method_specific_content(self, topic: str, method_tag: str, 
                                   grade: int, board: str, exclude_methods: List[str] = None):
        """Get content specific to a teaching/solving method."""
        try:
            index_name, namespace = self.topic_index_map[topic]
            vector_store = self.vector_stores[index_name]
            
            # Create method-specific query
            query = f"Learn {topic} using {method_tag} method"
            query_embedding = self.embedding_model.embed_query(query)
            
            # Build filter
            metadata_filter = {
                'grade': grade,
                'board': board,
                'method_tags': {"$in": [method_tag]}
            }
            
            if exclude_methods:
                metadata_filter['excluded_methods'] = {"$nin": exclude_methods}
            
            results = vector_store.similarity_search(
                query_embedding,
                namespace=namespace,
                top_k=10,
                filter=metadata_filter
            )
            
            # Group by subtopic
            grouped_results = {}
            for result in results:
                subtopic = result.get('subtopic', 'general')
                if subtopic not in grouped_results:
                    grouped_results[subtopic] = []
                grouped_results[subtopic].append(result)
            
            return {
                "method_specific_content": grouped_results,
                "method_tag": method_tag,
                "excluded_methods": exclude_methods or []
            }
            
        except Exception as e:
            logger.error(f"Error getting method-specific content: {str(e)}")
            return {"error": str(e)}
    
    def _generate_educational_response(self, question: str, context: str, 
                                     metadata_filter: Dict, content_metadata: List[Dict]):
        """Generate a response with awareness of content metadata."""
        
        # Extract key information from metadata
        grade = metadata_filter.get('grade', 9)
        board = metadata_filter.get('board', 'CBSE')
        language = metadata_filter.get('language', 'english')
        
        # Identify methods used in retrieved content
        all_methods = set()
        content_types = set()
        for meta in content_metadata:
            all_methods.update(meta.get('method_tags', []))
            content_types.add(meta.get('content_type', 'general'))
        
        # Board and grade-specific instructions
        instruction_map = {
            'CBSE': {
                'style': 'Follow NCERT pattern with clear explanations and step-by-step solutions.',
                'focus': 'Emphasize conceptual understanding and exam preparation.'
            },
            'ICSE': {
                'style': 'Provide comprehensive explanations with multiple approaches.',
                'focus': 'Include detailed reasoning and encourage analytical thinking.'
            },
            'SSC': {
                'style': 'Use simple, direct explanations with local context where applicable.',
                'focus': 'Focus on practical understanding and textbook methods.'
            }
        }
        
        board_instruction = instruction_map.get(board, instruction_map['CBSE'])
        
        # Grade-appropriate language
        if grade <= 5:
            grade_instruction = "Use simple language with examples a young child can understand."
        elif grade <= 8:
            grade_instruction = "Use clear explanations with appropriate technical terms defined."
        else:
            grade_instruction = "Use subject-appropriate terminology with detailed explanations."
        
        system_prompt = f"""You are an expert educator for grade {grade} {board} board students.
        
        {board_instruction['style']}
        {board_instruction['focus']}
        {grade_instruction}
        
        Available methods in the content: {', '.join(all_methods) if all_methods else 'general explanation'}
        Content types available: {', '.join(content_types)}
        
        IMPORTANT INSTRUCTIONS:
        1. Base your response ONLY on the provided context
        2. Use methods and approaches that match the student's grade and board
        3. If asked about methods not in the context, mention they'll learn it later
        4. Maintain consistency with the board's teaching methodology
        5. If content is in {language}, respond accordingly
        
        Context:
        {context}
        """
        
        messages = [
            {
                "role": "user",
                "content": f"Student Question: {question}\n\nPlease provide an answer appropriate for grade {grade} {board} board."
            }
        ]
        
        try:
            response = self.client.messages.create(
                model="claude-3-sonnet-20240229",
                system=system_prompt,
                messages=messages,
                max_tokens=1000,
                temperature=0.7
            )
            
            return response.content[0].text
            
        except Exception as e:
            logger.error(f"Error calling Claude: {str(e)}")
            return "I'm having trouble generating a response right now. Please try again."
    
    def _get_alternative_suggestions(self, topic: str, metadata_filter: Dict):
        """Get alternative suggestions based on current filters."""
        grade = metadata_filter.get('grade', 9)
        board = metadata_filter.get('board', 'CBSE')
        subtopic = metadata_filter.get('subtopic', '')
        
        suggestions = {
            'quadratic_equations': {
                'general': [
                    f"What are the methods to solve quadratic equations in grade {grade}?",
                    f"Show me {board} board examples of quadratic equations",
                    "How do I identify which method to use?"
                ],
                'factorization_method': [
                    "How do I factor x² + 5x + 6?",
                    "What is splitting the middle term?",
                    "When can I use simple factoring?"
                ],
                'formula_method': [
                    "What is the quadratic formula?",
                    "How do I use the discriminant?",
                    "Show me step-by-step formula application"
                ]
            },
            'digestive_system': {
                'general': [
                    f"What parts of digestive system do we study in grade {grade}?",
                    f"Explain digestion process for {board} board",
                    "What are the main digestive organs?"
                ],
                'anatomy_structure': [
                    "What are the organs in order?",
                    "Describe the structure of stomach",
                    "What is the function of each organ?"
                ],
                'digestion_process': [
                    "How does mechanical digestion work?",
                    "What is chemical digestion?",
                    "Explain the complete digestion process"
                ]
            }
        }
        
        topic_suggestions = suggestions.get(topic, {})
        return topic_suggestions.get(subtopic, topic_suggestions.get('general', []))