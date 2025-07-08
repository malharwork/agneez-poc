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
        
        self.topic_index_map = {
            'quadratic_equations': ('math_index', 'algebra_quadratic_equations'),
            'digestive_system': ('science_index', 'biology_digestive_system')
        }
        
        self.grade_topic_mapping = {
            'quadratic_equations': {
                'min_grade': 8,
                'optimal_grades': [9, 10, 11, 12],
                'typical_introduction': 'grades 8-9'
            },
            'digestive_system': {
                'min_grade': 3,
                'optimal_grades': [6, 7, 8, 9, 10],
                'typical_introduction': 'grades 3-4'
            }
        }
    
    def answer_educational_question(self, question: str, topic: str, metadata_filter: Dict, level: str = None):
        """Answer a question with comprehensive metadata filtering and grade appropriateness."""
        try:
            if topic not in self.topic_index_map:
                return {
                    "answer": f"Topic {topic} not found in the system.",
                    "error": "Invalid topic"
                }
            
            index_name, namespace = self.topic_index_map[topic]
            vector_store = self.vector_stores[index_name]
            
            query_embedding = self.embedding_model.embed_query(question)
            
            results = vector_store.similarity_search(
                query_embedding,
                namespace=namespace,
                top_k=5,
                filter=metadata_filter
            )
            
            if not results:
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
            
            query = f"Practice problems for {topic} at difficulty level {metadata_filter.get('difficulty_level', 3)}"
            query_embedding = self.embedding_model.embed_query(query)
            
            results = vector_store.similarity_search(
                query_embedding,
                namespace=namespace,
                top_k=10,
                filter=metadata_filter
            )
            
            sorted_results = sorted(
                results,
                key=lambda x: (x.get('difficulty_level', 3), -0.5)
            )
            
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
    
    def generate_learning_path(self, topic: str, grade: int, board: str, current_subtopic: str = None, mastery_level: float = 0.5):
        """Generate a personalized learning path based on current progress."""
        try:
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
            
            current_index = 0
            if current_subtopic and current_subtopic in sequence:
                current_index = sequence.index(current_subtopic)
            
            learning_path = {
                'current_subtopic': current_subtopic or sequence[0],
                'current_grade': grade,
                'board': board,
                'mastery_level': mastery_level,
                'next_steps': []
            }
            
            if mastery_level < 0.7:
                learning_path['recommendation'] = 'Continue practicing current topic'
                learning_path['next_steps'] = [
                    {
                        'subtopic': current_subtopic or sequence[0],
                        'focus': 'practice_problems',
                        'difficulty_adjustment': -0.5
                    }
                ]
            else:
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
            
            if mastery_level < 0.4:
                prereqs = prerequisites.get(current_subtopic, [])
                if prereqs:
                    learning_path['remediation'] = prereqs
            
            return learning_path
            
        except Exception as e:
            logger.error(f"Error generating learning path: {str(e)}")
            return {"error": str(e)}
    
    def _generate_educational_response(self, question: str, context: str, 
                                     metadata_filter: Dict, content_metadata: List[Dict]):
        """Generate a response with awareness of content metadata and grade appropriateness."""
        
        grade = metadata_filter.get('grade', 9)
        board = metadata_filter.get('board', 'CBSE')
        language = metadata_filter.get('language', 'english')
        
        all_methods = set()
        content_types = set()
        for meta in content_metadata:
            all_methods.update(meta.get('method_tags', []))
            content_types.add(meta.get('content_type', 'general'))
        
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
        
        if grade <= 5:
            grade_instruction = """Use very simple language appropriate for young children (ages 8-11). 
            - Use short sentences and familiar words
            - Include fun examples and analogies
            - Avoid complex mathematical terminology
            - Make it engaging and easy to understand
            - If a concept is too advanced, gently redirect to age-appropriate topics"""
        elif grade <= 8:
            grade_instruction = """Use clear explanations appropriate for middle school students (ages 11-14).
            - Use proper academic terminology but explain it clearly
            - Include step-by-step explanations
            - Provide relatable examples
            - Build concepts gradually"""
        else:
            grade_instruction = """Use subject-appropriate terminology with detailed explanations for high school students.
            - Include proper mathematical/scientific notation
            - Provide comprehensive explanations
            - Include advanced concepts where appropriate
            - Prepare for board exams"""
        
        system_prompt = f"""You are an expert educator for grade {grade} {board} board students.

        CRITICAL GRADE APPROPRIATENESS RULES:
        1. You are teaching a Grade {grade} student
        2. If the content seems too advanced for Grade {grade}, simplify it significantly or mention it will be covered in higher grades
        3. Never provide content that is clearly meant for much higher grades
        4. Always match the cognitive development level of a Grade {grade} student
        
        BOARD-SPECIFIC APPROACH:
        {board_instruction['style']}
        {board_instruction['focus']}
        
        GRADE-APPROPRIATE LANGUAGE:
        {grade_instruction}
        
        CONTENT CONTEXT:
        Available methods in the content: {', '.join(all_methods) if all_methods else 'general explanation'}
        Content types available: {', '.join(content_types)}
        
        IMPORTANT INSTRUCTIONS:
        1. Base your response ONLY on the provided context
        2. Use methods and approaches that match the student's grade {grade} and board {board}
        3. If asked about methods not appropriate for Grade {grade}, mention they'll learn it in higher grades
        4. Maintain consistency with the {board} board's teaching methodology for Grade {grade}
        5. If content is in {language}, respond accordingly
        6. NEVER provide advanced formulas or concepts inappropriate for Grade {grade}
        7. If the question is about advanced topics, acknowledge their curiosity but redirect to grade-appropriate content
        
        Context (filtered for Grade {grade}):
        {context}
        """
        
        messages = [
            {
                "role": "user",
                "content": f"Student Question: {question}\n\nPlease provide an answer appropriate for Grade {grade} {board} board. Remember, this student is in Grade {grade}, so keep your explanation suitable for their level."
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
        """Get alternative suggestions based on current filters and grade level."""
        grade = metadata_filter.get('grade', 9)
        board = metadata_filter.get('board', 'CBSE')
        subtopic = metadata_filter.get('subtopic', '')
        
        suggestions = {
            'quadratic_equations': {
                'general': {
                    'elementary': [
                        f"What are square numbers? (better for Grade {grade})",
                        f"How to recognize patterns in numbers?",
                        "What is multiplication?"
                    ],
                    'middle_school': [
                        f"What are quadratic expressions for Grade {grade}?",
                        f"How to factor simple expressions in {board}?",
                        "What is the difference between linear and quadratic?"
                    ],
                    'high_school': [
                        f"What are the methods to solve quadratic equations in Grade {grade}?",
                        f"Show me {board} board examples of quadratic equations",
                        "How do I identify which method to use?"
                    ]
                },
                'factorization_method': {
                    'middle_school': [
                        "How do I factor simple expressions?",
                        "What is the splitting method for Grade 8?",
                        "When can I use simple factoring?"
                    ],
                    'high_school': [
                        "How do I factor x² + 5x + 6?",
                        "What is splitting the middle term?",
                        "When can I use simple factoring?"
                    ]
                },
                'formula_method': {
                    'high_school': [
                        "What is the quadratic formula?",
                        "How do I use the discriminant?",
                        "Show me step-by-step formula application"
                    ]
                }
            },
            'digestive_system': {
                'general': {
                    'elementary': [
                        f"What happens to food when we eat? (Grade {grade} level)",
                        f"What are the main body parts for digestion?",
                        "Why do we need to chew our food?"
                    ],
                    'middle_school': [
                        f"What parts of digestive system do we study in Grade {grade}?",
                        f"How does digestion work for {board} board?",
                        "What are the main digestive organs?"
                    ],
                    'high_school': [
                        f"Explain detailed digestion process for Grade {grade} {board}",
                        "What are digestive enzymes?",
                        "How does absorption work in small intestine?"
                    ]
                }
            }
        }
        
        if grade <= 5:
            level = 'elementary'
        elif grade <= 8:
            level = 'middle_school'
        else:
            level = 'high_school'
        
        topic_suggestions = suggestions.get(topic, {})
        if subtopic in topic_suggestions:
            return topic_suggestions[subtopic].get(level, topic_suggestions['general'].get(level, []))
        else:
            return topic_suggestions.get('general', {}).get(level, [])