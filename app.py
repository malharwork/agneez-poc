import os
import logging
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from utils.content_generator import ContentGenerator
from utils.embeddings import EmbeddingModel
from utils.vectorstore import VectorStore
from utils.rag import EducationalRAG
import uuid

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')

# Initialize embedding model with multilingual support
embedding_model = EmbeddingModel(model_name='sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

# Initialize vector stores according to the new architecture
vector_stores = {
    'math_index': VectorStore(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_ENVIRONMENT,
        index_name='math-index',
        dimension=768  # Updated for new embedding model
    ),
    'science_index': VectorStore(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_ENVIRONMENT,
        index_name='science-index',
        dimension=768
    )
}

# Initialize RAG with multiple vector stores
rag = EducationalRAG(ANTHROPIC_API_KEY, vector_stores, embedding_model)
content_generator = ContentGenerator()

# Grade mapping for different boards
GRADE_MAPPING = {
    'elementary': {'CBSE': [3, 4, 5], 'ICSE': [3, 4, 5], 'SSC': [3, 4, 5]},
    'middle_school': {'CBSE': [6, 7, 8], 'ICSE': [6, 7, 8], 'SSC': [6, 7, 8]},
    'high_school': {'CBSE': [9, 10, 11, 12], 'ICSE': [9, 10, 11, 12], 'SSC': [9, 10, 11, 12]}
}

def initialize_knowledge_base():
    """Initialize the knowledge base with the new architecture."""
    try:
        # Topics configuration with proper namespace structure
        topics_config = {
            'quadratic_equations': {
                'index': 'math_index',
                'namespace': 'algebra_quadratic_equations',
                'subtopics': {
                    'patterns_introduction': {
                        'name': 'Patterns and Square Numbers',
                        'sub_methods': ['visual_patterns', 'number_sequences']
                    },
                    'factorization_method': {
                        'name': 'Solving by Factorization',
                        'sub_methods': ['simple_factoring', 'splitting_middle_term', 'grouping']
                    },
                    'formula_method': {
                        'name': 'Quadratic Formula',
                        'sub_methods': ['derivation', 'application', 'discriminant_analysis']
                    },
                    'completing_square': {
                        'name': 'Completing the Square',
                        'sub_methods': ['geometric_interpretation', 'algebraic_method']
                    },
                    'applications': {
                        'name': 'Real-world Applications',
                        'sub_methods': ['physics_problems', 'optimization', 'geometry']
                    }
                }
            },
            'digestive_system': {
                'index': 'science_index',
                'namespace': 'biology_digestive_system',
                'subtopics': {
                    'anatomy_structure': {
                        'name': 'Anatomical Structure',
                        'sub_methods': ['organs', 'tissues', 'cellular_structure']
                    },
                    'digestion_process': {
                        'name': 'Process of Digestion',
                        'sub_methods': ['mechanical_digestion', 'chemical_digestion', 'peristalsis']
                    },
                    'enzymes_secretions': {
                        'name': 'Enzymes and Secretions',
                        'sub_methods': ['digestive_enzymes', 'hormonal_control', 'pH_regulation']
                    },
                    'absorption_transport': {
                        'name': 'Absorption and Transport',
                        'sub_methods': ['villi_function', 'nutrient_transport', 'water_absorption']
                    },
                    'disorders_health': {
                        'name': 'Disorders and Health',
                        'sub_methods': ['common_disorders', 'prevention', 'dietary_management']
                    }
                }
            }
        }
        
        levels = ['elementary', 'middle_school', 'high_school']
        boards = ['CBSE', 'ICSE', 'SSC']
        languages = ['english', 'hindi', 'marathi']  # For SSC board
        
        for topic_key, topic_config in topics_config.items():
            vector_store = vector_stores[topic_config['index']]
            namespace = topic_config['namespace']
            
            # Check if namespace already has content
            existing_stats = vector_store.get_namespace_stats(namespace)
            if existing_stats.get('vector_count', 0) > 0:
                logger.info(f"Namespace {namespace} already has {existing_stats['vector_count']} vectors")
                continue
            
            logger.info(f"Initializing namespace {namespace}")
            all_chunks = []
            
            for level in levels:
                for board in boards:
                    # Determine appropriate grades
                    grades = GRADE_MAPPING[level][board]
                    
                    # Generate content
                    content_data = content_generator.generate_content(topic_key, level, board)
                    
                    if not content_data:
                        logger.warning(f"No content found for {topic_key}_{level}_{board}")
                        continue
                    
                    # Process each section with enhanced metadata
                    for section in content_data.get('sections', []):
                        # Determine subtopic and sub_method
                        subtopic_key, sub_method = _categorize_section_detailed(
                            section['title'], 
                            section['content'], 
                            topic_config['subtopics']
                        )
                        
                        # Determine content properties
                        content_type = _determine_content_type(section['content'])
                        problem_complexity = _assess_complexity(section['content'], level)
                        learning_stage = _determine_learning_stage(section['title'], content_type)
                        
                        # Extract method tags and excluded methods
                        method_tags, excluded_methods = _extract_method_info(
                            section['content'], 
                            subtopic_key, 
                            level, 
                            board
                        )
                        
                        # Determine language
                        language = 'english'
                        if board == 'SSC' and any(marathi_word in section['content'] for marathi_word in ['वर्ग', 'पचन', 'अवयव']):
                            language = 'marathi'
                        
                        # Create chunk with comprehensive metadata
                        for grade in grades:
                            chunk = {
                                'text': section['content'],
                                'content_id': f"{topic_key}_{subtopic_key}_{sub_method}_{uuid.uuid4().hex[:8]}",
                                'topic': topic_key,
                                'subtopic': subtopic_key,
                                'sub_method': sub_method,
                                
                                # Academic metadata
                                'grade': grade,
                                'board': board,
                                'language': language,
                                'difficulty_level': _map_difficulty_to_number(level, grade, grades),
                                'estimated_time_minutes': _estimate_time(content_type, problem_complexity),
                                
                                # Learning metadata
                                'method_tags': method_tags,
                                'excluded_methods': excluded_methods,
                                'solution_approach': subtopic_key if 'method' in subtopic_key else 'conceptual',
                                'learning_stage': learning_stage,
                                'prerequisite_concepts': content_data.get('prerequisites', []),
                                'learning_objectives': content_data.get('learning_objectives', []),
                                
                                # Content metadata
                                'content_type': content_type,
                                'problem_complexity': problem_complexity,
                                'has_worked_solution': 'solution' in section['content'].lower() or 'example' in section['content'].lower(),
                                'has_hints': 'hint' in section['content'].lower() or 'tip' in section['content'].lower(),
                                'media_type': 'text_with_equations' if any(char in section['content'] for char in ['²', '×', '÷', '+', '-', '=']) else 'text_only',
                                
                                # Performance metadata (initialized)
                                'average_success_rate': 0.75,  # Default, to be updated based on usage
                                'common_errors': _identify_common_errors(topic_key, subtopic_key),
                                'adaptation_weight': 1.0
                            }
                            all_chunks.append(chunk)
            
            # Embed and store all chunks
            if all_chunks:
                embedded_chunks = embedding_model.embed_texts(all_chunks)
                vector_store.add_documents(embedded_chunks, namespace)
                logger.info(f"Added {len(all_chunks)} chunks to namespace {namespace}")
                
    except Exception as e:
        logger.error(f"Error initializing knowledge base: {str(e)}")

def _categorize_section_detailed(title, content, subtopics_config):
    """Categorize section into subtopic and sub_method."""
    title_lower = title.lower()
    content_lower = content.lower()
    
    # Keywords for detailed categorization
    categorization_rules = {
        'patterns_introduction': {
            'keywords': ['pattern', 'square number', 'sequence', 'वर्ग संख्या'],
            'sub_methods': {
                'visual_patterns': ['visual', 'arrange', 'dots', 'blocks'],
                'number_sequences': ['sequence', 'series', 'differences']
            }
        },
        'factorization_method': {
            'keywords': ['factor', 'factorization', 'अवयव'],
            'sub_methods': {
                'simple_factoring': ['simple', 'basic factor'],
                'splitting_middle_term': ['split', 'middle term'],
                'grouping': ['group', 'grouping method']
            }
        },
        'formula_method': {
            'keywords': ['formula', 'quadratic formula', 'सूत्र'],
            'sub_methods': {
                'derivation': ['derive', 'proof'],
                'application': ['apply', 'use formula'],
                'discriminant_analysis': ['discriminant', 'nature of roots']
            }
        },
        'anatomy_structure': {
            'keywords': ['structure', 'organ', 'anatomy', 'रचना'],
            'sub_methods': {
                'organs': ['stomach', 'intestine', 'liver'],
                'tissues': ['tissue', 'epithelium', 'muscle'],
                'cellular_structure': ['cell', 'villi', 'microvilli']
            }
        },
        'digestion_process': {
            'keywords': ['process', 'digestion', 'पचन'],
            'sub_methods': {
                'mechanical_digestion': ['chew', 'mechanical', 'physical'],
                'chemical_digestion': ['enzyme', 'chemical', 'breakdown'],
                'peristalsis': ['peristalsis', 'movement', 'wave']
            }
        }
    }
    
    # Find matching subtopic
    for subtopic, config in categorization_rules.items():
        if subtopic in subtopics_config:
            for keyword in config['keywords']:
                if keyword in title_lower or keyword in content_lower:
                    # Find sub_method
                    sub_method = 'general'
                    if 'sub_methods' in config:
                        for method, method_keywords in config['sub_methods'].items():
                            if any(kw in content_lower for kw in method_keywords):
                                sub_method = method
                                break
                    return subtopic, sub_method
    
    # Default fallback
    return 'general', 'general'

def _determine_content_type(content):
    """Determine the type of content."""
    content_lower = content.lower()
    
    if 'example:' in content_lower or 'solve:' in content_lower:
        return 'worked_example'
    elif 'problem:' in content_lower or 'exercise:' in content_lower:
        return 'practice_problem'
    elif 'definition:' in content_lower or 'what is' in content_lower:
        return 'concept_explanation'
    elif 'activity:' in content_lower or 'project:' in content_lower:
        return 'activity'
    elif 'theorem:' in content_lower or 'proof:' in content_lower:
        return 'theory'
    else:
        return 'general_content'

def _assess_complexity(content, level):
    """Assess the complexity of the content."""
    if level == 'elementary':
        return 'simple'
    elif level == 'middle_school':
        if any(term in content.lower() for term in ['advanced', 'complex', 'difficult']):
            return 'moderate_complex'
        return 'moderate_simple'
    else:  # high_school
        if any(term in content.lower() for term in ['proof', 'derive', 'advanced']):
            return 'complex'
        return 'moderate_complex'

def _determine_learning_stage(title, content_type):
    """Determine the learning stage."""
    title_lower = title.lower()
    
    if 'introduction' in title_lower or 'what is' in title_lower:
        return 'introduction'
    elif 'practice' in title_lower or content_type == 'practice_problem':
        return 'practice'
    elif 'advanced' in title_lower or 'application' in title_lower:
        return 'application'
    elif 'review' in title_lower or 'summary' in title_lower:
        return 'review'
    else:
        return 'learning'

def _extract_method_info(content, subtopic, level, board):
    """Extract method tags and excluded methods based on content."""
    method_tags = []
    excluded_methods = []
    
    content_lower = content.lower()
    
    # For quadratic equations
    if 'quadratic' in content_lower:
        if 'factor' in content_lower:
            method_tags.append('factorization')
        if 'formula' in content_lower and level == 'high_school':
            method_tags.append('quadratic_formula')
        elif 'formula' in content_lower and level != 'high_school':
            excluded_methods.append('quadratic_formula')
        if 'complet' in content_lower and 'square' in content_lower:
            method_tags.append('completing_square')
        
        # Board-specific exclusions
        if board == 'SSC' and level == 'middle_school':
            excluded_methods.extend(['complex_numbers', 'advanced_factoring'])
    
    # For digestive system
    if 'digest' in content_lower:
        if 'enzyme' in content_lower:
            method_tags.append('enzymatic_process')
        if 'mechanical' in content_lower:
            method_tags.append('mechanical_process')
        if 'absorb' in content_lower or 'absorption' in content_lower:
            method_tags.append('absorption')
    
    return method_tags, excluded_methods

def _map_difficulty_to_number(level, grade, grade_range):
    """Map difficulty to a number 1-5."""
    base_difficulty = {
        'elementary': 1,
        'middle_school': 2,
        'high_school': 3
    }
    
    # Adjust based on grade within level
    grade_position = grade_range.index(grade) if grade in grade_range else 0
    adjustment = grade_position * 0.3
    
    return min(5, base_difficulty[level] + adjustment)

def _estimate_time(content_type, complexity):
    """Estimate time needed for the content."""
    time_matrix = {
        'concept_explanation': {'simple': 10, 'moderate_simple': 15, 'moderate_complex': 20, 'complex': 30},
        'worked_example': {'simple': 15, 'moderate_simple': 20, 'moderate_complex': 25, 'complex': 35},
        'practice_problem': {'simple': 10, 'moderate_simple': 15, 'moderate_complex': 20, 'complex': 25},
        'activity': {'simple': 20, 'moderate_simple': 30, 'moderate_complex': 40, 'complex': 50},
        'theory': {'simple': 15, 'moderate_simple': 25, 'moderate_complex': 35, 'complex': 45}
    }
    
    return time_matrix.get(content_type, {}).get(complexity, 15)

def _identify_common_errors(topic, subtopic):
    """Identify common errors for the topic/subtopic."""
    error_map = {
        'quadratic_equations': {
            'factorization_method': ['sign_mistakes', 'calculation_errors', 'wrong_factors'],
            'formula_method': ['calculation_errors', 'discriminant_mistakes', 'formula_memorization'],
            'default': ['algebraic_manipulation', 'arithmetic_errors']
        },
        'digestive_system': {
            'anatomy_structure': ['organ_sequence_errors', 'location_confusion'],
            'digestion_process': ['process_order_errors', 'enzyme_function_confusion'],
            'default': ['terminology_confusion', 'process_understanding']
        }
    }
    
    return error_map.get(topic, {}).get(subtopic, error_map.get(topic, {}).get('default', []))

# Initialize on startup
with app.app_context():
    initialize_knowledge_base()

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/api/topics', methods=['GET'])
def get_topics():
    """Get available topics with full metadata structure."""
    topics = {
        'quadratic_equations': {
            'name': 'Quadratic Equations',
            'index': 'math_index',
            'namespace': 'algebra_quadratic_equations',
            'subtopics': {
                'patterns_introduction': 'Patterns & Square Numbers',
                'factorization_method': 'Factorization Methods',
                'formula_method': 'Quadratic Formula',
                'completing_square': 'Completing the Square',
                'applications': 'Applications'
            },
            'grades': list(range(3, 13)),
            'boards': ['CBSE', 'ICSE', 'SSC'],
            'languages': ['english', 'hindi', 'marathi']
        },
        'digestive_system': {
            'name': 'Digestive System',
            'index': 'science_index',
            'namespace': 'biology_digestive_system',
            'subtopics': {
                'anatomy_structure': 'Anatomical Structure',
                'digestion_process': 'Digestion Process',
                'enzymes_secretions': 'Enzymes & Secretions',
                'absorption_transport': 'Absorption & Transport',
                'disorders_health': 'Health & Disorders'
            },
            'grades': list(range(3, 13)),
            'boards': ['CBSE', 'ICSE', 'SSC'],
            'languages': ['english', 'hindi', 'marathi']
        }
    }
    return jsonify(topics)

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat requests with enhanced filtering."""
    try:
        data = request.json
        message = data.get('message', '')
        topic = data.get('topic', '')
        level = data.get('level', '')
        board = data.get('board', '')
        grade = data.get('grade', None)
        subtopic = data.get('subtopic', None)
        method_preference = data.get('method_preference', None)
        exclude_methods = data.get('exclude_methods', [])
        language = data.get('language', 'english')
        
        if not all([message, topic, level, board]):
            return jsonify({'error': 'Missing required fields'}), 400
        
        # If grade not specified, use middle grade for the level
        if not grade:
            grades = GRADE_MAPPING[level][board]
            grade = grades[len(grades)//2]
        
        # Build comprehensive filter
        metadata_filter = {
            'grade': grade,
            'board': board,
            'language': language
        }
        
        if subtopic:
            metadata_filter['subtopic'] = subtopic
        
        if method_preference:
            metadata_filter['method_tags'] = {"$in": [method_preference]}
        
        if exclude_methods:
            metadata_filter['excluded_methods'] = {"$nin": exclude_methods}
        
        # Get response from RAG
        response = rag.answer_educational_question(
            message, topic, metadata_filter, level
        )
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/adaptive-content', methods=['POST'])
def get_adaptive_content():
    """Get content adapted to student's performance."""
    try:
        data = request.json
        topic = data.get('topic', '')
        current_performance = data.get('performance', 0.5)
        grade = data.get('grade', 9)
        board = data.get('board', 'CBSE')
        subtopic = data.get('subtopic', '')
        
        # Adjust difficulty based on performance
        if current_performance < 0.4:
            difficulty_range = {"$lte": 2}
        elif current_performance > 0.8:
            difficulty_range = {"$gte": 3}
        else:
            difficulty_range = {"$gte": 2, "$lte": 4}
        
        metadata_filter = {
            'grade': grade,
            'board': board,
            'subtopic': subtopic,
            'difficulty_level': difficulty_range
        }
        
        # Get adaptive content
        response = rag.get_adaptive_content(topic, metadata_filter)
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in adaptive content endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/learning-path', methods=['POST'])
def get_learning_path():
    """Get a personalized learning path."""
    try:
        data = request.json
        topic = data.get('topic', '')
        grade = data.get('grade', 9)
        board = data.get('board', '')
        current_subtopic = data.get('current_subtopic', '')
        mastery_level = data.get('mastery_level', 0.5)
        
        if not all([topic, board]):
            return jsonify({'error': 'Missing required fields'}), 400
        
        path = rag.generate_learning_path(
            topic, grade, board, current_subtopic, mastery_level
        )
        return jsonify(path)
        
    except Exception as e:
        logger.error(f"Error getting learning path: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/performance-update', methods=['POST'])
def update_performance():
    """Update performance metrics for content."""
    try:
        data = request.json
        content_id = data.get('content_id', '')
        success = data.get('success', True)
        error_type = data.get('error_type', None)
        time_taken = data.get('time_taken', 0)
        
        # Update performance metadata in vector store
        rag.update_performance_metrics(content_id, success, error_type, time_taken)
        
        return jsonify({'status': 'updated'})
        
    except Exception as e:
        logger.error(f"Error updating performance: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)