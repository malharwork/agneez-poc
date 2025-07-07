import os
import sys
import logging
import uuid
from flask import Flask, render_template, request, jsonify, session
from dotenv import load_dotenv
from utils.content_generator import ContentGenerator
from utils.embeddings import EmbeddingModel
from utils.vectorstore import VectorStore
from utils.rag import EducationalRAG
from utils.progress_tracker import SQLiteProgressTracker, EnhancedEducationalRAG

load_dotenv()

# Configure logging to show useful information
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

# Get the main logger
logger = logging.getLogger(__name__)

# Set specific log levels for different components
logging.getLogger('werkzeug').setLevel(logging.INFO)  # Flask server logs
logging.getLogger('utils.embeddings').setLevel(logging.INFO)  # Embedding logs
logging.getLogger('utils.vectorstore').setLevel(logging.INFO)  # Pinecone logs  
logging.getLogger('utils.rag').setLevel(logging.INFO)  # RAG logs
logging.getLogger('utils.progress_tracker').setLevel(logging.INFO)  # Progress logs

# Suppress overly verbose logs but keep important ones
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
logging.getLogger('transformers').setLevel(logging.WARNING)
logging.getLogger('torch').setLevel(logging.WARNING)

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-here')

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')

GRADE_TOPIC_MAPPING = {
    'quadratic_equations': {
        'min_grade': 8,  
        'optimal_grades': [9, 10, 11, 12],
        'early_introduction': 8
    },
    'digestive_system': {
        'min_grade': 3,  
        'optimal_grades': [6, 7, 8, 9, 10],
        'early_introduction': 3
    }
}

def is_topic_appropriate_for_grade(topic, grade):
    if topic not in GRADE_TOPIC_MAPPING:
        return True, None
    
    mapping = GRADE_TOPIC_MAPPING[topic]
    min_grade = mapping['min_grade']
    optimal_grades = mapping['optimal_grades']
    
    if grade < min_grade:
        return False, f"This topic is typically taught in grade {min_grade} and above."
    elif grade in optimal_grades:
        return True, None
    else:
        return True, f"This is an advanced topic for your grade level."

def get_or_create_student_id():
    if 'student_id' not in session:
        session['student_id'] = f"student_{uuid.uuid4().hex[:12]}"
    return session['student_id']

print("🚀 Starting Enhanced Adaptive Learning Assistant with Progress Tracking...")

try:
    embedding_model = EmbeddingModel(model_name='sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    
    vector_stores = {
        'math_index': VectorStore(
            api_key=PINECONE_API_KEY,
            environment=PINECONE_ENVIRONMENT,
            index_name='math-index',
            dimension=768
        ),
        'science_index': VectorStore(
            api_key=PINECONE_API_KEY,
            environment=PINECONE_ENVIRONMENT,
            index_name='science-index',
            dimension=768
        )
    }
    
    progress_tracker = SQLiteProgressTracker("student_progress.db")
    rag = EnhancedEducationalRAG(ANTHROPIC_API_KEY, vector_stores, embedding_model, progress_tracker)
    content_generator = ContentGenerator()
    
    print("✅ Components initialized successfully")
    print("📊 Progress tracking database ready")
except Exception as e:
    print(f"❌ Initialization error: {e}")

GRADE_MAPPING = {
    'elementary': {'CBSE': [3, 4, 5], 'ICSE': [3, 4, 5], 'SSC': [3, 4, 5]},
    'middle_school': {'CBSE': [6, 7, 8], 'ICSE': [6, 7, 8], 'SSC': [6, 7, 8]},
    'high_school': {'CBSE': [9, 10, 11, 12], 'ICSE': [9, 10, 11, 12], 'SSC': [9, 10, 11, 12]}
}

def get_level_from_grade(grade):
    if grade <= 5:
        return 'elementary'
    elif grade <= 8:
        return 'middle_school'
    else:
        return 'high_school'

def initialize_knowledge_base():
    try:
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
        
        for topic_key, topic_config in topics_config.items():
            vector_store = vector_stores[topic_config['index']]
            namespace = topic_config['namespace']
            
            existing_stats = vector_store.get_namespace_stats(namespace)
            if existing_stats.get('vector_count', 0) > 0:
                continue
            
            all_chunks = []
            
            for level in levels:
                for board in boards:
                    grades = GRADE_MAPPING[level][board]
                    content_data = content_generator.generate_content(topic_key, level, board)
                    
                    if not content_data:
                        continue
                    
                    for section in content_data.get('sections', []):
                        subtopic_key, sub_method = _categorize_section_detailed(
                            section['title'], 
                            section['content'], 
                            topic_config['subtopics']
                        )
                        
                        content_type = _determine_content_type(section['content'])
                        problem_complexity = _assess_complexity(section['content'], level)
                        learning_stage = _determine_learning_stage(section['title'], content_type)
                        
                        method_tags, excluded_methods = _extract_method_info(
                            section['content'], 
                            subtopic_key, 
                            level, 
                            board
                        )
                        
                        language = 'english'
                        if board == 'SSC' and any(marathi_word in section['content'] for marathi_word in ['वर्ग', 'पचन', 'अवयव']):
                            language = 'marathi'
                        
                        for grade in grades:
                            chunk = {
                                'text': section['content'],
                                'content_id': f"{topic_key}_{subtopic_key}_{sub_method}_{uuid.uuid4().hex[:8]}",
                                'topic': topic_key,
                                'subtopic': subtopic_key,
                                'sub_method': sub_method,
                                'grade': grade,
                                'board': board,
                                'language': language,
                                'difficulty_level': _map_difficulty_to_number(level, grade, grades),
                                'estimated_time_minutes': _estimate_time(content_type, problem_complexity),
                                'method_tags': method_tags,
                                'excluded_methods': excluded_methods,
                                'solution_approach': subtopic_key if 'method' in subtopic_key else 'conceptual',
                                'learning_stage': learning_stage,
                                'prerequisite_concepts': content_data.get('prerequisites', []),
                                'learning_objectives': content_data.get('learning_objectives', []),
                                'content_type': content_type,
                                'problem_complexity': problem_complexity,
                                'has_worked_solution': 'solution' in section['content'].lower() or 'example' in section['content'].lower(),
                                'has_hints': 'hint' in section['content'].lower() or 'tip' in section['content'].lower(),
                                'media_type': 'text_with_equations' if any(char in section['content'] for char in ['²', '×', '÷', '+', '-', '=']) else 'text_only',
                                'average_success_rate': 0.75,
                                'common_errors': _identify_common_errors(topic_key, subtopic_key),
                                'adaptation_weight': 1.0
                            }
                            all_chunks.append(chunk)
            
            if all_chunks:
                embedded_chunks = embedding_model.embed_texts(all_chunks)
                vector_store.add_documents(embedded_chunks, namespace)
                
    except Exception as e:
        pass

def _categorize_section_detailed(title, content, subtopics_config):
    title_lower = title.lower()
    content_lower = content.lower()
    
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
        }
    }
    
    for subtopic, config in categorization_rules.items():
        if subtopic in subtopics_config:
            for keyword in config['keywords']:
                if keyword in title_lower or keyword in content_lower:
                    sub_method = 'general'
                    if 'sub_methods' in config:
                        for method, method_keywords in config['sub_methods'].items():
                            if any(kw in content_lower for kw in method_keywords):
                                sub_method = method
                                break
                    return subtopic, sub_method
    
    return 'general', 'general'

def _determine_content_type(content):
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
    if level == 'elementary':
        return 'simple'
    elif level == 'middle_school':
        if any(term in content.lower() for term in ['advanced', 'complex', 'difficult']):
            return 'moderate_complex'
        return 'moderate_simple'
    else:
        if any(term in content.lower() for term in ['proof', 'derive', 'advanced']):
            return 'complex'
        return 'moderate_complex'

def _determine_learning_stage(title, content_type):
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
    method_tags = []
    excluded_methods = []
    content_lower = content.lower()
    
    if 'quadratic' in content_lower:
        if 'factor' in content_lower:
            method_tags.append('factorization')
        if 'formula' in content_lower and level == 'high_school':
            method_tags.append('quadratic_formula')
        elif 'formula' in content_lower and level != 'high_school':
            excluded_methods.append('quadratic_formula')
        if 'complet' in content_lower and 'square' in content_lower:
            method_tags.append('completing_square')
    
    if 'digest' in content_lower:
        if 'enzyme' in content_lower:
            method_tags.append('enzymatic_process')
        if 'mechanical' in content_lower:
            method_tags.append('mechanical_process')
        if 'absorb' in content_lower or 'absorption' in content_lower:
            method_tags.append('absorption')
    
    return method_tags, excluded_methods

def _map_difficulty_to_number(level, grade, grade_range):
    base_difficulty = {
        'elementary': 1,
        'middle_school': 2,
        'high_school': 3
    }
    grade_position = grade_range.index(grade) if grade in grade_range else 0
    adjustment = grade_position * 0.3
    return min(5, base_difficulty[level] + adjustment)

def _estimate_time(content_type, complexity):
    time_matrix = {
        'concept_explanation': {'simple': 10, 'moderate_simple': 15, 'moderate_complex': 20, 'complex': 30},
        'worked_example': {'simple': 15, 'moderate_simple': 20, 'moderate_complex': 25, 'complex': 35},
        'practice_problem': {'simple': 10, 'moderate_simple': 15, 'moderate_complex': 20, 'complex': 25},
        'activity': {'simple': 20, 'moderate_simple': 30, 'moderate_complex': 40, 'complex': 50},
        'theory': {'simple': 15, 'moderate_simple': 25, 'moderate_complex': 35, 'complex': 45}
    }
    return time_matrix.get(content_type, {}).get(complexity, 15)

def _identify_common_errors(topic, subtopic):
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

with app.app_context():
    initialize_knowledge_base()

print("✅ System ready! Visit http://localhost:5000")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/student/profile', methods=['POST'])
def update_student_profile():
    try:
        data = request.json
        student_id = get_or_create_student_id()
        grade = data.get('grade')
        board = data.get('board')
        language = data.get('language', 'english')
        
        if not all([grade, board]):
            return jsonify({'error': 'Grade and board are required'}), 400
        
        progress_tracker.create_or_update_student(student_id, grade, board, language)
        
        return jsonify({
            'student_id': student_id,
            'status': 'profile_updated',
            'grade': grade,
            'board': board,
            'language': language
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/student/progress', methods=['GET'])
def get_student_progress():
    try:
        student_id = get_or_create_student_id()
        progress_summary = progress_tracker.get_student_progress_summary(student_id)
        return jsonify(progress_summary)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/student/analytics', methods=['GET'])
def get_student_analytics():
    try:
        student_id = get_or_create_student_id()
        days = request.args.get('days', 30, type=int)
        analytics = progress_tracker.get_performance_analytics(student_id, days)
        return jsonify(analytics)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/session/start', methods=['POST'])
def start_learning_session():
    try:
        student_id = get_or_create_student_id()
        session_id = progress_tracker.start_learning_session(student_id)
        session['learning_session_id'] = session_id
        return jsonify({'session_id': session_id, 'status': 'session_started'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/session/end', methods=['POST'])
def end_learning_session():
    try:
        data = request.json
        session_id = session.get('learning_session_id')
        topics_covered = data.get('topics_covered', [])
        
        if session_id:
            progress_tracker.end_learning_session(session_id, topics_covered)
            session.pop('learning_session_id', None)
        
        return jsonify({'status': 'session_ended'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/topics', methods=['GET'])
def get_topics():
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
    try:
        data = request.json
        message = data.get('message', '')
        topic = data.get('topic', '')
        board = data.get('board', '')
        grade = data.get('grade', None)
        subtopic = data.get('subtopic', None)
        method_preference = data.get('method_preference', None)
        exclude_methods = data.get('exclude_methods', [])
        language = data.get('language', 'english')
        
        if not all([message, topic, board]) or grade is None:
            missing_fields = []
            if not message: missing_fields.append('message')
            if not topic: missing_fields.append('topic')
            if not board: missing_fields.append('board')
            if grade is None: missing_fields.append('grade')
            
            return jsonify({
                'error': f'Missing required fields: {", ".join(missing_fields)}'
            }), 400
        
        student_id = get_or_create_student_id()
        progress_tracker.create_or_update_student(student_id, grade, board, language)
        
        is_appropriate, grade_message = is_topic_appropriate_for_grade(topic, grade)
        
        if not is_appropriate:
            topic_name = topic.replace('_', ' ').title()
            response_message = f"""I understand you're curious about {topic_name}! 🌟
            
However, {topic_name} is typically taught in higher grades (usually starting from grade {GRADE_TOPIC_MAPPING[topic]['min_grade']}). 

Right now, you're in grade {grade}, so it's perfectly normal that this topic hasn't been covered yet. Your teachers will introduce you to {topic_name} when you're ready for it in the coming years.

Keep being curious about learning - that's wonderful! For now, you might want to focus on the topics that are part of your current grade curriculum. Is there anything else from your current studies that I can help you with?"""
            
            return jsonify({
                'answer': response_message,
                'grade_appropriate': False,
                'recommended_grade': GRADE_TOPIC_MAPPING[topic]['min_grade'],
                'current_grade': grade
            })
        
        level = get_level_from_grade(grade)
        
        metadata_filter = {
            'grade': grade,
            'board': board,
            'language': language
        }
        
        if subtopic:
            metadata_filter['subtopic'] = subtopic
        
        if method_preference:
            metadata_filter['method_tags'] = method_preference
        
        response = rag.answer_educational_question_with_tracking(
            student_id=student_id,
            question=message,
            topic=topic,
            metadata_filter=metadata_filter,
            level=level
        )
        
        if not response.get('content_metadata') or len(response.get('content_metadata', [])) == 0:
            relaxed_filter = {
                'grade': grade,
                'board': board
            }
            
            response = rag.answer_educational_question_with_tracking(
                student_id=student_id,
                question=message,
                topic=topic,
                metadata_filter=relaxed_filter,
                level=level
            )
        
        response['grade_appropriate'] = True
        response['grade_message'] = grade_message
        
        if not response.get('answer'):
            response['answer'] = f"I found some information about {topic} for Grade {grade} {board}, but let me provide a general explanation based on your question about: {message}"
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'error': f'Internal server error: {str(e)}',
            'type': 'server_error'
        }), 500

@app.route('/api/interaction/record', methods=['POST'])
def record_interaction():
    try:
        data = request.json
        student_id = get_or_create_student_id()
        
        content_id = data.get('content_id', '')
        topic = data.get('topic', '')
        subtopic = data.get('subtopic', '')
        success = data.get('success', True)
        time_taken = data.get('time_taken', 0)
        error_type = data.get('error_type', None)
        difficulty_level = data.get('difficulty_level', None)
        method_tags = data.get('method_tags', [])
        question_text = data.get('question_text', '')
        user_answer = data.get('user_answer', '')
        
        rag.record_student_interaction(
            student_id=student_id,
            content_id=content_id,
            topic=topic,
            success=success,
            subtopic=subtopic,
            time_taken=time_taken,
            error_type=error_type,
            difficulty_level=difficulty_level,
            question_text=question_text,
            user_answer=user_answer
        )
        
        return jsonify({'status': 'interaction_recorded'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/adaptive-content', methods=['POST'])
def get_adaptive_content():
    try:
        data = request.json
        topic = data.get('topic', '')
        current_performance = data.get('performance', 0.5)
        grade = data.get('grade', 9)
        board = data.get('board', 'CBSE')
        subtopic = data.get('subtopic', '')
        language = data.get('language', 'english')
        
        student_id = get_or_create_student_id()
        
        if not topic:
            return jsonify({'error': 'Topic is required'}), 400
        
        is_appropriate, grade_message = is_topic_appropriate_for_grade(topic, grade)
        
        if not is_appropriate:
            return jsonify({
                'adaptive_content': [],
                'grade_appropriate': False,
                'message': f"This topic is typically taught in grade {GRADE_TOPIC_MAPPING[topic]['min_grade']} and above."
            })
        
        current_mastery = progress_tracker.get_student_mastery(student_id, topic)
        
        base_filter = {
            'grade': grade,
            'board': board,
            'language': language
        }
        
        if subtopic:
            base_filter['subtopic'] = subtopic
        
        response = rag.rag.get_adaptive_content(topic, base_filter)
        
        if response.get('adaptive_content'):
            content = response['adaptive_content']
            
            effective_performance = max(current_mastery, current_performance)
            
            if effective_performance < 0.4:
                filtered_content = [c for c in content if c.get('difficulty_level', 3) <= 2]
            elif effective_performance > 0.8:
                filtered_content = [c for c in content if c.get('difficulty_level', 3) >= 3]
            else:
                filtered_content = [c for c in content if 1 <= c.get('difficulty_level', 3) <= 4]
            
            if not filtered_content:
                filtered_content = content[:3]
            
            response['adaptive_content'] = filtered_content[:5]
            response['current_mastery'] = current_mastery
            response['performance_adjusted'] = effective_performance
        
        if not response.get('adaptive_content'):
            minimal_filter = {'board': board, 'grade': grade}
            response = rag.rag.get_adaptive_content(topic, minimal_filter)
        
        response['grade_appropriate'] = True
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/learning-path', methods=['POST'])
def get_learning_path():
    try:
        data = request.json
        topic = data.get('topic', '')
        grade = data.get('grade', 9)
        board = data.get('board', '')
        current_subtopic = data.get('current_subtopic', '')
        mastery_level = data.get('mastery_level', 0.5)
        
        student_id = get_or_create_student_id()
        
        if not all([topic, board]):
            return jsonify({'error': 'Missing required fields: topic and board'}), 400
        
        actual_mastery = progress_tracker.get_student_mastery(student_id, topic)
        effective_mastery = max(actual_mastery, mastery_level)
        
        path = rag.rag.generate_learning_path(
            topic, grade, board, current_subtopic, effective_mastery
        )
        
        path['database_mastery'] = actual_mastery
        path['effective_mastery'] = effective_mastery
        
        recommendations = progress_tracker.get_learning_recommendations(student_id)
        path['personalized_recommendations'] = recommendations
        
        return jsonify(path)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/performance-update', methods=['POST'])
def update_performance():
    try:
        data = request.json
        content_id = data.get('content_id', '')
        success = data.get('success', True)
        error_type = data.get('error_type', None)
        time_taken = data.get('time_taken', 0)
        
        rag.rag.update_performance_metrics(content_id, success, error_type, time_taken)
        return jsonify({'status': 'updated'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, port=5000, use_reloader=False, threaded=True)