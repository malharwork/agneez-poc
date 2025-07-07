import sqlite3
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class SQLiteProgressTracker:
    def __init__(self, db_path: str = "student_progress.db"):
        self.db_path = db_path
        self.init_database()
    
    @contextmanager
    def get_db_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  
        try:
            yield conn
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()
    
    def init_database(self):
        """Initialize SQLite database with all required tables"""
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Students table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS students (
                    student_id TEXT PRIMARY KEY,
                    grade INTEGER,
                    board TEXT,
                    language TEXT DEFAULT 'english',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    total_questions INTEGER DEFAULT 0,
                    total_correct INTEGER DEFAULT 0,
                    total_time_minutes INTEGER DEFAULT 0
                )
            ''')
            
            # Student interactions table (detailed log of every interaction)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id TEXT,
                    content_id TEXT,
                    topic TEXT,
                    subtopic TEXT,
                    success BOOLEAN,
                    time_taken INTEGER DEFAULT 0,
                    error_type TEXT,
                    difficulty_level INTEGER,
                    method_tags TEXT,  -- JSON array of method tags
                    question_text TEXT,
                    user_answer TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (student_id) REFERENCES students (student_id)
                )
            ''')
            
            # Topic mastery table (aggregated data per topic)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS topic_mastery (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id TEXT,
                    topic TEXT,
                    total_attempts INTEGER DEFAULT 0,
                    correct_attempts INTEGER DEFAULT 0,
                    total_time INTEGER DEFAULT 0,
                    mastery_level REAL DEFAULT 0.0,
                    last_attempt TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (student_id) REFERENCES students (student_id),
                    UNIQUE(student_id, topic)
                )
            ''')
            
            # Subtopic mastery table (detailed progress per subtopic)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS subtopic_mastery (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id TEXT,
                    topic TEXT,
                    subtopic TEXT,
                    attempts INTEGER DEFAULT 0,
                    correct INTEGER DEFAULT 0,
                    mastery_level REAL DEFAULT 0.0,
                    last_attempt TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (student_id) REFERENCES students (student_id),
                    UNIQUE(student_id, topic, subtopic)
                )
            ''')
            
            # Learning sessions table (track study sessions)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS learning_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id TEXT,
                    session_start TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    session_end TIMESTAMP,
                    topics_covered TEXT,  -- JSON array of topics
                    questions_attempted INTEGER DEFAULT 0,
                    questions_correct INTEGER DEFAULT 0,
                    total_time_minutes INTEGER DEFAULT 0,
                    FOREIGN KEY (student_id) REFERENCES students (student_id)
                )
            ''')
            
            # Error patterns table (track common mistakes)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS error_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id TEXT,
                    topic TEXT,
                    error_type TEXT,
                    frequency INTEGER DEFAULT 1,
                    last_occurrence TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (student_id) REFERENCES students (student_id),
                    UNIQUE(student_id, topic, error_type)
                )
            ''')
            
            # Create indexes for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_interactions_student_topic ON interactions(student_id, topic)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_interactions_timestamp ON interactions(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_topic_mastery_student ON topic_mastery(student_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_subtopic_mastery_student ON subtopic_mastery(student_id, topic)')
            
            conn.commit()
            logger.info("Database initialized successfully")
    
    def create_or_update_student(self, student_id: str, grade: int, board: str, language: str = 'english') -> bool:
        """Create new student or update existing student profile"""
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO students 
                (student_id, grade, board, language, last_active)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (student_id, grade, board, language))
            
            conn.commit()
            return True
    
    def record_interaction(self, student_id: str, content_id: str, topic: str, 
                         success: bool, subtopic: str = None, time_taken: int = 0,
                         error_type: str = None, difficulty_level: int = None,
                         method_tags: List[str] = None, question_text: str = None,
                         user_answer: str = None) -> bool:
        """Record a detailed interaction"""
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Insert interaction record
            cursor.execute('''
                INSERT INTO interactions 
                (student_id, content_id, topic, subtopic, success, time_taken, 
                 error_type, difficulty_level, method_tags, question_text, user_answer)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (student_id, content_id, topic, subtopic, success, time_taken,
                  error_type, difficulty_level, json.dumps(method_tags or []),
                  question_text, user_answer))
            
            # Update student totals
            cursor.execute('''
                UPDATE students 
                SET total_questions = total_questions + 1,
                    total_correct = total_correct + ?,
                    total_time_minutes = total_time_minutes + ?,
                    last_active = CURRENT_TIMESTAMP
                WHERE student_id = ?
            ''', (1 if success else 0, time_taken, student_id))
            
            # Update topic mastery
            cursor.execute('''
                INSERT OR REPLACE INTO topic_mastery 
                (student_id, topic, total_attempts, correct_attempts, total_time, last_attempt)
                VALUES (
                    ?, ?, 
                    COALESCE((SELECT total_attempts FROM topic_mastery WHERE student_id = ? AND topic = ?), 0) + 1,
                    COALESCE((SELECT correct_attempts FROM topic_mastery WHERE student_id = ? AND topic = ?), 0) + ?,
                    COALESCE((SELECT total_time FROM topic_mastery WHERE student_id = ? AND topic = ?), 0) + ?,
                    CURRENT_TIMESTAMP
                )
            ''', (student_id, topic, student_id, topic, student_id, topic, 
                  1 if success else 0, student_id, topic, time_taken))
            
            # Calculate and update mastery level
            cursor.execute('''
                UPDATE topic_mastery 
                SET mastery_level = CAST(correct_attempts AS REAL) / total_attempts
                WHERE student_id = ? AND topic = ?
            ''', (student_id, topic))
            
            # Update subtopic mastery if provided
            if subtopic:
                cursor.execute('''
                    INSERT OR REPLACE INTO subtopic_mastery 
                    (student_id, topic, subtopic, attempts, correct, last_attempt)
                    VALUES (
                        ?, ?, ?,
                        COALESCE((SELECT attempts FROM subtopic_mastery WHERE student_id = ? AND topic = ? AND subtopic = ?), 0) + 1,
                        COALESCE((SELECT correct FROM subtopic_mastery WHERE student_id = ? AND topic = ? AND subtopic = ?), 0) + ?,
                        CURRENT_TIMESTAMP
                    )
                ''', (student_id, topic, subtopic, student_id, topic, subtopic,
                      student_id, topic, subtopic, 1 if success else 0))
                
                # Calculate subtopic mastery level
                cursor.execute('''
                    UPDATE subtopic_mastery 
                    SET mastery_level = CAST(correct AS REAL) / attempts
                    WHERE student_id = ? AND topic = ? AND subtopic = ?
                ''', (student_id, topic, subtopic))
            
            # Track error patterns
            if error_type and not success:
                cursor.execute('''
                    INSERT OR REPLACE INTO error_patterns 
                    (student_id, topic, error_type, frequency, last_occurrence)
                    VALUES (
                        ?, ?, ?,
                        COALESCE((SELECT frequency FROM error_patterns WHERE student_id = ? AND topic = ? AND error_type = ?), 0) + 1,
                        CURRENT_TIMESTAMP
                    )
                ''', (student_id, topic, error_type, student_id, topic, error_type))
            
            conn.commit()
            return True
    
    def get_student_mastery(self, student_id: str, topic: str) -> float:
        """Get student's mastery level for a specific topic"""
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT mastery_level FROM topic_mastery 
                WHERE student_id = ? AND topic = ?
            ''', (student_id, topic))
            
            result = cursor.fetchone()
            return result['mastery_level'] if result else 0.0
    
    def get_student_progress_summary(self, student_id: str) -> Dict[str, Any]:
        """Get comprehensive progress summary for a student"""
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Get student basic info
            cursor.execute('SELECT * FROM students WHERE student_id = ?', (student_id,))
            student = cursor.fetchone()
            
            if not student:
                return {"error": "Student not found"}
            
            # Get topic mastery data
            cursor.execute('''
                SELECT topic, total_attempts, correct_attempts, mastery_level, 
                       total_time, last_attempt
                FROM topic_mastery 
                WHERE student_id = ?
                ORDER BY last_attempt DESC
            ''', (student_id,))
            topics = cursor.fetchall()
            
            # Get subtopic details for each topic
            topic_details = {}
            for topic in topics:
                cursor.execute('''
                    SELECT subtopic, attempts, correct, mastery_level
                    FROM subtopic_mastery 
                    WHERE student_id = ? AND topic = ?
                    ORDER BY mastery_level ASC
                ''', (student_id, topic['topic']))
                
                subtopics = cursor.fetchall()
                
                # Get common errors for this topic
                cursor.execute('''
                    SELECT error_type, frequency
                    FROM error_patterns 
                    WHERE student_id = ? AND topic = ?
                    ORDER BY frequency DESC
                    LIMIT 5
                ''', (student_id, topic['topic']))
                
                errors = cursor.fetchall()
                
                topic_details[topic['topic']] = {
                    'mastery_level': topic['mastery_level'],
                    'total_attempts': topic['total_attempts'],
                    'correct_attempts': topic['correct_attempts'],
                    'accuracy': f"{(topic['correct_attempts'] / max(1, topic['total_attempts']) * 100):.1f}%",
                    'time_spent_minutes': topic['total_time'],
                    'last_attempt': topic['last_attempt'],
                    'subtopics': [dict(sub) for sub in subtopics],
                    'common_errors': [dict(err) for err in errors],
                    'weakest_subtopic': min(subtopics, key=lambda x: x['mastery_level'], default=None)
                }
            
            # Get recent activity (last 10 interactions)
            cursor.execute('''
                SELECT topic, subtopic, success, time_taken, timestamp, difficulty_level
                FROM interactions 
                WHERE student_id = ?
                ORDER BY timestamp DESC
                LIMIT 10
            ''', (student_id,))
            
            recent_activity = [dict(row) for row in cursor.fetchall()]
            
            # Calculate overall statistics
            overall_accuracy = (student['total_correct'] / max(1, student['total_questions'])) * 100
            
            return {
                'student_info': {
                    'student_id': student_id,
                    'grade': student['grade'],
                    'board': student['board'],
                    'language': student['language'],
                    'member_since': student['created_at'],
                    'last_active': student['last_active']
                },
                'overall_stats': {
                    'total_questions': student['total_questions'],
                    'total_correct': student['total_correct'],
                    'overall_accuracy': f"{overall_accuracy:.1f}%",
                    'total_time_hours': f"{(student['total_time_minutes'] / 60):.1f}",
                    'topics_attempted': len(topics)
                },
                'topic_progress': topic_details,
                'recent_activity': recent_activity,
                'learning_recommendations': self.get_learning_recommendations(student_id)
            }
    
    def get_learning_recommendations(self, student_id: str) -> Dict[str, Any]:
        """Generate personalized learning recommendations"""
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Get topics with low mastery (need remediation)
            cursor.execute('''
                SELECT topic, mastery_level, total_attempts
                FROM topic_mastery 
                WHERE student_id = ? AND mastery_level < 0.6
                ORDER BY mastery_level ASC
            ''', (student_id,))
            
            weak_topics = [dict(row) for row in cursor.fetchall()]
            
            # Get topics with good mastery (ready for advancement)
            cursor.execute('''
                SELECT topic, mastery_level, total_attempts
                FROM topic_mastery 
                WHERE student_id = ? AND mastery_level >= 0.8
                ORDER BY mastery_level DESC
            ''', (student_id,))
            
            strong_topics = [dict(row) for row in cursor.fetchall()]
            
            # Get most common error patterns across all topics
            cursor.execute('''
                SELECT error_type, SUM(frequency) as total_frequency
                FROM error_patterns 
                WHERE student_id = ?
                GROUP BY error_type
                ORDER BY total_frequency DESC
                LIMIT 3
            ''', (student_id,))
            
            common_errors = [dict(row) for row in cursor.fetchall()]
            
            recommendations = {
                'priority_actions': [],
                'weak_areas': weak_topics,
                'strong_areas': strong_topics,
                'focus_errors': common_errors
            }
            
            # Generate specific recommendations
            if weak_topics:
                recommendations['priority_actions'].append({
                    'action': 'remediation',
                    'description': f"Focus on {weak_topics[0]['topic']} - current mastery: {weak_topics[0]['mastery_level']:.1%}",
                    'urgency': 'high'
                })
            
            if strong_topics:
                recommendations['priority_actions'].append({
                    'action': 'advancement',
                    'description': f"Try advanced problems in {strong_topics[0]['topic']} - current mastery: {strong_topics[0]['mastery_level']:.1%}",
                    'urgency': 'medium'
                })
            
            if common_errors:
                recommendations['priority_actions'].append({
                    'action': 'error_correction',
                    'description': f"Practice to avoid {common_errors[0]['error_type']} errors",
                    'urgency': 'medium'
                })
            
            return recommendations
    
    def get_performance_analytics(self, student_id: str, days: int = 30) -> Dict[str, Any]:
        """Get detailed performance analytics for the last N days"""
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            
            cutoff_date = datetime.now() - timedelta(days=days)
            
            # Daily performance trend
            cursor.execute('''
                SELECT DATE(timestamp) as date,
                       COUNT(*) as questions,
                       SUM(CASE WHEN success THEN 1 ELSE 0 END) as correct,
                       AVG(time_taken) as avg_time
                FROM interactions 
                WHERE student_id = ? AND timestamp >= ?
                GROUP BY DATE(timestamp)
                ORDER BY date
            ''', (student_id, cutoff_date))
            
            daily_performance = [dict(row) for row in cursor.fetchall()]
            
            # Performance by difficulty level
            cursor.execute('''
                SELECT difficulty_level,
                       COUNT(*) as attempts,
                       AVG(CASE WHEN success THEN 1.0 ELSE 0.0 END) as success_rate
                FROM interactions 
                WHERE student_id = ? AND timestamp >= ? AND difficulty_level IS NOT NULL
                GROUP BY difficulty_level
                ORDER BY difficulty_level
            ''', (student_id, cutoff_date))
            
            difficulty_performance = [dict(row) for row in cursor.fetchall()]
            
            # Time of day performance
            cursor.execute('''
                SELECT CAST(strftime('%H', timestamp) AS INTEGER) as hour,
                       COUNT(*) as questions,
                       AVG(CASE WHEN success THEN 1.0 ELSE 0.0 END) as accuracy
                FROM interactions 
                WHERE student_id = ? AND timestamp >= ?
                GROUP BY hour
                ORDER BY hour
            ''', (student_id, cutoff_date))
            
            hourly_performance = [dict(row) for row in cursor.fetchall()]
            
            return {
                'analysis_period_days': days,
                'daily_trend': daily_performance,
                'difficulty_breakdown': difficulty_performance,
                'hourly_patterns': hourly_performance
            }
    
    def start_learning_session(self, student_id: str) -> int:
        """Start a new learning session and return session ID"""
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO learning_sessions (student_id, session_start)
                VALUES (?, CURRENT_TIMESTAMP)
            ''', (student_id,))
            
            session_id = cursor.lastrowid
            conn.commit()
            return session_id
    
    def end_learning_session(self, session_id: int, topics_covered: List[str]) -> bool:
        """End a learning session and update statistics"""
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Get session data
            cursor.execute('''
                SELECT student_id, session_start FROM learning_sessions 
                WHERE id = ?
            ''', (session_id,))
            
            session = cursor.fetchone()
            if not session:
                return False
            
            # Calculate session statistics
            cursor.execute('''
                SELECT COUNT(*) as questions,
                       SUM(CASE WHEN success THEN 1 ELSE 0 END) as correct,
                       SUM(time_taken) as total_time
                FROM interactions 
                WHERE student_id = ? AND timestamp >= ?
            ''', (session['student_id'], session['session_start']))
            
            stats = cursor.fetchone()
            
            # Update session record
            cursor.execute('''
                UPDATE learning_sessions 
                SET session_end = CURRENT_TIMESTAMP,
                    topics_covered = ?,
                    questions_attempted = ?,
                    questions_correct = ?,
                    total_time_minutes = ?
                WHERE id = ?
            ''', (json.dumps(topics_covered), stats['questions'], 
                  stats['correct'], stats['total_time'], session_id))
            
            conn.commit()
            return True

# Integration with the existing RAG system
class EnhancedEducationalRAG:
    def __init__(self, anthropic_api_key: str, vector_stores: Dict[str, Any], 
                 embedding_model, progress_tracker: SQLiteProgressTracker):
        from utils.rag import EducationalRAG
        self.rag = EducationalRAG(anthropic_api_key, vector_stores, embedding_model)
        self.progress_tracker = progress_tracker
    
    def answer_educational_question_with_tracking(self, student_id: str, question: str, 
                                                topic: str, metadata_filter: Dict, 
                                                level: str = None, user_answer: str = None):
        """Answer question and track student interaction"""
        
        # Get the regular RAG response
        response = self.rag.answer_educational_question(question, topic, metadata_filter, level)
        
        # Get student's current mastery for personalization
        current_mastery = self.progress_tracker.get_student_mastery(student_id, topic)
        
        # Add personalized recommendations to response
        recommendations = self.progress_tracker.get_learning_recommendations(student_id)
        
        response.update({
            'student_mastery': f"{current_mastery:.1%}",
            'recommendations': recommendations.get('priority_actions', [])[:2],  # Top 2 recommendations
            'tracking_enabled': True
        })
        
        return response
    
    def record_student_interaction(self, student_id: str, content_id: str, topic: str,
                                 success: bool, subtopic: str = None, time_taken: int = 0,
                                 error_type: str = None, difficulty_level: int = None,
                                 question_text: str = None, user_answer: str = None):
        """Record student interaction with detailed tracking"""
        
        return self.progress_tracker.record_interaction(
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