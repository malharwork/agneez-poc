import React, { useState, useEffect } from 'react';
import axios from 'axios';
import LessonPlayer from './LessonPlayer';

/**
 * Educational RAG Integration Component
 * This component provides integration with the Educational RAG system
 * and can be embedded in existing React applications.
 */
const RagIntegration = ({
  defaultTopic = null,
  defaultGrade = null,
  defaultBoard = null,
  defaultSubtopic = null,
  onLessonComplete = () => {},
  mode = 'chat', // 'chat', 'lesson', or 'adaptive'
  className = '',
  style = {}
}) => {
  // State
  const [topic, setTopic] = useState(defaultTopic);
  const [grade, setGrade] = useState(defaultGrade);
  const [board, setBoard] = useState(defaultBoard);
  const [subtopic, setSubtopic] = useState(defaultSubtopic);
  const [message, setMessage] = useState('');
  const [subtopics, setSubtopics] = useState([]);
  const [topics, setTopics] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // Chat-specific state
  const [chatHistory, setChatHistory] = useState([]);
  
  // Lesson-specific state
  const [lessonScript, setLessonScript] = useState(null);
  
  // Adaptive content-specific state
  const [adaptiveContent, setAdaptiveContent] = useState(null);
  
  // Load available topics on component mount
  useEffect(() => {
    const fetchTopics = async () => {
      try {
        const response = await axios.get('/api/topics');
        setTopics(response.data);
        
        // If a default topic is set, load its subtopics
        if (defaultTopic && response.data[defaultTopic]) {
          setSubtopics(Object.entries(response.data[defaultTopic].subtopics).map(([key, value]) => ({
            id: key,
            name: value
          })));
        }
      } catch (err) {
        console.error('Error fetching topics:', err);
        setError('Failed to load topics');
      }
    };
    
    fetchTopics();
  }, [defaultTopic]);
  
  // Load subtopics when topic changes
  useEffect(() => {
    if (topic && topics[topic]) {
      setSubtopics(Object.entries(topics[topic].subtopics).map(([key, value]) => ({
        id: key,
        name: value
      })));
    } else {
      setSubtopics([]);
    }
  }, [topic, topics]);
  
  // Handle topic change
  const handleTopicChange = (e) => {
    const newTopic = e.target.value;
    setTopic(newTopic);
    setSubtopic(null); // Reset subtopic when topic changes
  };
  
  // Handle sending chat messages
  const handleSendMessage = async (e) => {
    e.preventDefault();
    
    if (!message.trim() || !topic || !grade || !board) {
      return;
    }
    
    try {
      setLoading(true);
      
      // Add user message to chat history
      setChatHistory(prev => [...prev, { role: 'user', content: message }]);
      
      // Send message to RAG system
      const response = await axios.post('/api/rag/chat', {
        message,
        topic,
        grade,
        board,
        subtopic
      });
      
      // Add response to chat history
      setChatHistory(prev => [...prev, { 
        role: 'assistant', 
        content: response.data.answer,
        metadata: response.data.content_metadata
      }]);
      
      // Clear message input
      setMessage('');
      
    } catch (err) {
      console.error('Error sending message:', err);
      setError('Failed to send message');
      
      // Add error message to chat history
      setChatHistory(prev => [...prev, { 
        role: 'assistant', 
        content: 'Sorry, I encountered an error. Please try again.',
        error: true
      }]);
    } finally {
      setLoading(false);
    }
  };
  
  // Generate a lesson
  const generateLesson = async () => {
    if (!topic || !grade || !board) {
      setError('Please select a topic, grade, and board');
      return;
    }
    
    try {
      setLoading(true);
      setError(null);
      
      const response = await axios.post('/api/rag/generate-lesson', {
        topic,
        grade,
        board,
        subtopic
      });
      
      setLessonScript(response.data);
    } catch (err) {
      console.error('Error generating lesson:', err);
      setError('Failed to generate lesson');
    } finally {
      setLoading(false);
    }
  };
  
  // Get adaptive content
  const getAdaptiveContent = async () => {
    if (!topic || !grade || !board) {
      setError('Please select a topic, grade, and board');
      return;
    }
    
    try {
      setLoading(true);
      setError(null);
      
      const response = await axios.post('/api/rag/adaptive-content', {
        topic,
        grade,
        board,
        subtopic,
        asLessonScript: true
      });
      
      setAdaptiveContent(response.data);
    } catch (err) {
      console.error('Error getting adaptive content:', err);
      setError('Failed to get adaptive content');
    } finally {
      setLoading(false);
    }
  };
  
  // Handle mode-specific actions
  useEffect(() => {
    if (topic && grade && board) {
      if (mode === 'lesson') {
        generateLesson();
      } else if (mode === 'adaptive') {
        getAdaptiveContent();
      }
    }
  }, [mode, topic, grade, board, subtopic]);
  
  // Render chat interface
  const renderChatInterface = () => (
    <div className="rag-chat-interface">
      <div className="chat-history">
        {chatHistory.length === 0 ? (
          <div className="chat-welcome">
            <p>Select a topic, grade, and board, then ask a question!</p>
          </div>
        ) : (
          chatHistory.map((msg, index) => (
            <div key={index} className={`chat-message ${msg.role}`}>
              <div className="message-content">
                {msg.content}
              </div>
              {msg.metadata && (
                <div className="message-metadata">
                  {msg.metadata.map((meta, idx) => (
                    <span key={idx} className="meta-tag">
                      {meta.content_type && 
                        <span className="content-type">{meta.content_type.replace(/_/g, ' ')}</span>
                      }
                      {meta.difficulty_level && 
                        <span className="difficulty">Level {meta.difficulty_level}</span>
                      }
                    </span>
                  ))}
                </div>
              )}
            </div>
          ))
        )}
      </div>
      
      <form className="chat-input-form" onSubmit={handleSendMessage}>
        <input
          type="text"
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          placeholder="Ask a question about the topic..."
          disabled={loading || !topic || !grade || !board}
        />
        <button 
          type="submit" 
          disabled={loading || !message.trim() || !topic || !grade || !board}
        >
          Send
        </button>
      </form>
    </div>
  );
  
  // Render lesson interface
  const renderLessonInterface = () => (
    <div className="rag-lesson-interface">
      {lessonScript ? (
        <LessonPlayer 
          lessonScript={lessonScript}
          onComplete={onLessonComplete}
        />
      ) : (
        <div className="lesson-placeholder">
          {loading ? (
            <p>Generating lesson...</p>
          ) : (
            <div>
              <p>Select a topic, grade, and board to generate a lesson.</p>
              <button 
                onClick={generateLesson}
                disabled={!topic || !grade || !board || loading}
              >
                Generate Lesson
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
  
  // Render adaptive content interface
  const renderAdaptiveInterface = () => (
    <div className="rag-adaptive-interface">
      {adaptiveContent ? (
        <LessonPlayer 
          lessonScript={adaptiveContent}
          onComplete={onLessonComplete}
        />
      ) : (
        <div className="adaptive-placeholder">
          {loading ? (
            <p>Finding adaptive content...</p>
          ) : (
            <div>
              <p>Select a topic, grade, and board to get adaptive content.</p>
              <button 
                onClick={getAdaptiveContent}
                disabled={!topic || !grade || !board || loading}
              >
                Get Adaptive Content
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
  
  return (
    <div className={`rag-integration ${className}`} style={style}>
      <div className="rag-controls">
        <div className="control-group">
          <label htmlFor="topic">Topic:</label>
          <select 
            id="topic"
            value={topic || ''}
            onChange={handleTopicChange}
            disabled={loading}
          >
            <option value="">Select a topic</option>
            {Object.entries(topics).map(([key, value]) => (
              <option key={key} value={key}>{value.name}</option>
            ))}
          </select>
        </div>
        
        <div className="control-group">
          <label htmlFor="grade">Grade:</label>
          <select 
            id="grade"
            value={grade || ''}
            onChange={(e) => setGrade(Number(e.target.value))}
            disabled={loading}
          >
            <option value="">Select grade</option>
            {[3, 4, 5, 6, 7, 8, 9, 10, 11, 12].map(g => (
              <option key={g} value={g}>Grade {g}</option>
            ))}
          </select>
        </div>
        
        <div className="control-group">
          <label htmlFor="board">Board:</label>
          <select 
            id="board"
            value={board || ''}
            onChange={(e) => setBoard(e.target.value)}
            disabled={loading}
          >
            <option value="">Select board</option>
            <option value="CBSE">CBSE</option>
            <option value="ICSE">ICSE</option>
            <option value="SSC">SSC</option>
          </select>
        </div>
        
        {subtopics.length > 0 && (
          <div className="control-group">
            <label htmlFor="subtopic">Subtopic:</label>
            <select 
              id="subtopic"
              value={subtopic || ''}
              onChange={(e) => setSubtopic(e.target.value)}
              disabled={loading}
            >
              <option value="">All subtopics</option>
              {subtopics.map(sub => (
                <option key={sub.id} value={sub.id}>{sub.name}</option>
              ))}
            </select>
          </div>
        )}
      </div>
      
      {error && (
        <div className="rag-error">{error}</div>
      )}
      
      <div className="rag-content">
        {mode === 'chat' && renderChatInterface()}
        {mode === 'lesson' && renderLessonInterface()}
        {mode === 'adaptive' && renderAdaptiveInterface()}
      </div>
    </div>
  );
};

export default RagIntegration;