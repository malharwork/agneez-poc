const express = require('express');
const router = express.Router();
const { 
  generateLessonScript, 
  getLearningPath, 
  getAdaptiveContent,
  transformAdaptiveContentToLessonScript
} = require('../../integrations/rag_adapter');

/**
 * Generate a lesson script from the RAG system
 * @route POST /api/rag/generate-lesson
 * @param {string} topic - Topic name
 * @param {number} grade - Grade level
 * @param {string} board - Educational board
 * @param {string} [subtopic] - Optional subtopic
 * @param {string} [message] - Optional specific question/prompt
 */
router.post('/generate-lesson', async (req, res) => {
  try {
    const { topic, grade, board, subtopic, message, language } = req.body;
    
    // Validate required fields
    if (!topic || !grade || !board) {
      return res.status(400).json({ 
        error: 'Missing required fields: topic, grade, and board are required' 
      });
    }
    
    // Generate lesson script
    const lessonScript = await generateLessonScript({
      topic,
      board,
      grade,
      subtopic,
      message,
      language
    });
    
    // Store in database if needed
    // const savedLesson = await db.lessons.create({ lessonScript, topic, grade, board });
    
    return res.json(lessonScript);
  } catch (error) {
    console.error('Error generating lesson:', error);
    return res.status(500).json({ 
      error: 'Failed to generate lesson',
      details: error.message 
    });
  }
});

/**
 * Get a learning path from the RAG system
 * @route POST /api/rag/learning-path
 * @param {string} topic - Topic name
 * @param {number} grade - Grade level
 * @param {string} board - Educational board
 * @param {string} [currentSubtopic] - Current subtopic
 * @param {number} [masteryLevel] - Mastery level (0-1)
 */
router.post('/learning-path', async (req, res) => {
  try {
    const { topic, grade, board, currentSubtopic, masteryLevel } = req.body;
    
    // Validate required fields
    if (!topic || !grade || !board) {
      return res.status(400).json({ 
        error: 'Missing required fields: topic, grade, and board are required' 
      });
    }
    
    // Get learning path
    const learningPath = await getLearningPath({
      topic,
      board,
      grade,
      currentSubtopic,
      masteryLevel
    });
    
    return res.json(learningPath);
  } catch (error) {
    console.error('Error getting learning path:', error);
    return res.status(500).json({ 
      error: 'Failed to get learning path',
      details: error.message 
    });
  }
});

/**
 * Get adaptive content from the RAG system
 * @route POST /api/rag/adaptive-content
 * @param {string} topic - Topic name
 * @param {number} grade - Grade level
 * @param {string} board - Educational board
 * @param {string} [subtopic] - Optional subtopic
 */
router.post('/adaptive-content', async (req, res) => {
  try {
    const { topic, grade, board, subtopic, language, asLessonScript } = req.body;
    
    // Validate required fields
    if (!topic || !grade || !board) {
      return res.status(400).json({ 
        error: 'Missing required fields: topic, grade, and board are required' 
      });
    }
    
    // Get adaptive content
    const adaptiveContent = await getAdaptiveContent({
      topic,
      board,
      grade,
      subtopic,
      language
    });
    
    // Transform to lesson script if requested
    if (asLessonScript === true) {
      const lessonScript = transformAdaptiveContentToLessonScript(
        adaptiveContent,
        topic,
        grade,
        board
      );
      return res.json(lessonScript);
    }
    
    return res.json(adaptiveContent);
  } catch (error) {
    console.error('Error getting adaptive content:', error);
    return res.status(500).json({ 
      error: 'Failed to get adaptive content',
      details: error.message 
    });
  }
});

/**
 * Get a chat response from the RAG system
 * @route POST /api/rag/chat
 * @param {string} message - User message
 * @param {string} topic - Topic name
 * @param {number} grade - Grade level
 * @param {string} board - Educational board
 * @param {string} [subtopic] - Optional subtopic
 */
router.post('/chat', async (req, res) => {
  try {
    const { message, topic, grade, board, subtopic, language } = req.body;
    
    // Validate required fields
    if (!message || !topic || !grade || !board) {
      return res.status(400).json({ 
        error: 'Missing required fields: message, topic, grade, and board are required' 
      });
    }
    
    // Call the Python RAG service directly for chat
    const response = await axios.post(`${process.env.RAG_SERVICE_URL || 'http://localhost:5000'}/api/chat`, {
      message,
      topic,
      grade,
      board,
      subtopic,
      language
    });
    
    return res.json(response.data);
  } catch (error) {
    console.error('Error getting chat response:', error);
    return res.status(500).json({ 
      error: 'Failed to get chat response',
      details: error.message 
    });
  }
});

/**
 * Convert chat response to lesson script
 * @route POST /api/rag/chat-to-lesson
 * @param {string} message - User message
 * @param {string} topic - Topic name
 * @param {number} grade - Grade level
 * @param {string} board - Educational board
 * @param {string} [subtopic] - Optional subtopic
 */
router.post('/chat-to-lesson', async (req, res) => {
  try {
    const { message, topic, grade, board, subtopic, language } = req.body;
    
    // Validate required fields
    if (!message || !topic || !grade || !board) {
      return res.status(400).json({ 
        error: 'Missing required fields: message, topic, grade, and board are required' 
      });
    }
    
    // First get a chat response
    const chatResponse = await axios.post(`${process.env.RAG_SERVICE_URL || 'http://localhost:5000'}/api/chat`, {
      message,
      topic,
      grade,
      board,
      subtopic,
      language
    });
    
    // Transform the chat response to a lesson script
    const lessonScript = transformRagToLessonScript(chatResponse.data);
    
    return res.json(lessonScript);
  } catch (error) {
    console.error('Error converting chat to lesson:', error);
    return res.status(500).json({ 
      error: 'Failed to convert chat to lesson',
      details: error.message 
    });
  }
});

module.exports = router;