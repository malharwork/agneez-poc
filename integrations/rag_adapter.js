/**
 * RAG Adapter - Integrates with the Python-based RAG system and transforms responses
 * into the lesson script format required by the frontend.
 */

const axios = require('axios');
const { transformRagToLessonScript } = require('../utils/transform_utils');

// Configure RAG service endpoint
const RAG_SERVICE_URL = process.env.RAG_SERVICE_URL || 'http://localhost:5000';

/**
 * Generate a lesson script for a given topic
 * @param {Object} options - Options for lesson generation
 * @param {string} options.topic - The topic name
 * @param {string} options.board - The educational board (CBSE, ICSE, SSC)
 * @param {number} options.grade - The grade level
 * @param {string} [options.subtopic] - Optional subtopic
 * @param {string} [options.message] - Optional specific message/question
 * @param {string} [options.language='english'] - The language
 * @param {Array<string>} [options.methodPreference] - Optional method preferences
 * @returns {Promise<Object>} - Lesson script in the required format
 */
async function generateLessonScript(options) {
  try {
    const {
      topic,
      board,
      grade,
      subtopic,
      message = '',
      language = 'english',
      methodPreference = []
    } = options;
    
    // Validate required parameters
    if (!topic || !board || !grade) {
      throw new Error('Missing required parameters: topic, board, and grade are required');
    }
    
    // Call the RAG service
    const response = await axios.post(`${RAG_SERVICE_URL}/api/lesson`, {
      topic,
      board,
      grade,
      subtopic,
      message,
      language,
      method_preference: methodPreference.length > 0 ? methodPreference : undefined
    });
    
    // Get the RAG response data
    const ragResponse = response.data;
    
    // Check for errors or inappropriate grade level
    if (ragResponse.error) {
      throw new Error(ragResponse.error);
    }
    
    if (ragResponse.grade_appropriate === false) {
      // Return a simple lesson script explaining why the topic is not appropriate
      return createNotAppropriateLesson(ragResponse);
    }
    
    // Transform the RAG response to a lesson script
    const lessonScript = transformRagToLessonScript(ragResponse);
    
    // Add chapter and subject information
    lessonScript.chapter = ragResponse.chapter || 'General Knowledge';
    lessonScript.subject = ragResponse.subject || 'General';
    
    return lessonScript;
  } catch (error) {
    console.error('Error generating lesson script:', error.message);
    throw error;
  }
}

/**
 * Create a lesson script for content that's not grade-appropriate
 * @param {Object} ragResponse - The RAG response
 * @returns {Object} - Lesson script explaining why the content is not appropriate
 */
function createNotAppropriateLesson(ragResponse) {
  const topic = ragResponse.recommended_grade ? 
    `This topic is typically taught in grade ${ragResponse.recommended_grade} and above` : 
    'This topic may not be appropriate for your current grade level';
  
  // Create a simple lesson script
  return {
    title: `About ${ragResponse.current_grade ? `Grade ${ragResponse.current_grade}` : ''} Topics`,
    startEvent: "intro",
    lessonEvents: {
      "intro": {
        type: "TEACH",
        next: "explanation",
        avatar: {
          gesture: "WELCOME"
        },
        speech: {
          content: "<speak><prosody rate='slow'>Hello there!<break time='200ms'/> I'd like to help you understand which topics are appropriate for your grade level.</prosody></speak>"
        },
        whiteboard: {
          elements: [
            {
              type: "TEXT",
              elementId: "title",
              content: "Learning at the Right Level",
              style: {
                fontSize: 32,
                fontWeight: "700",
                textAlign: "center",
                width: 400
              },
              animation: [
                {
                  type: "FADE_IN",
                  duration: 1000
                }
              ]
            }
          ]
        }
      },
      "explanation": {
        type: "TEACH",
        next: "suggestions",
        avatar: {
          gesture: "EXPLAIN"
        },
        speech: {
          content: `<speak><prosody rate='slow'>${ragResponse.answer.replace(/\n/g, '<break time="200ms"/> ')}</prosody></speak>`
        },
        whiteboard: {
          elements: [
            {
              type: "TEXT",
              elementId: "explanation",
              content: ragResponse.answer,
              style: {
                fontSize: 20,
                width: 400
              },
              animation: [
                {
                  type: "FADE_IN",
                  duration: 1000
                }
              ]
            }
          ]
        }
      },
      "suggestions": {
        type: "TEACH",
        next: "END",
        avatar: {
          gesture: "WELCOME"
        },
        speech: {
          content: "<speak><prosody rate='slow'>Would you like to explore some topics that are perfect for your grade level instead?<break time='200ms'/> I'd be happy to suggest some interesting ones!</prosody></speak>"
        },
        whiteboard: {
          elements: [
            {
              type: "TEXT",
              elementId: "suggestions",
              content: "Here are some topics that would be great for your current grade level:<br><br>• Basic arithmetic and pre-algebra<br>• Introduction to plants and animals<br>• Earth and space science<br>• History and geography",
              style: {
                fontSize: 20,
                width: 400
              },
              animation: [
                {
                  type: "FADE_IN",
                  duration: 1000
                }
              ]
            }
          ]
        }
      }
    }
  };
}

/**
 * Get learning path from the RAG system
 * @param {Object} options - Options for learning path
 * @param {string} options.topic - The topic name
 * @param {string} options.board - The educational board (CBSE, ICSE, SSC)
 * @param {number} options.grade - The grade level
 * @param {string} [options.currentSubtopic] - Current subtopic
 * @param {number} [options.masteryLevel=0.5] - Mastery level (0-1)
 * @returns {Promise<Object>} - Learning path data
 */
async function getLearningPath(options) {
  try {
    const {
      topic,
      board,
      grade,
      currentSubtopic,
      masteryLevel = 0.5
    } = options;
    
    // Validate required parameters
    if (!topic || !board || !grade) {
      throw new Error('Missing required parameters: topic, board, and grade are required');
    }
    
    // Call the RAG service
    const response = await axios.post(`${RAG_SERVICE_URL}/api/learning-path`, {
      topic,
      grade,
      board,
      current_subtopic: currentSubtopic,
      mastery_level: masteryLevel
    });
    
    return response.data;
  } catch (error) {
    console.error('Error getting learning path:', error.message);
    throw error;
  }
}

/**
 * Get adaptive content from the RAG system
 * @param {Object} options - Options for adaptive content
 * @param {string} options.topic - The topic name
 * @param {string} options.board - The educational board (CBSE, ICSE, SSC)
 * @param {number} options.grade - The grade level
 * @param {string} [options.subtopic] - Optional subtopic
 * @param {string} [options.language='english'] - The language
 * @returns {Promise<Object>} - Adaptive content data
 */
async function getAdaptiveContent(options) {
  try {
    const {
      topic,
      board,
      grade,
      subtopic,
      language = 'english'
    } = options;
    
    // Validate required parameters
    if (!topic || !board || !grade) {
      throw new Error('Missing required parameters: topic, board, and grade are required');
    }
    
    // Call the RAG service
    const response = await axios.post(`${RAG_SERVICE_URL}/api/adaptive-content`, {
      topic,
      grade,
      board,
      subtopic,
      language
    });
    
    return response.data;
  } catch (error) {
    console.error('Error getting adaptive content:', error.message);
    throw error;
  }
}

/**
 * Transform adaptive content to lesson script format
 * @param {Object} adaptiveContent - Adaptive content from RAG
 * @param {string} topic - The topic name
 * @param {number} grade - The grade level
 * @param {string} board - The educational board
 * @returns {Object} - Lesson script for adaptive content
 */
function transformAdaptiveContentToLessonScript(adaptiveContent, topic, grade, board) {
  // Create a lesson script for adaptive content
  const lessonScript = {
    title: `Adaptive Content: ${topic.replace(/_/g, ' ')} - Grade ${grade} ${board}`,
    startEvent: "intro",
    lessonEvents: {
      "intro": {
        type: "TEACH",
        next: "content1",
        avatar: {
          gesture: "WELCOME"
        },
        speech: {
          content: "<speak><prosody rate='slow'>I've selected some content that's perfect for your current level.<break time='200ms'/> Let's explore these materials to help you learn more effectively.</prosody></speak>"
        },
        whiteboard: {
          elements: [
            {
              type: "TEXT",
              elementId: "title",
              content: `Adaptive Learning: ${topic.replace(/_/g, ' ')}`,
              style: {
                fontSize: 32,
                fontWeight: "700",
                textAlign: "center",
                width: 400
              },
              animation: [
                {
                  type: "FADE_IN",
                  duration: 1000
                }
              ]
            }
          ]
        }
      }
    }
  };
  
  // Process adaptive content items
  if (adaptiveContent.adaptive_content && adaptiveContent.adaptive_content.length > 0) {
    let currentEventId = "content1";
    
    adaptiveContent.adaptive_content.forEach((content, index) => {
      const nextEventId = index === adaptiveContent.adaptive_content.length - 1 ? "END" : `content${index + 2}`;
      const contentText = content.text || "No content available";
      
      lessonScript.lessonEvents[currentEventId] = {
        type: "TEACH",
        next: nextEventId,
        avatar: {
          gesture: "EXPLAIN"
        },
        speech: {
          content: `<speak><prosody rate='slow'>Here's a ${content.content_type?.replace(/_/g, ' ') || 'resource'} about ${content.subtopic?.replace(/_/g, ' ') || topic.replace(/_/g, ' ')}.<break time='200ms'/> ${contentText.substring(0, 100).replace(/\n/g, '<break time="200ms"/> ')}</prosody></speak>`
        },
        whiteboard: {
          elements: [
            {
              type: "TEXT",
              elementId: `content_title_${index}`,
              content: `${content.content_type?.replace(/_/g, ' ') || 'Resource'}: ${content.subtopic?.replace(/_/g, ' ') || topic.replace(/_/g, ' ')}`,
              style: {
                fontSize: 24,
                fontWeight: "600",
                width: 400
              }
            },
            {
              type: "TEXT",
              elementId: `content_text_${index}`,
              content: contentText,
              style: {
                fontSize: 18,
                width: 400
              }
            }
          ]
        }
      };
      
      currentEventId = nextEventId;
    });
  } else {
    lessonScript.lessonEvents["content1"] = {
      type: "TEACH",
      next: "END",
      avatar: {
        gesture: "THINK"
      },
      speech: {
        content: "<speak><prosody rate='slow'>I couldn't find specific adaptive content for your current level.<break time='200ms'/> Let's explore the topic more broadly instead.</prosody></speak>"
      },
      whiteboard: {
        elements: [
          {
            type: "TEXT",
            elementId: "no_content",
            content: "No specific adaptive content found. Let's start with the basics of the topic.",
            style: {
              fontSize: 20,
              width: 400
            }
          }
        ]
      }
    };
  }
  
  return lessonScript;
}

module.exports = {
  generateLessonScript,
  getLearningPath,
  getAdaptiveContent,
  transformAdaptiveContentToLessonScript
};