import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';

// Avatar component for rendering different gestures
const Avatar = ({ gesture }) => {
  // Map gestures to corresponding avatar images/animations
  const gestureMap = {
    WELCOME: '/assets/avatars/welcome.svg',
    EXPLAIN: '/assets/avatars/explain.svg',
    POINT: '/assets/avatars/point.svg',
    THINK: '/assets/avatars/think.svg'
  };

  return (
    <div className="avatar-container">
      <img 
        src={gestureMap[gesture] || gestureMap.EXPLAIN} 
        alt={`Avatar with ${gesture} gesture`} 
        className="avatar-image"
      />
    </div>
  );
};

// Whiteboard element renderer
const WhiteboardElement = ({ element }) => {
  switch(element.type) {
    case 'TEXT':
      return (
        <div 
          id={element.elementId}
          className="whiteboard-text"
          style={{
            fontSize: `${element.style?.fontSize || 16}px`,
            fontWeight: element.style?.fontWeight || 'normal',
            textAlign: element.style?.textAlign || 'left',
            width: `${element.style?.width || 300}px`,
            opacity: element.animation ? 0 : 1, // Start with 0 opacity if has animation
            animation: element.animation ? 
              `${element.animation[0].type.toLowerCase()} ${element.animation[0].duration}ms forwards` : 
              'none'
          }}
          dangerouslySetInnerHTML={{ __html: element.content }}
        />
      );
    
    case 'EQUATION':
      return (
        <div 
          id={element.elementId}
          className="whiteboard-equation"
          style={{
            fontSize: `${element.style?.fontSize || 16}px`,
            width: `${element.style?.width || 300}px`,
            opacity: element.animation ? 0 : 1,
            animation: element.animation ? 
              `${element.animation[0].type.toLowerCase()} ${element.animation[0].duration}ms forwards` : 
              'none'
          }}
        >
          <div className="katex-container" dangerouslySetInnerHTML={{ 
            __html: `\\(${element.katexContent}\\)` 
          }} />
        </div>
      );
    
    case 'MCQ':
      return (
        <div 
          id={element.elementId}
          className="whiteboard-mcq"
          style={{
            width: `${element.style?.width || 300}px`
          }}
        >
          <div className="mcq-question">{element.question}</div>
          <div className="mcq-options">
            {Object.entries(element.options).map(([key, value]) => (
              <div key={key} className="mcq-option">
                <input 
                  type={element.questionType === 'SINGLE' ? 'radio' : 'checkbox'}
                  id={`${element.elementId}_${key}`}
                  name={element.elementId}
                  value={key}
                />
                <label htmlFor={`${element.elementId}_${key}`}>{value}</label>
              </div>
            ))}
          </div>
        </div>
      );
    
    case 'TABLE2':
      return (
        <div 
          id={element.elementId}
          className="whiteboard-table"
          style={{
            width: `${element.style?.width || 300}px`
          }}
        >
          <table className={element.renderBorder ? 'bordered-table' : ''}>
            {element.header.length > 0 && (
              <thead>
                <tr>
                  {element.header.map((headerCell, index) => (
                    <th 
                      key={index}
                      style={{
                        fontSize: `${element.headerStyle?.fontSize || 16}px`,
                        fontWeight: element.headerStyle?.fontWeight || 'bold'
                      }}
                    >
                      {headerCell}
                    </th>
                  ))}
                </tr>
              </thead>
            )}
            <tbody>
              {element.content.map((row, rowIndex) => (
                <tr key={rowIndex}>
                  {row.map((cell, colIndex) => (
                    <td 
                      key={colIndex}
                      style={{
                        fontSize: `${element.contentStyle?.fontSize || 14}px`,
                        fontWeight: element.contentStyle?.fontWeight || 'normal',
                        backgroundColor: (rowIndex === element.rowIndex && colIndex === element.columnIndex) ? 
                          '#e6f7ff' : 'transparent'
                      }}
                      dangerouslySetInnerHTML={{ __html: cell }}
                    />
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      );
    
    // Add more element types as needed
    
    default:
      return <div>Unsupported element type: {element.type}</div>;
  }
};

// Main lesson player component
const LessonPlayer = ({ lessonId, topicName, grade, board, subtopic }) => {
  const [lessonScript, setLessonScript] = useState(null);
  const [currentEventId, setCurrentEventId] = useState(null);
  const [currentEvent, setCurrentEvent] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [userResponses, setUserResponses] = useState({});
  const audioRef = useRef(null);
  
  // Fetch lesson script on component mount
  useEffect(() => {
    const fetchLessonScript = async () => {
      try {
        setLoading(true);
        
        // If lessonId is provided, fetch the specific lesson
        if (lessonId) {
          const response = await axios.get(`/api/lessons/${lessonId}`);
          setLessonScript(response.data);
        } 
        // Otherwise generate a new lesson using the RAG system
        else if (topicName && grade && board) {
          const response = await axios.post('/api/rag/generate-lesson', {
            topic: topicName,
            grade,
            board,
            subtopic
          });
          setLessonScript(response.data);
        } else {
          throw new Error('Either lessonId or topic details must be provided');
        }
        
        setError(null);
      } catch (err) {
        console.error('Error fetching lesson:', err);
        setError(err.message || 'Failed to load lesson');
      } finally {
        setLoading(false);
      }
    };
    
    fetchLessonScript();
  }, [lessonId, topicName, grade, board, subtopic]);
  
  // Set initial event when lesson script is loaded
  useEffect(() => {
    if (lessonScript && lessonScript.startEvent) {
      setCurrentEventId(lessonScript.startEvent);
    }
  }, [lessonScript]);
  
  // Update current event when event ID changes
  useEffect(() => {
    if (lessonScript && currentEventId && lessonScript.lessonEvents[currentEventId]) {
      setCurrentEvent(lessonScript.lessonEvents[currentEventId]);
    } else if (currentEventId === 'END') {
      setCurrentEvent(null);
    }
  }, [lessonScript, currentEventId]);
  
  // Play speech when current event changes
  useEffect(() => {
    if (currentEvent && currentEvent.speech?.content) {
      // Convert SSML to audio using text-to-speech service
      const playSpeech = async () => {
        try {
          // Remove this in production and use your actual TTS service
          console.log('Playing speech:', currentEvent.speech.content);
          
          // Simulate speech duration for demonstration
          await new Promise(resolve => setTimeout(resolve, 2000));
          
          // Proceed to next event if there is one
          if (currentEvent.next && currentEvent.next !== 'END') {
            setCurrentEventId(currentEvent.next);
          } else if (currentEvent.next === 'END') {
            setCurrentEventId('END');
          }
        } catch (err) {
          console.error('Error playing speech:', err);
        }
      };
      
      playSpeech();
    }
  }, [currentEvent]);
  
  // Handle MCQ responses
  const handleMCQResponse = (mcqId, optionKey, isMultiple) => {
    if (isMultiple) {
      // For multiple choice, toggle the selection
      setUserResponses(prev => {
        const currentSelections = prev[mcqId] || [];
        if (currentSelections.includes(optionKey)) {
          return {
            ...prev,
            [mcqId]: currentSelections.filter(key => key !== optionKey)
          };
        } else {
          return {
            ...prev,
            [mcqId]: [...currentSelections, optionKey]
          };
        }
      });
    } else {
      // For single choice, replace the selection
      setUserResponses(prev => ({
        ...prev,
        [mcqId]: [optionKey]
      }));
      
      // Process CHOICE event if current event is INTERACT
      if (currentEvent.type === 'INTERACT' && currentEvent.next) {
        // Get the CHOICE event
        const choiceEventId = currentEvent.next;
        const choiceEvent = lessonScript.lessonEvents[choiceEventId];
        
        if (choiceEvent && choiceEvent.type === 'CHOICE') {
          // Find the matching choice
          const matchingChoice = choiceEvent.choices.find(choice => {
            // Simple condition evaluation (this would need a proper JSONata evaluator in production)
            if (choice.condition.includes(`'${optionKey}'`)) {
              return true;
            }
            return false;
          });
          
          if (matchingChoice) {
            setCurrentEventId(matchingChoice.nextEvent);
          } else if (choiceEvent.choices.length > 0) {
            // Default to the last choice (usually the catch-all)
            setCurrentEventId(choiceEvent.choices[choiceEvent.choices.length - 1].nextEvent);
          }
        }
      }
    }
  };
  
  if (loading) {
    return <div className="lesson-loading">Loading lesson...</div>;
  }
  
  if (error) {
    return <div className="lesson-error">Error: {error}</div>;
  }
  
  if (!lessonScript) {
    return <div className="lesson-error">No lesson data available</div>;
  }
  
  if (currentEventId === 'END') {
    return (
      <div className="lesson-complete">
        <h2>Lesson Complete!</h2>
        <p>You have completed the lesson on {lessonScript.title}.</p>
        <button onClick={() => setCurrentEventId(lessonScript.startEvent)}>
          Restart Lesson
        </button>
      </div>
    );
  }
  
  if (!currentEvent) {
    return <div className="lesson-loading">Preparing lesson content...</div>;
  }
  
  return (
    <div className="lesson-player">
      <div className="lesson-header">
        <h1>{lessonScript.title}</h1>
      </div>
      
      <div className="lesson-content">
        {/* Avatar area */}
        <div className="avatar-area">
          {currentEvent.avatar && (
            <Avatar gesture={currentEvent.avatar.gesture} />
          )}
        </div>
        
        {/* Whiteboard area */}
        <div className="whiteboard-area">
          {currentEvent.whiteboard && currentEvent.whiteboard.elements && (
            <div className="whiteboard-elements">
              {currentEvent.whiteboard.elements.map((element) => (
                <WhiteboardElement 
                  key={element.elementId} 
                  element={element} 
                />
              ))}
            </div>
          )}
          
          {/* Interactive elements */}
          {currentEvent.type === 'INTERACT' && (
            <div className="interactive-controls">
              {currentEvent.whiteboard?.elements.map(element => {
                if (element.type === 'MCQ') {
                  return (
                    <div key={element.elementId} className="mcq-controls">
                      <button 
                        onClick={() => {
                          const selectedOptions = userResponses[element.elementId] || [];
                          if (selectedOptions.length > 0) {
                            // Proceed to next event
                            if (currentEvent.next) {
                              setCurrentEventId(currentEvent.next);
                            }
                          } else {
                            alert('Please select an option');
                          }
                        }}
                        className="mcq-submit-btn"
                      >
                        Submit Answer
                      </button>
                    </div>
                  );
                }
                return null;
              })}
            </div>
          )}
        </div>
      </div>
      
      {/* Navigation controls */}
      <div className="lesson-controls">
        <button 
          onClick={() => {
            if (currentEvent.next && currentEvent.type !== 'INTERACT') {
              setCurrentEventId(currentEvent.next);
            }
          }}
          disabled={currentEvent.type === 'INTERACT'}
        >
          Next
        </button>
      </div>
      
      {/* Hidden audio element for speech */}
      <audio ref={audioRef} style={{ display: 'none' }} />
    </div>
  );
};

export default LessonPlayer;