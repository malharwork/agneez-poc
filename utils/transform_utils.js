const { v4: uuidv4 } = require('uuid');

/**
 * Generate a unique event ID
 * @param {string} prefix - Prefix for the event ID
 * @returns {string} - Unique event ID
 */
const generateEventId = (prefix = 'event') => {
  return `${prefix}_${uuidv4().substring(0, 8)}`;
};

/**
 * Generate a unique element ID
 * @param {string} prefix - Prefix for the element ID
 * @returns {string} - Unique element ID
 */
const generateElementId = (prefix = 'elem') => {
  return `${prefix}_${uuidv4().substring(0, 8)}`;
};

/**
 * Split content into logical segments
 * @param {string} content - Content to split
 * @returns {Array<string>} - Array of content segments
 */
const splitContentIntoSegments = (content) => {
  // Basic splitting by paragraphs and removing empty entries
  const segments = content.split(/\n\n+/).filter(segment => segment.trim().length > 0);
  
  // Handle cases with very long paragraphs by splitting them further
  const result = [];
  
  for (const segment of segments) {
    // If segment is too long, split it further by sentences
    if (segment.length > 300) {
      const sentences = segment.match(/[^.!?]+[.!?]+/g) || [segment];
      let currentChunk = '';
      
      for (const sentence of sentences) {
        if (currentChunk.length + sentence.length > 300) {
          if (currentChunk) {
            result.push(currentChunk.trim());
          }
          currentChunk = sentence;
        } else {
          currentChunk += sentence;
        }
      }
      
      if (currentChunk) {
        result.push(currentChunk.trim());
      }
    } else {
      result.push(segment.trim());
    }
  }
  
  return result;
};

/**
 * Convert plain text to SSML format
 * @param {string} text - Plain text content
 * @returns {string} - SSML formatted content
 */
const convertToSSML = (text) => {
  // Replace newlines with breaks
  let ssml = text.replace(/\n/g, '<break time="500ms"/> ');
  
  // Add breaks after punctuation
  ssml = ssml.replace(/([.!?])\s+/g, '$1<break time="200ms"/> ');
  
  // Add breaks after commas
  ssml = ssml.replace(/([,;:])\s+/g, '$1<break time="100ms"/> ');
  
  // Wrap in SSML tags
  return `<speak><prosody rate='slow'>${ssml}</prosody></speak>`;
};

/**
 * Detect if a segment contains an equation
 * @param {string} segment - Content segment
 * @returns {boolean} - True if segment contains an equation
 */
const containsEquation = (segment) => {
  const equationPatterns = [
    /[a-z]\^[0-9]/i,      // Example: x^2
    /\\frac{.*}{.*}/,     // Example: \frac{a}{b}
    /\([a-z0-9+-\\]+\)/,  // Example: (x+2)
    /[a-z][0-9]=[0-9]/i,  // Example: x2=4
    /[a-z]²/i,            // Example: x²
    /[a-z]\s*[+\-*/]\s*[a-z0-9]/i, // Example: a + b
    /[=<>≤≥]/,            // Example: a = b
    /\b\d+[a-z]\b/i       // Example: 2x
  ];
  
  return equationPatterns.some(pattern => pattern.test(segment));
};

/**
 * Detect if a segment contains a list or steps
 * @param {string} segment - Content segment
 * @returns {boolean} - True if segment contains a list
 */
const containsList = (segment) => {
  // Check for numbered lists or bullet points
  return /^\s*(\d+\.|\*|\-)\s+/.test(segment) || 
         segment.includes('Step 1') || 
         segment.includes('First') || 
         (segment.includes('steps') && segment.length < 150);
};

/**
 * Detect if a segment is a question
 * @param {string} segment - Content segment
 * @returns {boolean} - True if segment is a question
 */
const isQuestion = (segment) => {
  return segment.trim().endsWith('?') || 
         segment.includes('calculate') || 
         segment.includes('Find the') || 
         segment.includes('Solve') || 
         segment.includes('What is') ||
         /try\s+this/i.test(segment);
};

/**
 * Extract a potential MCQ from a segment
 * @param {string} segment - Content segment
 * @returns {Object|null} - MCQ object or null if not an MCQ
 */
const extractMCQ = (segment) => {
  // Check if segment contains MCQ patterns
  const hasOptions = segment.includes('A)') || 
                     segment.includes('a)') || 
                     segment.includes('option') || 
                     segment.includes('Options');
  
  if (!hasOptions || !isQuestion(segment)) {
    return null;
  }
  
  // Extract question and options
  const lines = segment.split('\n');
  let question = lines[0].trim();
  
  // Extract options
  const options = {};
  const optionRegex = /\(?([A-Da-d])\)?[\s\.:\)]+(.+)/;
  let correctOptions = [];
  
  // Default to first option as correct if we can't determine
  correctOptions.push('opt1');
  
  for (let i = 1; i < lines.length; i++) {
    const match = lines[i].match(optionRegex);
    if (match) {
      const optionKey = 'opt' + i;
      const optionText = match[2].trim();
      options[optionKey] = optionText;
      
      // Try to guess correct option (if it contains "correct" or similar)
      if (optionText.toLowerCase().includes('correct') || 
          lines[i].toLowerCase().includes('correct') || 
          optionText.includes('✓')) {
        correctOptions = [optionKey];
      }
    }
  }
  
  if (Object.keys(options).length < 2) {
    return null;
  }
  
  return {
    question,
    options,
    correctOptions,
    questionType: "SINGLE" // Default to single choice
  };
};

/**
 * Convert plain text equations to KaTeX format
 * @param {string} text - Plain text equation
 * @returns {string} - KaTeX formatted equation
 */
const convertToKatex = (text) => {
  // Basic conversions from plain text to KaTeX
  let katex = text;
  
  // Square/cube
  katex = katex.replace(/([a-z0-9])²/gi, '$1^2');
  katex = katex.replace(/([a-z0-9])³/gi, '$1^3');
  
  // Fractions
  katex = katex.replace(/(\d+)\/(\d+)/g, '\\frac{$1}{$2}');
  
  // Square root
  katex = katex.replace(/√(\w+)/g, '\\sqrt{$1}');
  katex = katex.replace(/sqrt\(([^)]+)\)/g, '\\sqrt{$1}');
  
  // Quadratic formula specifically
  if (katex.includes('±') && katex.includes('sqrt')) {
    katex = katex.replace(/(\w+)\s*=\s*\((-?\w+)\s*±\s*sqrt\(([^)]+)\)\)\s*\/\s*([^)]+)/g, 
                         '$1 = \\frac{$2 \\pm \\sqrt{$3}}{$4}');
  }
  
  return katex;
};

/**
 * Create an equation element for the whiteboard
 * @param {string} equation - Equation text
 * @param {string} elementId - Element ID
 * @returns {Object} - Equation element
 */
const createEquationElement = (equation, elementId) => {
  return {
    type: "EQUATION",
    elementId,
    katexContent: convertToKatex(equation),
    style: {
      fontSize: 24,
      width: 400
    },
    animation: [
      {
        type: "FADE_IN",
        duration: 1000
      }
    ]
  };
};

/**
 * Create a text element for the whiteboard
 * @param {string} content - Text content
 * @param {string} elementId - Element ID
 * @param {Object} style - Style object
 * @returns {Object} - Text element
 */
const createTextElement = (content, elementId, style = {}) => {
  return {
    type: "TEXT",
    elementId,
    content,
    style: {
      fontSize: 20,
      width: 400,
      ...style
    },
    animation: [
      {
        type: "FADE_IN",
        duration: 1000
      }
    ]
  };
};

/**
 * Create an MCQ element for the whiteboard
 * @param {Object} mcqData - MCQ data
 * @param {string} elementId - Element ID
 * @returns {Object} - MCQ element
 */
const createMCQElement = (mcqData, elementId) => {
  return {
    type: "MCQ",
    elementId,
    questionType: mcqData.questionType,
    question: mcqData.question,
    options: mcqData.options,
    correctOptions: mcqData.correctOptions,
    style: {
      width: 400
    }
  };
};

/**
 * Generate appropriate avatar gesture based on content
 * @param {string} content - Content
 * @returns {string} - Avatar gesture
 */
const getAvatarGesture = (content) => {
  if (content.includes('example') || content.includes('Example')) {
    return "POINT";
  } else if (content.includes('?')) {
    return "THINK";
  } else if (content.includes('Welcome') || content.includes('Great job')) {
    return "WELCOME";
  } else {
    return "EXPLAIN";
  }
};

/**
 * Create a TEACH event
 * @param {Object} params - Parameters
 * @returns {Object} - TEACH event
 */
const createTeachEvent = ({ content, nextEventId, elements = [] }) => {
  const gesture = getAvatarGesture(content);
  
  return {
    type: "TEACH",
    next: nextEventId,
    avatar: {
      gesture
    },
    speech: {
      content: convertToSSML(content)
    },
    whiteboard: {
      elements
    }
  };
};

/**
 * Create an INTERACT event
 * @param {Object} params - Parameters
 * @returns {Object} - INTERACT event
 */
const createInteractEvent = ({ content, nextEventId, mcqElement }) => {
  return {
    type: "INTERACT",
    next: nextEventId,
    avatar: {
      gesture: "THINK"
    },
    speech: {
      content: convertToSSML(content)
    },
    whiteboard: {
      elements: [mcqElement]
    }
  };
};

/**
 * Create a CHOICE event
 * @param {Object} params - Parameters
 * @returns {Object} - CHOICE event
 */
const createChoiceEvent = ({ mcqElementId, correctNextEventId, incorrectNextEventId }) => {
  return {
    type: "CHOICE",
    choices: [
      {
        condition: `$includes($${mcqElementId}, '${mcqElementId}_opt1')`,
        value: true,
        nextEvent: correctNextEventId
      },
      {
        condition: "true",
        value: true,
        nextEvent: incorrectNextEventId
      }
    ]
  };
};

/**
 * Create a WAIT event
 * @param {Object} params - Parameters
 * @returns {Object} - WAIT event
 */
const createWaitEvent = ({ waitTime, nextEventId }) => {
  return {
    type: "WAIT",
    waitTime,
    next: nextEventId
  };
};

/**
 * Create a lesson title from metadata
 * @param {Object} metadata - Metadata
 * @returns {string} - Lesson title
 */
const createLessonTitle = (metadata) => {
  const { topic, subtopic, grade, board } = metadata;
  
  let title = topic.replace(/_/g, ' ');
  title = title.split(' ').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
  
  if (subtopic) {
    const formattedSubtopic = subtopic.replace(/_/g, ' ');
    title = `${title}: ${formattedSubtopic.charAt(0).toUpperCase() + formattedSubtopic.slice(1)}`;
  }
  
  return `${title} - Grade ${grade} ${board}`;
};

/**
 * Transform RAG response to Lesson Script
 * @param {Object} ragResponse - RAG response
 * @returns {Object} - Lesson Script
 */
const transformRagToLessonScript = (ragResponse) => {
  const { answer, content_metadata = [], filter_applied = {}, topic } = ragResponse;
  
  // Extract metadata
  const metadata = {
    topic: topic || 'unknown_topic',
    subtopic: content_metadata[0]?.subtopic || '',
    grade: filter_applied.grade || 9,
    board: filter_applied.board || 'CBSE'
  };
  
  // Create lesson title
  const title = createLessonTitle(metadata);
  
  // Split content into segments
  const segments = splitContentIntoSegments(answer);
  
  // Initialize lesson script
  const lessonScript = {
    title,
    startEvent: "intro",
    lessonEvents: {}
  };
  
  // Create intro event
  const introEventId = "intro";
  lessonScript.lessonEvents[introEventId] = createTeachEvent({
    content: `Welcome to today's lesson on ${metadata.topic.replace(/_/g, ' ')}!`,
    nextEventId: "content1",
    elements: [
      createTextElement(title, "title", {
        fontSize: 32,
        fontWeight: "700",
        textAlign: "center"
      })
    ]
  });
  
  // Process segments
  let currentEventId = "content1";
  let eventIndex = 1;
  
  for (let i = 0; i < segments.length; i++) {
    const segment = segments[i];
    const isLastSegment = i === segments.length - 1;
    const nextEventId = isLastSegment ? "END" : `content${eventIndex + 1}`;
    
    // Check if segment contains an equation
    if (containsEquation(segment)) {
      // Extract equation from the segment
      const equationId = generateElementId("equation");
      lessonScript.lessonEvents[currentEventId] = createTeachEvent({
        content: segment,
        nextEventId,
        elements: [createEquationElement(segment, equationId)]
      });
    }
    // Check if segment is an MCQ
    else if (isQuestion(segment)) {
      const mcqData = extractMCQ(segment);
      
      if (mcqData) {
        const mcqId = generateElementId("mcq");
        const mcqElement = createMCQElement(mcqData, mcqId);
        
        // Create interact event
        lessonScript.lessonEvents[currentEventId] = createInteractEvent({
          content: segment,
          nextEventId: `choice${eventIndex}`,
          mcqElement
        });
        
        // Create choice event
        const choiceEventId = `choice${eventIndex}`;
        const correctEventId = `correct${eventIndex}`;
        const incorrectEventId = `incorrect${eventIndex}`;
        
        lessonScript.lessonEvents[choiceEventId] = createChoiceEvent({
          mcqElementId: mcqId,
          correctNextEventId: correctEventId,
          incorrectNextEventId: incorrectEventId
        });
        
        // Create correct response event
        lessonScript.lessonEvents[correctEventId] = createTeachEvent({
          content: "Great job! That's correct!",
          nextEventId: isLastSegment ? "END" : `content${eventIndex + 1}`,
          elements: []
        });
        
        // Create incorrect response event
        lessonScript.lessonEvents[incorrectEventId] = createTeachEvent({
          content: `The correct answer is ${mcqData.correctOptions.map(opt => mcqData.options[opt]).join(', ')}.`,
          nextEventId: isLastSegment ? "END" : `content${eventIndex + 1}`,
          elements: []
        });
        
        eventIndex += 3; // We added 3 events
        currentEventId = isLastSegment ? "END" : `content${eventIndex}`;
        continue;
      }
    }
    // Regular text content
    else {
      const textId = generateElementId("text");
      lessonScript.lessonEvents[currentEventId] = createTeachEvent({
        content: segment,
        nextEventId,
        elements: [createTextElement(segment, textId)]
      });
    }
    
    eventIndex++;
    currentEventId = nextEventId;
  }
  
  return lessonScript;
};

module.exports = {
  transformRagToLessonScript,
  generateEventId,
  generateElementId,
  splitContentIntoSegments,
  convertToSSML,
  containsEquation,
  containsList,
  isQuestion,
  extractMCQ,
  convertToKatex,
  createEquationElement,
  createTextElement,
  createMCQElement,
  getAvatarGesture,
  createTeachEvent,
  createInteractEvent,
  createChoiceEvent,
  createWaitEvent,
  createLessonTitle
};