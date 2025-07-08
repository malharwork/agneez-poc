const { generateLessonScript } = require('./integrations/rag_adapter');

async function testRagAdapter() {
  try {
    console.log('Testing RAG adapter...');
    
    // Set a timeout to prevent the test from hanging indefinitely
    const timeout = setTimeout(() => {
      console.error('Test timed out after 30 seconds');
      process.exit(1);
    }, 30000);
    
    // Test generating a lesson script
    const lessonScript = await generateLessonScript({
      topic: 'quadratic_equations',
      board: 'CBSE',
      grade: 9,
      subtopic: 'factorization_method'
    });
    
    clearTimeout(timeout);
    
    console.log('Lesson Script Generated:');
    console.log(JSON.stringify(lessonScript, null, 2));
    console.log('\nTest completed successfully!');
  } catch (error) {
    console.error('Error testing RAG adapter:', error.message);
    
    // Provide additional debugging information
    if (error.code === 'ECONNREFUSED') {
      console.error('\nConnection Error: Make sure the Python RAG service is running at http://localhost:5000');
      console.error('Run the Flask application with: python app.py');
    } else if (error.code === 'MODULE_NOT_FOUND') {
      console.error('\nModule Error: Make sure all required Node.js modules are installed');
      console.error('Run: npm install axios uuid');
    } else if (error.response) {
      // The request was made and the server responded with a status code outside the 2xx range
      console.error('\nServer Error:');
      console.error('Status:', error.response.status);
      console.error('Data:', JSON.stringify(error.response.data, null, 2));
    }
  }
}

// Run the test
testRagAdapter();