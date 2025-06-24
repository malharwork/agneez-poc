import os
import logging
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from utils.pdf_loader import get_pdf_info, get_chapters_from_contents
from utils.embeddings import EmbeddingModel
from utils.vectorstore import VectorStore
from utils.rag import RAG

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

PDF_PATH = os.getenv('PDF_PATH', 'data/textbook.pdf')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')
PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME', 'textbook-index')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')

embedding_model = EmbeddingModel()
vector_store = VectorStore(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENVIRONMENT,
    index_name=PINECONE_INDEX_NAME
)
rag = RAG(ANTHROPIC_API_KEY, vector_store, embedding_model)

import utils.pdf_loader as pdf_loader

def initialize_data():
    """Initialize the application data."""
    try:
        # Check if PDF has been processed
        namespaces = vector_store.list_namespaces()
        chapters = get_chapters_from_contents(PDF_PATH)
        
        # If some chapters are missing, process the PDF
        chapter_namespaces = [f"chapter_{ch['chapter_number']}" for ch in chapters]
        missing_namespaces = [ns for ns in chapter_namespaces if ns not in namespaces]
        
        if missing_namespaces:
            logger.info(f"Processing PDF: {len(missing_namespaces)} chapters need to be processed")
            rag.process_pdf_to_pinecone(PDF_PATH, pdf_loader)
        else:
            logger.info("PDF already processed, skipping.")
    except Exception as e:
        logger.error(f"Error initializing data: {str(e)}")

with app.app_context():
    initialize_data()

@app.route('/')
def index():
    """Render the main page."""
    try:
        pdf_info = get_pdf_info(PDF_PATH)
        chapters = get_chapters_from_contents(PDF_PATH)
        return render_template('index.html', pdf_info=pdf_info, chapters=chapters)
    except Exception as e:
        logger.error(f"Error rendering index: {str(e)}")
        return render_template('error.html', error=str(e))

@app.route('/api/chapters', methods=['GET'])
def api_chapters():
    """Get the list of chapters in the PDF."""
    try:
        chapters = get_chapters_from_contents(PDF_PATH)
        return jsonify(chapters), 200
    except Exception as e:
        logger.error(f"Error retrieving chapters: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def api_chat():
    """Handle chat requests."""
    try:
        data = request.json
        user_input = data.get('message', '')
        context_type = data.get('context_type', 'chapter')  
        
        if context_type == 'chapter':
            chapter_number = data.get('chapter_number')
            if not chapter_number:
                return jsonify({"error": "Chapter number is required"}), 400
            
            namespace = f"chapter_{chapter_number}"
            
            result = rag.answer_question(user_input, namespace)
            
        elif context_type == 'page_range':
            start_page = data.get('start_page')
            end_page = data.get('end_page')
            if not start_page or not end_page:
                return jsonify({"error": "Start and end page are required"}), 400
            result = rag.answer_from_page_range(user_input, start_page, end_page, PDF_PATH, pdf_loader)
            
        else:
            return jsonify({"error": "Invalid context type"}), 400
        
        return jsonify(result), 200
            
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/process-pdf', methods=['POST'])
def api_process_pdf():
    """Manually trigger PDF processing."""
    try:
        result = rag.process_pdf_to_pinecone(PDF_PATH, pdf_loader)
        if result:
            return jsonify({"message": "PDF processed successfully"}), 200
        else:
            return jsonify({"error": "PDF processing failed"}), 500
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)