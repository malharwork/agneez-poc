import os
from pypdf import PdfReader
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

def get_pdf_info(pdf_path: str) -> Dict[str, any]:
    """
    Get basic information about a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Dictionary with information about the PDF
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at {pdf_path}")
    
    reader = PdfReader(pdf_path)
    
    return {
        "total_pages": len(reader.pages),
        "title": reader.metadata.get("/Title", "English Textbook"),
        "author": reader.metadata.get("/Author", "Unknown"),
        "creation_date": reader.metadata.get("/CreationDate", "Unknown")
    }

def get_chapters_from_contents(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Use the manually defined table of contents to create chapter data.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        List of dictionaries with chapter number, title, and page range
    """
    # Define chapters based on the contents page in the image
    chapters = [
        # Unit One
        {"chapter_number": 1, "title": "What a Bird Thought", "start_page": 1, "end_page": 1, "unit": "Unit One"},
        {"chapter_number": 2, "title": "Daydreams", "start_page": 2, "end_page": 4, "unit": "Unit One"},
        {"chapter_number": 3, "title": "Be a Good Listener", "start_page": 5, "end_page": 6, "unit": "Unit One"},
        {"chapter_number": 4, "title": "Strawberries", "start_page": 7, "end_page": 7, "unit": "Unit One"},
        {"chapter_number": 5, "title": "The Twelve Months", "start_page": 8, "end_page": 13, "unit": "Unit One"},
        {"chapter_number": 6, "title": "Announcements", "start_page": 14, "end_page": 15, "unit": "Unit One"},
        {"chapter_number": 7, "title": "Major Dhyan Chand", "start_page": 16, "end_page": 18, "unit": "Unit One"},
        {"chapter_number": 8, "title": "Peer Profile", "start_page": 19, "end_page": 20, "unit": "Unit One"},
        
        # Unit Two
        {"chapter_number": 9, "title": "The Triantiwontigongolope", "start_page": 21, "end_page": 22, "unit": "Unit Two"},
        {"chapter_number": 10, "title": "Three Sacks of Rice", "start_page": 23, "end_page": 25, "unit": "Unit Two"},
        {"chapter_number": 11, "title": "Be a Good Speaker", "start_page": 26, "end_page": 27, "unit": "Unit Two"},
        {"chapter_number": 12, "title": "Count your Garden", "start_page": 28, "end_page": 28, "unit": "Unit Two"},
        {"chapter_number": 13, "title": "The Adventures of Gulliver", "start_page": 29, "end_page": 32, "unit": "Unit Two"},
        {"chapter_number": 14, "title": "A Lesson for All", "start_page": 33, "end_page": 37, "unit": "Unit Two"},
        {"chapter_number": 15, "title": "Bird Bath", "start_page": 38, "end_page": 38, "unit": "Unit Two"},
        {"chapter_number": 16, "title": "Write your own Story", "start_page": 39, "end_page": 41, "unit": "Unit Two"},
        
        # Unit Three
        {"chapter_number": 17, "title": "On the Water", "start_page": 42, "end_page": 43, "unit": "Unit Three"},
        {"chapter_number": 18, "title": "Weeds in the Garden", "start_page": 44, "end_page": 46, "unit": "Unit Three"},
        {"chapter_number": 19, "title": "Be a Good Host and Guest", "start_page": 47, "end_page": 50, "unit": "Unit Three"},
        {"chapter_number": 20, "title": "Only One Mother", "start_page": 51, "end_page": 51, "unit": "Unit Three"},
        {"chapter_number": 21, "title": "The Journey to the Great Oz", "start_page": 52, "end_page": 56, "unit": "Unit Three"},
        {"chapter_number": 22, "title": "A Book Review", "start_page": 57, "end_page": 58, "unit": "Unit Three"},
        {"chapter_number": 23, "title": "Write your own Poem", "start_page": 59, "end_page": 59, "unit": "Unit Three"},
        {"chapter_number": 24, "title": "Senses Alert", "start_page": 60, "end_page": 61, "unit": "Unit Three"},
        
        # Unit Four
        {"chapter_number": 25, "title": "The Man in the Moon", "start_page": 62, "end_page": 62, "unit": "Unit Four"},
        {"chapter_number": 26, "title": "Water in the Well", "start_page": 63, "end_page": 64, "unit": "Unit Four"},
        {"chapter_number": 27, "title": "The Legend of Marathon", "start_page": 65, "end_page": 67, "unit": "Unit Four"},
        {"chapter_number": 28, "title": "All about Money", "start_page": 68, "end_page": 70, "unit": "Unit Four"},
        {"chapter_number": 29, "title": "A Lark", "start_page": 71, "end_page": 71, "unit": "Unit Four"},
        {"chapter_number": 30, "title": "Be a Netizen", "start_page": 72, "end_page": 74, "unit": "Unit Four"},
        {"chapter_number": 31, "title": "Give your Mind a Workout!", "start_page": 75, "end_page": 76, "unit": "Unit Four"},
        {"chapter_number": 32, "title": "Helen Keller", "start_page": 77, "end_page": 79, "unit": "Unit Four"},
        {"chapter_number": 33, "title": "Rangoli", "start_page": 80, "end_page": 82, "unit": "Unit Four"}
    ]
    for chapter in chapters:
        chapter["page_range"] = f"{chapter['start_page']}-{chapter['end_page']}" if chapter["start_page"] != chapter["end_page"] else str(chapter["start_page"])
    
    return chapters

def extract_text_from_chapter(pdf_path: str, chapter: Dict[str, Any]) -> str:
    """
    Extract text from a specific chapter.
    
    Args:
        pdf_path: Path to the PDF file
        chapter: Chapter dictionary with start_page and end_page
        
    Returns:
        Extracted text from the chapter
    """
    try:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found at {pdf_path}")
        
        reader = PdfReader(pdf_path)
        
        # Adjust for 0-indexing in PyPDF
        start_idx = max(0, chapter["start_page"] - 1)
        end_idx = min(len(reader.pages), chapter["end_page"])
        
        # Extract text from all pages in the chapter
        chapter_text = ""
        for i in range(start_idx, end_idx):
            try:
                page_text = reader.pages[i].extract_text()
                if page_text:
                    chapter_text += page_text + "\n"
            except Exception as e:
                logger.warning(f"Error extracting text from page {i+1}: {str(e)}")
        
        if not chapter_text.strip():
            logger.warning(f"No text extracted for Chapter {chapter['chapter_number']}: {chapter['title']}")
        
        return chapter_text
    except Exception as e:
        logger.error(f"Error extracting chapter text: {str(e)}")
        return ""

def extract_text_from_page_range(pdf_path, start_page, end_page):
    """
    Extract text from a range of pages in a PDF with multiple fallback methods.
    
    Args:
        pdf_path: Path to the PDF file
        start_page: Start page number (1-indexed)
        end_page: End page number (1-indexed)
        
    Returns:
        Extracted text
    """
    import fitz  
    import PyMuPDF
    import pdfplumber
    import logging
    
    logger = logging.getLogger(__name__)
    
    try:
        doc = fitz.open(pdf_path)
        text = ""
        
        # Adjust for 0-indexing
        for page_num in range(max(0, start_page-1), min(end_page, doc.page_count)):
            page = doc[page_num]
            text += page.get_text()
        
        doc.close()
        
        if text and len(text.strip()) > 100:  # Check if we got meaningful text
            return text
        else:
            logger.warning(f"PyMuPDF extraction yielded insufficient text for pages {start_page}-{end_page}, trying pdfplumber...")
    
    except Exception as e:
        logger.warning(f"PyMuPDF extraction failed: {str(e)}, trying pdfplumber...")
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            
            # Adjust for 0-indexing
            for page_num in range(max(0, start_page-1), min(end_page, len(pdf.pages))):
                page = pdf.pages[page_num]
                text += page.extract_text() or ""
        
        if text and len(text.strip()) > 100:
            return text
        else:
            logger.warning(f"pdfplumber extraction yielded insufficient text for pages {start_page}-{end_page}")
    
    except Exception as e:
        logger.warning(f"pdfplumber extraction failed: {str(e)}")
    
    # If we get here, both methods failed
    logger.error(f"All text extraction methods failed for pages {start_page}-{end_page}")
    return ""

def create_chapter_chunks(pdf_path: str, chapter: Dict[str, Any], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict[str, Any]]:
    """
    Create chunks from a specific chapter.
    
    Args:
        pdf_path: Path to the PDF file
        chapter: Chapter dictionary
        chunk_size: Maximum size of each chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of dictionaries with chunk text and metadata
    """
    try:
        chapter_text = extract_text_from_chapter(pdf_path, chapter)
        
        # Create chunks
        chunks = []
        
        # If no text was extracted, return empty list
        if not chapter_text or len(chapter_text.strip()) == 0:
            logger.warning(f"No text extracted for Chapter {chapter['chapter_number']}: {chapter['title']}")
            return chunks
        
        # If the text is shorter than chunk_size, keep it as is
        if len(chapter_text) <= chunk_size:
            chunks.append({
                "text": chapter_text,
                "chapter_number": chapter["chapter_number"],
                "chapter_title": chapter["title"],
                "unit": chapter["unit"],
                "page_range": chapter["page_range"]
            })
            return chunks
        
        # Otherwise, split it into smaller chunks with overlap
        start = 0
        chunk_id = 0
        while start < len(chapter_text):
            end = min(start + chunk_size, len(chapter_text))
            
            # If this is not the first chunk, include overlap
            if start > 0:
                start = max(0, start - chunk_overlap)
            
            chunk_text = chapter_text[start:end]
            chunks.append({
                "text": chunk_text,
                "chunk_id": chunk_id,
                "chapter_number": chapter["chapter_number"],
                "chapter_title": chapter["title"],
                "unit": chapter["unit"],
                "page_range": chapter["page_range"]
            })
            chunk_id += 1
            start = end
        
        logger.info(f"Created {len(chunks)} chunks for Chapter {chapter['chapter_number']}: {chapter['title']}")
        return chunks
    except Exception as e:
        logger.error(f"Error creating chapter chunks: {str(e)}")
        return []