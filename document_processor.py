# document_processor.py
import os
import re
from typing import List, Dict, Any
from pypdf import PdfReader
import docx
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def extract_metadata(self, filepath: str) -> Dict[str, Any]:
        """Extract metadata from a document"""
        filename = os.path.basename(filepath)
        file_ext = os.path.splitext(filename)[1].lower()
        file_size = os.path.getsize(filepath)
        
        metadata = {
            "filename": filename,
            "extension": file_ext,
            "size": file_size,
            "pages": 0,
            "sections": []
        }
        
        return metadata
    
    def process_pdf(self, filepath: str) -> str:
        """Process PDF document and extract text"""
        reader = PdfReader(filepath)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            # Clean up the text: replace single newlines with spaces, but keep double newlines
            page_text = re.sub(r'(?<!\n)\n(?!\n)', ' ', page_text)
            text += page_text + "\n"
        return text
    
    def process_docx(self, filepath: str) -> str:
        """Process DOCX document and extract text"""
        doc = docx.Document(filepath)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    
    def process_txt(self, filepath: str) -> str:
        """Process TXT document and extract text"""
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        return text
    
    def process_document(self, filepath: str) -> Dict[str, Any]:
        """Process a document based on its type"""
        metadata = self.extract_metadata(filepath)
        file_ext = metadata["extension"]
        
        if file_ext == '.pdf':
            text = self.process_pdf(filepath)
        elif file_ext in ['.docx', '.doc']:
            text = self.process_docx(filepath)
        elif file_ext == '.txt':
            text = self.process_txt(filepath)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        # Clean text
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Split into chunks
        chunks = self.text_splitter.split_text(text)
        
        # Extract potential sections (for research reports)
        sections = self.extract_sections(text)
        
        return {
            "metadata": metadata,
            "text": text,
            "chunks": chunks,
            "sections": sections
        }
    
    def extract_sections(self, text: str) -> List[Dict[str, str]]:
        """Extract potential sections from document text"""
        # This is a simple implementation - could be enhanced with ML
        section_pattern = r'\n([A-Z][A-Za-z\s]{3,40})\n[-=]*\n'
        sections = re.findall(section_pattern, text)
        
        return [{"title": section, "content": ""} for section in sections]