import os
import logging
from typing import List, Dict, Any, Optional
from PyPDF2 import PdfReader
import docx2txt
from unstructured.partition.text import partition_text
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP, DOCUMENT_STORE_PATH

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles document loading, parsing, and chunking for the RAG system."""
    
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        """Initialize the document processor.
        
        Args:
            chunk_size: Size of text chunks for processing
            chunk_overlap: Overlap between chunks to maintain context
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        
    def load_document(self, file_path: str) -> str:
        """Load and extract text from a document file.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Extracted text content from the document
        
        Raises:
            ValueError: If the file format is not supported
        """
        _, file_extension = os.path.splitext(file_path)
        file_extension = file_extension.lower()
        
        try:
            if file_extension == '.pdf':
                return self._extract_from_pdf(file_path)
            elif file_extension == '.docx':
                return self._extract_from_docx(file_path)
            elif file_extension == '.txt':
                return self._extract_from_txt(file_path)
            elif file_extension == '.md':
                return self._extract_from_txt(file_path)  # Markdown can be treated as text
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {str(e)}")
            raise
    
    def _extract_from_pdf(self, file_path: str) -> str:
        """Extract text from a PDF file."""
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text
    
    def _extract_from_docx(self, file_path: str) -> str:
        """Extract text from a DOCX file."""
        return docx2txt.process(file_path)
    
    def _extract_from_txt(self, file_path: str) -> str:
        """Extract text from a plain text file."""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    def chunk_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Split text into chunks for processing.
        
        Args:
            text: The text to split into chunks
            metadata: Optional metadata to associate with each chunk
            
        Returns:
            List of dictionaries containing text chunks and metadata
        """
        chunks = self.text_splitter.split_text(text)
        
        # Create document chunks with metadata
        documents = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy() if metadata else {}
            chunk_metadata["chunk_id"] = i
            documents.append({
                "text": chunk,
                "metadata": chunk_metadata
            })
        
        logger.info(f"Split document into {len(documents)} chunks")
        return documents
    
    def process_document(self, file_path: str) -> List[Dict[str, Any]]:
        """Process a document: load, extract text, and split into chunks.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of dictionaries containing text chunks and metadata
        """
        # Extract filename for metadata
        filename = os.path.basename(file_path)
        
        # Load and extract text
        text = self.load_document(file_path)
        
        # Create metadata
        metadata = {
            "source": filename,
            "file_path": file_path,
        }
        
        # Chunk the document
        return self.chunk_text(text, metadata)
    
    def save_document(self, file_obj, filename: str) -> str:
        """Save an uploaded document to the document store.
        
        Args:
            file_obj: File object from the upload
            filename: Name of the file
            
        Returns:
            Path where the document was saved
        """
        os.makedirs(DOCUMENT_STORE_PATH, exist_ok=True)
        file_path = os.path.join(DOCUMENT_STORE_PATH, filename)
        
        file_obj.save(file_path)
        logger.info(f"Saved document to {file_path}")
        
        return file_path
