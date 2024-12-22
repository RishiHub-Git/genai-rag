from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import warnings
warnings.filterwarnings('ignore')

class DocProcessingService:
    def __init__(self, directory_path, file_pattern='*.pdf', chunk_size=2000, chunk_overlap=400):
        """
        Initializes the service with directory path, file pattern, and chunking configuration.
        """
        self.directory_path = directory_path
        self.file_pattern = file_pattern
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_pdf_documents(self):
        """
        Load PDF documents from the specified directory.
        """
        doc_loader = DirectoryLoader(self.directory_path, glob=self.file_pattern, loader_cls=PyPDFLoader)
        documents = doc_loader.load()
        print('Document loaded')
        return documents
    
    def process_texts(self):
        """
        High-level method to process PDF documents and return text chunks.
        """
        documents = self.load_pdf_documents()
        all_texts = self.extract_text(documents)
        text_chunks = self.split_text_into_chunks(all_texts)
        return text_chunks
    
    def process_documents(self):
        """
        High-level method to process PDF documents and return text chunks.
        """
        documents = self.load_pdf_documents()
        text_chunks = self.split_document_into_chunks(documents)
        return text_chunks

    def extract_text(self, documents):
        """
        Extract and concatenate text content from the loaded documents.
        """
        return ' '.join([document.page_content for document in documents])
    
    def split_text_into_chunks(self, all_texts):
        """
        Split the concatenated text into smaller chunks for vector conversion.
        """
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        return text_splitter.split_text(all_texts)
    
    def split_document_into_chunks(self, documents):
        """
        Split the concatenated text into smaller chunks for vector conversion.
        """
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
        return text_splitter.split_documents(documents)
