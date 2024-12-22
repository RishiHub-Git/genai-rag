from doc_processing_service import DocProcessingService
from embedding_huggingface_service import EmbeddingHGFService
from embedding_open_service import EmbeddingOpenService
from prompt_service import PromptService
from llm_cohere_service import LLMCohereService
from llm_open_service import LLMOpenService
from embeddings_enum import EmbeddingOption
from vectorstores_enum import VectorStoresOptions
import warnings
warnings.filterwarnings('ignore')

class ProcessRAG:
    def __init__(self):
        self.texts = None
        self.embeddings = EmbeddingOption.OpenAIEmbedding
        self.vector_proccessor = None
        self.vector_retriever = None
        self.prompt_template = None
        self.process_document()
        self.prepare_vector()
        self.prepare_prompt()

    
    def process_document(self):
        doc_proccessor = DocProcessingService(
                    directory_path='static/resource',
                    file_pattern='*.pdf',
                    chunk_size=1000,
                    chunk_overlap=200)
        #if self.transformer is TransformersOption.HuggingFaceEmbedding:
        #    self.texts = doc_proccessor.process_texts()
        #else:
        self.texts = doc_proccessor.process_texts()
    
    def prepare_vector(self):
        if self.embeddings is EmbeddingOption.HuggingFaceEmbedding:
            self.vector_proccessor = EmbeddingHGFService()
        else:
            self.vector_proccessor = EmbeddingOpenService(VectorStoresOptions.FAISS)
        
        self.vector_proccessor.embedding_and_vector_persist(self.texts)
        self.vector_retriever = self.vector_proccessor.get_vector_retriever()

    def prepare_prompt(self):
        prompt = PromptService()
        self.prompt_template = prompt.get_prompt_template()

    def process_prompt(self, questions):
        if self.embeddings is EmbeddingOption.HuggingFaceEmbedding:
            llm_service = LLMCohereService()
            response = llm_service.generate_answer(questions, self.vector_retriever, self.prompt_template)
            return response
        else:
            llm_service = LLMOpenService()
            response = llm_service.generate_answer(questions, self.vector_retriever)
            return response