from abc import ABC, abstractmethod

class Embeddings(ABC):
    @abstractmethod
    def embedding_and_vector_persist(self, texts: any):
        pass
    
    @abstractmethod
    def get_vector_retriever(self):
        pass
