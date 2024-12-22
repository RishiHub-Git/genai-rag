import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from embeddings_abstract import Embeddings
#from variables import Variables

import warnings
warnings.filterwarnings('ignore')

class EmbeddingHGFService(Embeddings):
    def __init__(self):
        """
        Initialize the VectorService with vectorDb as None.
        """
        self.vectorDb = None
        #self.API_variable = Variables.get_variable("HuggingFaceKey")
        os.environ["HuggingFaceKey"] = "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"

    def embedding_and_vector_persist(self, texts: any):
        """
        Create embeddings for the given texts and persist them into a vector database.
        """
        # Load the embedding model
        embedding = HuggingFaceEmbeddings(model_name= "sentence-transformers/all-MiniLM-L6-v2")
        self.__save_vector(texts= texts, embedding= embedding)

    def __save_vector(self, texts, embedding):
        # Create and persist the vector database
        self.vectorDb = Chroma.from_texts(texts, embedding=embedding, persist_directory="Vector_Directory")
        self.vectorDb.persist()
        print("Vector database has been created and persisted successfully.")

    def get_vector_retriever(self):
        """
        Return the vector retriever for similarity searches.
        """
        if self.vectorDb is None:
            raise ValueError("Vector database is not initialized. Call embedding_and_vector_persist first.")
        
        return self.vectorDb.as_retriever(search_type = "similarity",search_kwargs={'k': 6})