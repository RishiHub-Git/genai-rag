from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from embeddings_abstract import Embeddings
from vectorstores_enum import VectorStoresOptions
from variables import Variables

import warnings
warnings.filterwarnings('ignore')

class EmbeddingOpenService(Embeddings):
    def __init__(self, vectorStoreOption):
        """
        Initialize the VectorService with vectorDb as None.
        """
        self.VectorStore = vectorStoreOption
        self.vectorDb = None
        self.FAISSDb = None
        self.API_variable = Variables.get_variable("llmopen")

    def embedding_and_vector_persist(self, texts: any):
        """
        Create embeddings for the given texts and persist them into a vector database.
        """
        # Load the embedding model
        embedding = OpenAIEmbeddings(openai_api_key=self.API_variable)
        self.__save_vector(texts= texts, embedding= embedding)

    def __save_vector(self, texts, embedding):
        # Create and persist the vector database
        if self.VectorStore is VectorStoresOptions.Chroma:
            try:
                self.vectorDb = Chroma(embedding_function=embedding, persist_directory="Vector_Directory")
                print("Vector Db loaded")
            except:
                self.vectorDb = Chroma.from_texts(texts, embedding=embedding, persist_directory="Vector_Directory")
                self.vectorDb.persist()
                print("vector Db saved")
            
            print("Vector database has been created and persisted successfully.")
        else:
            self.FAISSDb = FAISS.from_texts(texts, embedding)
            #vector_stores = FAISS.from_documents(texts, embedding)
            print("Vector Db generated from FAISS")

    def get_vector_retriever(self):
        """
        Return the vector retriever for similarity searches.
        """
        if self.VectorStore is VectorStoresOptions.Chroma:
            print("Chroma Vector Db Retriever")
            return self.vectorDb.as_retriever()
        else:
            print("FAISS Vector Db Retriever")
            return self.FAISSDb.as_retriever(seacrch_type="similarity", search_kwargs={"k": 6})