import os
from langchain_community.llms import Cohere
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from variables import Variables

import warnings
warnings.filterwarnings('ignore')

class LLMCohereService:
    def __init__(self):
        """
        This is the constructor
        """
        self.API_variable = Variables.get_variable("cohere")

    def format_documents(self, document):
        return "\n\n".join(doc.page_content for doc in document)

    def generate_answer(self, question : any, vector_retriever : any, prompt : any):
        cohere_llm = Cohere(model="command", 
                            temperature = 0.1, 
                            cohere_api_key = self.API_variable)

        rag_chain = (
            {"context": vector_retriever | self.format_documents, "question": RunnablePassthrough()}
            | prompt
            | cohere_llm
            | StrOutputParser()
        )
        return rag_chain.invoke(question)
    
    