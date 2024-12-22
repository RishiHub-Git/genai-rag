import openai
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from variables import Variables
import warnings
warnings.filterwarnings('ignore')

class LLMOpenService:
    def __init__(self):
        """
        This is the constructor
        """
        self.API_variable = Variables.get_variable("llmopen")

    def format_documents(self, document):
        return "\n\n".join(doc.page_content for doc in document)

    def generate_answer(self, question : any, vector_retriever : any):
        llm = ChatOpenAI(model = "gpt-3.5-turbo",
                 openai_api_key = self.API_variable,
                 temperature = 1.0,
                 max_tokens = 512
                 )

        retrieval_chain = RetrievalQA.from_chain_type(llm,
                                              retriever = vector_retriever,
                                              return_source_documents = False
                                              )
        result = retrieval_chain(question)
        print(result)
        return result
    
    