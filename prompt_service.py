from langchain_core.prompts import PromptTemplate
import warnings
warnings.filterwarnings('ignore')

class PromptService:
    def get_prompt_template(self):
        """
        Get the Prompt Template
        """
        prompt_template = """Answer the question as precise as possible using provided context. If the question is outside the context then answer 'The Topic is out of Context.' \n\n
        Context: \n {context}?\n
        Question: \n {question} \n
        Answer:"""

        #PromptTemplate is Only needed if you have variable in your prompt like {context} and {question} here
        prompt = PromptTemplate.from_template(template = prompt_template)
        return prompt
    