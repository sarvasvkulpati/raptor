import logging
import os
from abc import ABC, abstractmethod

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

class BaseSummarizationModel(ABC):
    @abstractmethod
    def summarize(self, context, max_tokens=150):
        pass

class GPT4StandardSummarizationModel(BaseSummarizationModel):
    def __init__(self, model="gpt-4o"):
        """
        Initializes the GPT-4 summarization model.
        
        Args:
            model (str, optional): The model to use. Defaults to "gpt-4o".
        """
        self.model = model
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def summarize(self, context, max_tokens=500, stop_sequence=None):
        """
        Generates a summary using the GPT-4 model.
        
        Args:
            context (str): The text to summarize.
            max_tokens (int, optional): Maximum tokens in the summary. Defaults to 500.
            stop_sequence (str, optional): Sequence at which to stop. Defaults to None.
            
        Returns:
            str: The generated summary.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": f"Write a summary of the following, including as many key details as possible: {context}:"
                    }
                ],
                max_tokens=max_tokens,
                stop=stop_sequence
            )
            return response.choices[0].message.content

        except Exception as e:
            print(e)
            return str(e)

class GPT4DetailedSummarizationModel(BaseSummarizationModel):
    def __init__(self, model="gpt-4o"):
        """
        Initializes the GPT-4 detailed summarization model.
        
        Args:
            model (str, optional): The model to use. Defaults to "gpt-4o".
        """
        self.model = model
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def summarize(self, context, max_tokens=500, stop_sequence=None):
        """
        Generates a detailed summary using the GPT-4 model.
        
        Args:
            context (str): The text to summarize.
            max_tokens (int, optional): Maximum tokens in the summary. Defaults to 500.
            stop_sequence (str, optional): Sequence at which to stop. Defaults to None.
            
        Returns:
            str: The generated detailed summary.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant specializing in detailed summarization."},
                    {
                        "role": "user",
                        "content": f"Write a comprehensive summary of the following, capturing all important details, key points, and maintaining the context's depth: {context}:"
                    }
                ],
                max_tokens=max_tokens,
                stop=stop_sequence
            )
            return response.choices[0].message.content

        except Exception as e:
            print(e)
            return str(e)