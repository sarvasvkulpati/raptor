import logging
import os
from abc import ABC, abstractmethod
import torch
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential
from transformers import T5ForConditionalGeneration, T5Tokenizer

class BaseQAModel(ABC):
    @abstractmethod
    def answer_question(self, context, question):
        pass

class GPT4QAModel(BaseQAModel):
    def __init__(self, model="gpt-4o"):
        """
        Initializes the GPT-4 model with the specified model version.

        Args:
            model (str, optional): The model version to use. Defaults to "gpt-4o".
        """
        self.model = model
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def answer_question(self, context, question, max_tokens=150, stop_sequence=None):
        """
        Answers a question based on the given context.

        Args:
            context (str): The context to use for answering.
            question (str): The question to answer.
            max_tokens (int, optional): Maximum tokens in response. Defaults to 150.
            stop_sequence (str, optional): Sequence at which to stop. Defaults to None.

        Returns:
            str: The answer to the question.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": f"Using the following information: {context}\nAnswer the following question in less than 5-7 words, if possible: {question}"
                    }
                ],
                temperature=0,
                max_tokens=max_tokens,
                stop=stop_sequence
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(e)
            return ""

class GPT4StandardQAModel(BaseQAModel):
    def __init__(self, model="gpt-4o"):
        """
        Initializes the GPT-4 model for standard QA tasks.

        Args:
            model (str, optional): The model version to use. Defaults to "gpt-4o".
        """
        self.model = model
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def _attempt_answer_question(self, context, question, max_tokens=150, stop_sequence=None):
        """
        Internal method to attempt answering a question.

        Args:
            context (str): The context to use for answering.
            question (str): The question to answer.
            max_tokens (int, optional): Maximum tokens in response. Defaults to 150.
            stop_sequence (str, optional): Sequence at which to stop. Defaults to None.

        Returns:
            str: The answer to the question.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a Question Answering Portal"},
                {
                    "role": "user",
                    "content": f"Given Context: {context}\nGive the best full answer amongst the options to question: {question}"
                }
            ],
            temperature=0,
        )
        return response.choices[0].message.content.strip()

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def answer_question(self, context, question, max_tokens=150, stop_sequence=None):
        try:
            return self._attempt_answer_question(
                context, question, max_tokens=max_tokens, stop_sequence=stop_sequence
            )
        except Exception as e:
            print(e)
            return str(e)

class UnifiedQAModel(BaseQAModel):
    def __init__(self, model_name="allenai/unifiedqa-v2-t5-3b-1363200"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

    def run_model(self, input_string, **generator_args):
        input_ids = self.tokenizer.encode(input_string, return_tensors="pt").to(self.device)
        res = self.model.generate(input_ids, **generator_args)
        return self.tokenizer.batch_decode(res, skip_special_tokens=True)

    def answer_question(self, context, question):
        input_string = question + " \\n " + context
        output = self.run_model(input_string)
        return output[0]