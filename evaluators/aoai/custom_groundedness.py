import json
import os

# from promptflow.client import load_flow
from typing_extensions import Annotated
import openai
from pydantic import BaseModel
from utils.sdg_generator_helper import get_instruction


class CustomGroundednessEvaluator:
    def __init__(self, model_config):
        self.model_config=model_config
        self.system_message = get_instruction("configs/prompts/judge/system_message_single_groundedness.txt")

    # Grade output schema
    class GroundedGrade(BaseModel):
        score: Annotated[int, ...,
            """Provide a score indicating whether the agent response hallucinates from the documents:
            0 = hallucination, 1 = no hallucination."""
        ]
        explanation: Annotated[str, ...,
            """Explain your reasoning for the score."""
        ]

    def get_openai_client(self):
        return openai.AzureOpenAI(
            azure_endpoint=self.model_config["azure_endpoint"],
            api_key=self.model_config["api_key"],
            api_version=self.model_config["api_version"],
        )

    def invoke_llm(self, query, context, response):
        user_message = (
            f"QUESTION: {query}\n"
            f"CHUNK: {context}\n"
            f"RESPONSE: {response}"
        )
        llm_response = self.get_openai_client().beta.chat.completions.parse(
            model = self.model_config["azure_deployment"],
            messages = [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": user_message}
            ],
            temperature = 0.2,
            response_format=self.GroundedGrade,
        )
        return llm_response.choices[0].message.content

    def __call__(self, *, query: str, context: str, response: str, **kwargs):
        llm_response = self.invoke_llm(query=query, context=context, response=response)
        try:
            response = json.loads(llm_response)
        except Exception as ex:
            response = llm_response
            print(f"Error parsing response: {response}, {ex}")
        return response
