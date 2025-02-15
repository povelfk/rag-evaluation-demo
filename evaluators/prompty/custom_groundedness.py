import json
import os
# from promptflow.client import load_flow

import prompty.azure_beta
import prompty
from typing_extensions import Annotated
from pydantic import BaseModel, Field


class CustomGroundednessEvaluator:
    def __init__(self, model_config):
        # current_dir = os.path.dirname(__file__)
        # prompty_path = os.path.join(current_dir, "custom_groundedness.prompty")
        self.model_config = model_config

        # self._flow = load_flow(source=prompty_path, model={"configuration": model_config})
    class GroundedGrade(BaseModel):
        explanation: Annotated[str, ..., "Explain your reasoning for the score"]
        score: Annotated[int, ..., "Provide the score on if the answer hallucinates from the documents, 0 for hallucination, 1 for no hallucination"]
    
    def __call__(self, *, query: str, context: str, response: str, **kwargs):
        llm_response = prompty.execute(
            "custom_groundedness.prompty",
            inputs={
                "query": query,
                "context": context,
                "response": response
            },
            parameters={
                "response_format": self.GroundedGrade
            }
        )
        try:
            response = json.loads(llm_response)
        except Exception as e:
            response = llm_response
        return response

    
