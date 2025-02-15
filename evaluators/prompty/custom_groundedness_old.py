import json
import os
from promptflow.client import load_flow

class CustomGroundednessEvaluator:
    def __init__(self, model_config):
        current_dir = os.path.dirname(__file__)
        prompty_path = os.path.join(current_dir, "custom_groundedness.prompty")
        self._flow = load_flow(source=prompty_path, model={"configuration": model_config})

    def __call__(self, *, query: str, context: str, response: str, **kwargs):
        llm_response = self._flow(query=query, context=context, response=response)
        try:
            response = json.loads(llm_response)
        except Exception as ex:
            response = llm_response
        return response
