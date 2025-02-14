import json
import os
import logging
from typing import List, Any
from typing_extensions import Annotated

import dotenv
from pydantic import BaseModel, Field
from openai import AzureOpenAI

# Load environment variables from the specified file
dotenv.load_dotenv("../../.env")

def get_model_config():
    return {
        "azure_endpoint": os.environ["AZURE_OPENAI_API_BASE"],
        "api_key": os.environ["AZURE_OPENAI_API_KEY"],
        "azure_deployment": os.environ["AZURE_OPENAI_MODEL_MINI"], #"gpt-4o-mini-truthfulqa-judge-test"
        "api_version": os.environ["AZURE_OPENAI_API_VERSION"]
    }

def get_openai_client() -> Any:
    try:
        model_config = get_model_config()
        client = AzureOpenAI(
            azure_endpoint=model_config["azure_endpoint"],
            api_key=model_config["api_key"],
            api_version=model_config["api_version"],
        )
        return client
    except KeyError as e:
        print("Missing required model_config key: %s", e)
        raise

def embed_text(text: str) -> List[float]:
    response = get_openai_client().embeddings.create(
        input=text,
        model=os.getenv("AZURE_OPENAI_EMBEDDING_MODEL")
    )
    return response.data[0].embedding



if __name__ == "__main__":
    text = "What is the capital of France?"
    print(embed_text(text))