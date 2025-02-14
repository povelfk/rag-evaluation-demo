import json
import os
import logging
from typing import List, Any
from typing_extensions import Annotated

import openai
import dotenv
from pydantic import BaseModel, Field

from ..sdg_generator_helper import get_instruction

# Load environment variables from the specified file
dotenv.load_dotenv("../.env")

# Configure logging (optional)
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

class QuestionSchema(BaseModel):
    """
    Schema for the JSON response returned by the LLM.
    
    The 'question' field is a list of synthetic questions related to the original topic.
    The 'response' field is a list of synthetic agent responses corresponding to the synthetic questions.
    The 'explanation' field is a list of explanations describing why the question and agent response are (or are not) grounded based on the provided context.
    """
    question: Annotated[
        List[str],
        Field(..., description="A list of synthetic questions related to the original topic.")
    ]
    response: Annotated[
        List[str],
        Field(..., description="A list of synthetic agent responses corresponding to the synthetic questions.")
    ]
    explanation: Annotated[
        List[str],
        Field(..., description="A list of explanations detailing why the question and agent response are (or are not) grounded based on the provided context.")
    ]



class GenerateSyntheticData:
    def __init__(self, model_config: dict, multi: bool=False):
        self.model_config=model_config
        if multi:
            system_message_path="configs/prompts/sdg/system_message_multi_grounded_not_grounded_questions.txt"
        else:
            system_message_path="configs/prompts/sdg/system_message_single_grounded_not_grounded_questions.txt"
        self.system_message = get_instruction(system_message_path)
        

    def get_openai_client(self) -> Any:
        try:
            client = openai.AzureOpenAI(
                azure_endpoint=self.model_config["azure_endpoint"],
                api_key=self.model_config["api_key"],
                api_version=self.model_config["api_version"],
            )
            return client
        except KeyError as e:
            print("Missing required model_config key: %s", e)
            raise

    def build_user_prompt(
        self,
        chunk: str,
        is_grounded: bool,
        domain: str,
        difficulty: str,
        topic: str,
        language: str,
        instructions: str,
        question_length: int,
        task: str,
    ) -> str:
        """
        Constructs the user prompt string based on provided parameters.
        Since we are only generating one question at a time, the "Number of Questions" parameter is removed.
        """
        prompt = (
            f"# Chunk:\n{chunk}\n\n"
            f"# Domain:\n{domain}\n\n"
            f"# Difficulty:\n{difficulty}\n\n"
            f"# Topic:\n{topic}\n\n"
            f"# Language:\n{language}\n\n"
            f"# Instructions:\n{instructions}\n\n"
            f"# Question Length (number of words):\n{question_length}\n\n"
            f"# Task:\n{task}\n\n"
            f"# Grounded: {is_grounded}\n"
        )
        return prompt

    def invoke_llm(
        self,
        chunk: str,
        is_grounded: bool,
        domain: str,
        difficulty: str,
        topic: str,
        language: str,
        instructions: str,
        question_length: int,
        task: str,
    ) -> dict:
        """
        Invokes the LLM using the provided parameters and returns the parsed response.
        """
        client = self.get_openai_client()
        user_prompt = self.build_user_prompt(
            chunk, is_grounded, domain, difficulty, topic,
            language, instructions, question_length, task
        )
        model_name = self.model_config.get(
            "azure_deployment",
            os.environ.get("AZURE_OPENAI_MODEL_MINI", "gpt-4o-mini")
        )

        try:
            response = client.beta.chat.completions.parse(
                model=model_name,
                messages=[
                    {"role": "system", "content": self.system_message},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,
                max_tokens=800,
                response_format=QuestionSchema
            )
        except Exception as e:
            print("Error during LLM invocation: %s", e)
            raise

        try:
            if not response.choices:
                raise ValueError("No choices returned from the LLM.")
            parsed = response.choices[0].message.parsed
            result = parsed.dict()  # For Pydantic v2, you might use model_dump() instead.
            return result
        except Exception as e:
            print("Error parsing LLM response: %s", e)
            raise

    def __call__(
        self,
        chunk: str,
        is_grounded: bool,
        domain: str,
        difficulty: str,
        topic: str,
        language: str,
        instructions: str,
        question_length: int,
        task: str,
    ) -> dict:
        """
        Makes the class instance callable. Delegates to invoke_llm.
        Note that we no longer include a num_questions parameter since it is fixed to 1.
        """
        return self.invoke_llm(
            chunk=chunk,
            is_grounded=is_grounded,
            domain=domain,
            difficulty=difficulty,
            topic=topic,
            language=language,
            instructions=instructions,
            question_length=question_length,
            task=task,
        )

# Example usage (for demonstration purposes)
if __name__ == "__main__":
    model_config = {
        "azure_endpoint": os.getenv("AZURE_ENDPOINT"),
        "api_key": os.getenv("AZURE_API_KEY"),
        "api_version": os.getenv("AZURE_API_VERSION"),
        # Optionally include your model name in the configuration
        "azure_deployment": os.getenv("AZURE_OPENAI_MODEL_MINI", "gpt-4o-mini")
    }

    system_message = (
        "You are an AI assistant specialized in generating synthetic questions.\n\n"
        "You will be given:\n"
        "1. A chunk from a search index (the information source).\n"
        "2. A boolean 'is_grounded' indicating whether your new question should be answerable based on the given chunk (True) or not (False).\n"
        "3. Additional metadata such as domain, difficulty, topic, language, instructions, question length, and the task.\n\n"
        "Your task is to produce a new, synthetic question that adheres to these guidelines:\n\n"
        "• If 'is_grounded' is True, the question must be clearly answerable using the golden context, or parts of the golden context.\n"
        "• If 'is_grounded' is False, the question must not be answerable by the golden context.\n"
        "• The question must remain relevant in topic, style, tone, and difficulty to the original question.\n"
        "• Avoid trivial restatements, direct copies, or contradictions (when is_grounded is True).\n"
        "• The question should be distinct.\n\n"
        "**Output Formatting**\n"
        "Your output must be strictly valid JSON, containing exactly one key \"questions\" mapped to an array of strings. For example:\n"
        "{\n  \"questions\": [\n    \"Synthetic Question 1\"\n  ]\n}\n"
        "No extra keys or commentary—only the “questions” array of strings."
    )

    task = (
        "Generate exactly one new question related to the provided domain, difficulty, and other parameters. \n"
        "• If 'is_grounded' is True, ensure the question is answerable directly from the golden context.\n"
        "• If 'is_grounded' is False, ensure the question is not answerable from the golden context.\n"
        "• The question should be creative and factually correct.\n"
        "Return your response in valid JSON format with a single key \"questions\" containing an array with one question."
    )

    generator = GenerateSyntheticData(model_config, system_message)

    # Example call to generate a single synthetic question
    try:
        result = generator(
            chunk="What are the health benefits of green tea?",
            is_grounded=True,
            domain="Health & Wellness",
            difficulty="Intermediate",
            topic="Nutrition",
            language="English",
            instructions="Create a creative yet factually correct question.",
            question_length=6,
            task=task,
        )
        print(json.dumps(result, indent=2))
    except Exception as error:
        print("Failed to generate synthetic question: %s", error)
