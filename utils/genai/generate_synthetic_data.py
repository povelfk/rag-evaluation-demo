import json
import os
import logging
from typing import List, Any
from typing_extensions import Annotated

import dotenv
from pydantic import BaseModel, Field
from tqdm import tqdm

# from ..sdg_generator_helper import get_instruction

import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from utils.sdg_generator_helper import (
    get_instruction,
    get_domain,
    get_tone,
    get_question_length,
    get_difficulty,
    get_topic,
    get_language,
    set_is_grounded
)

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



def get_top_k_similar_chunks(chunks, current_index, k=3):
    """
    For the chunk at current_index, compute cosine similarity to all other chunks,
    and return the texts of the top-k most similar chunks (excluding the current one).
    """
    embeddings = np.array([chunk["chunk_embedding"] for chunk in chunks])
    current_embedding = embeddings[current_index].reshape(1, -1)
    similarities = cosine_similarity(current_embedding, embeddings).flatten()
    
    similarities[current_index] = -np.inf # Exclude the current chunk by setting its similarity to -inf

    top_k_indices = np.argpartition(similarities, -k)[-k:] # Get the indices of the top-k most similar chunks
    top_k_indices = top_k_indices[np.argsort(similarities[top_k_indices])[::-1]] # Sort the indices by similarity
    return [chunks[i]["chunk"] for i in top_k_indices]

def build_multi_chunk_context(primary_chunk, additional_chunks):
    """
    Combine the primary chunk with additional similar chunks into one context string.
    Use clear delimiters to indicate different parts.
    """
    context_parts = []
    context_parts.append("Primary Chunk:\n" + primary_chunk)
    for i, add_chunk in enumerate(additional_chunks, start=1):
        context_parts.append(f"Additional Context {i}:\n" + add_chunk)
    
    full_context = "\n\n".join(context_parts)
    return full_context


def generate_synthetic_questions(chunks, generator, num_questions_per_chunk, multi=False):
    """
    For each chunk, retrieve the top_k similar chunks using cosine similarity,
    build a multi-chunk context, and generate synthetic questions.
    """
    all_results = []
    failed_results = []
    total_iterations = len(chunks) * num_questions_per_chunk

    with tqdm(total=total_iterations, desc="Generating synthetic questions") as pbar:
        for current_index, chunk in enumerate(chunks):
            if multi:
                top_k=3
                similar_chunks = get_top_k_similar_chunks(chunks, current_index, k=top_k)
                multi_chunk_context = build_multi_chunk_context(chunk["chunk"], similar_chunks)
                context=multi_chunk_context
                task_path = "configs/settings/tasks/task_multi_grounded_not_grounded_questions.txt"
            else:
                context = chunk["chunk"]
                task_path = "configs/settings/tasks/task_single_grounded_not_grounded_questions.txt"
                multi_chunk_context = "single chunk was used"

            for question_index in range(num_questions_per_chunk):
                # Sample metadata for each question
                domain = get_domain()
                tone = get_tone()
                difficulty = get_difficulty()
                question_length = get_question_length(min_length=4)
                topic = get_topic()
                language = get_language()
                instructions = "None"
                task = get_instruction(task_path)
                is_grounded = set_is_grounded()

                try:
                    # Call the generator with the multi-chunk context
                    result = generator(
                        chunk=context,
                        is_grounded=is_grounded,
                        domain=domain,
                        difficulty=difficulty,
                        topic=topic,
                        language=language,
                        instructions=instructions,
                        question_length=question_length,
                        task=task
                    )
                    # Since we requested one question, grab the first element.
                    question = result["question"][0]
                    explanation = result["explanation"][0]
                    response = result["response"][0]
                    synthetic_chunk_id = f"{chunk['chunk_id']}_synthetic_{question_index+1}"


                    all_results.append({
                        "synthetic_question": question,
                        "explanation": explanation,
                        "synthetic_response": response,
                        "chunk_id": synthetic_chunk_id,
                        "is_grounded": is_grounded,
                        "chunk_data": chunk["chunk"],
                        "aggregated_context": multi_chunk_context,
                        "question_number": question_index,
                        "domain": domain,
                        "difficulty": difficulty,
                        "tone": tone,
                        "language": language,
                        "question_length": question_length
                    })
                except Exception as e:
                    failed_results.append({
                        "error": str(e),
                        "chunk_id": chunk["chunk_id"],
                        "is_grounded": is_grounded,
                        "chunk_data": chunk["chunk"],
                        "question_number": question_index,
                        "domain": domain,
                        "difficulty": difficulty,
                        "tone": tone,
                        "language": language,
                        "question_length": question_length
                    })

                pbar.update(1)

    return all_results, failed_results
