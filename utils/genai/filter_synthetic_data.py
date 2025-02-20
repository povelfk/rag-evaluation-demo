import re
import numpy as np
from typing import List, Dict, Tuple
from tqdm import tqdm

from utils.genai.invoker import embed_text

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Computes cosine similarity between two numpy vectors.
    """
    dot_product = np.dot(vec1, vec2)
    norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    return 0.0 if norm_product == 0 else dot_product / norm_product

def jaccard_similarity(q1: str, q2: str) -> float:
    """
    Computes Jaccard similarity between two question strings based on token overlap.
    """
    set1 = set(re.findall(r"\w+", q1.lower()))
    set2 = set(re.findall(r"\w+", q2.lower()))
    if not set1 and not set2:
        return 0.0
    return len(set1.intersection(set2)) / len(set1.union(set2))

def is_duplicate(
    q1: str,
    q2: str,
    jaccard_threshold: float,
    embedding_threshold: float,
    combination_mode: str = "or",
    use_embedding: bool = False
) -> bool:
    """
    Determines if two questions are duplicates based on:
      - Jaccard similarity (token overlap), and optionally,
      - Embedding-based cosine similarity.
    """
    j_score = jaccard_similarity(q1, q2)
    if not use_embedding:
        return j_score >= jaccard_threshold

    vec1 = embed_text(q1)
    vec2 = embed_text(q2)
    c_score = cosine_similarity(vec1, vec2)

    if combination_mode == "or":
        return (j_score >= jaccard_threshold) or (c_score >= embedding_threshold)
    elif combination_mode == "and":
        return (j_score >= jaccard_threshold) and (c_score >= embedding_threshold)
    elif combination_mode == "weighted":
        alpha = 0.5  # Adjust as needed
        combined_score = alpha * j_score + (1 - alpha) * c_score
        weighted_threshold = 0.8  # Tune this threshold as needed
        return combined_score >= weighted_threshold
    else:
        raise ValueError(f"Invalid combination_mode: {combination_mode}.")

def filter_synthetic_questions(
    synthetic_data: List[Dict],
    jaccard_threshold: float = 0.7,
    embedding_threshold: float = 0.8,
    combination_mode: str = "or",
    min_question_length: int = 5,
    max_question_length: int = 200,
    use_embedding: bool = False
) -> Tuple[List[Dict], List[Dict]]:
    """
    Filters a dataset of synthetic questions based on length and duplicate detection.
    
    Returns two lists:
      - accepted_data: Questions that passed all filters.
      - filtered_out_data: Questions that were filtered out, with an added key "filtered_reason"
        explaining why the question was removed.

    Parameters:
      - synthetic_data: List of dictionaries, each containing a "synthetic_question" key.
      - jaccard_threshold: Minimum Jaccard similarity to consider two questions duplicates.
      - embedding_threshold: Threshold for embedding-based cosine similarity (if used).
      - combination_mode: How to combine duplicate criteria ("or", "and", or "weighted").
      - min_question_length: Minimum allowed question length (in characters).
      - max_question_length: Maximum allowed question length (in characters).
      - use_embedding: Whether to use embedding-based similarity (default is False).

    Returns:
      A tuple with:
         (accepted_data, filtered_out_data)
    """
    accepted_data = []
    filtered_out_data = []
    accepted_questions = []  # For duplicate detection

    for item in tqdm(synthetic_data, desc="Filtering synthetic questions"):
        # Retrieve and clean the synthetic question.
        q = item.get("synthetic_question", "").strip()
        
        # Check if question is empty.
        if not q:
            item["filtered_reason"] = "Empty question"
            filtered_out_data.append(item)
            continue

        # Check length constraints.
        if len(q) < min_question_length:
            item["filtered_reason"] = f"Question too short: {len(q)} characters (min {min_question_length})"
            filtered_out_data.append(item)
            continue

        if len(q) > max_question_length:
            item["filtered_reason"] = f"Question too long: {len(q)} characters (max {max_question_length})"
            filtered_out_data.append(item)
            continue

        # Check for duplicates against accepted questions.
        if any(
            is_duplicate(q, accepted, jaccard_threshold, embedding_threshold, combination_mode, use_embedding)
            for accepted in accepted_questions
        ):
            item["filtered_reason"] = "Duplicate question"
            filtered_out_data.append(item)
            continue

        # If all checks pass, accept the question.
        accepted_questions.append(q)
        accepted_data.append(item)

    return accepted_data, filtered_out_data

# Example usage with your provided sample.
if __name__ == "__main__":
    sample_data = [
      {
        "synthetic_question": "What is the maximum weight capacity of the SummitClimber Backpack?",
        "is_grounded": False,
        "chunk_data": {
          "title": "product_info_9.md",
          "chunk": "Belt:**\n\n1. Secure the hip belt around your hips and adjust it for a comfortable fit.\n2. The belt should sit on your hips, distributing the weight of the backpack.\n\n**Sternum Strap:**\n\n1. Connect the sternum strap across your chest and adjust the height to your preference.\n2. The strap helps stabilize the backpack and prevent shoulder strap slippage.\n\n### 5. Packing and Organization\n\nThe SummitClimber Backpack offers multiple compartments and pockets for efficient packing and organization. Here are some tips for packing your backpack:\n\n- Main Compartment:\n  - Use the main compartment for larger items such as clothing, sleeping bags, or food.\n  - Pack heavier items closer to your back for better weight distribution.\n\n- Front Pocket and Side Pockets:\n  - Utilize the front zipper pocket and side mesh pockets for quick access to smaller essentials like snacks, maps, or water bottles.\n\n- Compression Straps:\n  - Use the compression straps to secure and stabilize the contents of your backpack.\n\n### 6. Backpack Care and Maintenance\n\nTo ensure the longevity and performance of your SummitClimber Backpack, follow these care and maintenance guidelines:\n\n- Clean the backpack regularly with mild soap and water.\n- Avoid using harsh chemicals or abrasive cleaners.\n- Allow the backpack to air dry completely before storing.\n- Store the backpack in a cool, dry place away from direct sunlight.\n\n### 7. Warranty Information\n\nThe SummitClimber Backpack comes with a limited warranty. Please refer to the warranty card included in the package for more information.\n\n### 8. Contact Information\n\nIf you have any questions or need further assistance, please contact our customer support:\n\n- Customer Support Phone: +1-555-123-4567\n- Customer Support Email: support@summitclimber.com\n\nWe hope you enjoy your SummitClimber Backpack and have a great outdoor experience!\n\n## Caution:\n1. **Overloading**: Avoid overloading the backpack beyond its recommended weight capacity."
        },
        "question_number": 0,
        "domain": "Technical specifications",
        "difficulty": "Intermediate",
        "tone": "Neutral",
        "language": "English",
        "question_length": 7
      }
    ]

    accepted, rejected = filter_synthetic_questions(
        synthetic_data=sample_data,
        jaccard_threshold=0.7,
        embedding_threshold=0.8,
        combination_mode="or",
        min_question_length=5,
        max_question_length=150,
        use_embedding=False
    )

    import json
    print("Accepted Questions:")
    print(json.dumps(accepted, indent=2))
    
    print("\nFiltered Out Questions:")
    print(json.dumps(rejected, indent=2))
    
    print(f"\nNumber of filtered out synthetic questions: {len(sample_data) - len(accepted)}")
