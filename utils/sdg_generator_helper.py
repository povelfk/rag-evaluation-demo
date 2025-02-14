import json
import os
import random
from tqdm import tqdm
from typing import List, Dict
import re

import numpy as np
import pandas as pd

import nltk
nltk.download('punkt')


CONFIGS_DIR = os.path.join("configs", "settings")

def load_options_from_jsonl(filename):
    """
    Loads options from a JSONL file.
    
    Each line in the file should be a JSON object. The JSON object can be:
      - a simple value (e.g., a string), or
      - a dictionary with at least an "option" key and optionally a "weight" key.
    
    Returns a tuple (options, weights) where:
      - options: a list of option values
      - weights: a list of weights (or None if all weights are default)
    """
    options = []
    weights = []
    filepath = os.path.join(CONFIGS_DIR, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # skip blank lines
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Error decoding JSON in {filename}: {e}")

            # If the entry is a dict with an "option" key, use that.
            if isinstance(entry, dict) and "option" in entry:
                options.append(entry["option"])
                # If a weight is provided, use it; otherwise assume a default weight of 1.
                weights.append(entry.get("weight", 1))
            else:
                # Otherwise assume the line itself is the option.
                options.append(entry)
                weights.append(1)
    # If all weights are 1, we can ignore weights.
    if all(weight == 1 for weight in weights):
        weights = None
    return options, weights

# Cache dictionaries to avoid reloading files each time.
_cached_options = {}

def get_cached_options(filename):
    """
    Loads and caches options from a given filename in the configs/settings folder.
    """
    if filename not in _cached_options:
        _cached_options[filename] = load_options_from_jsonl(filename)
    return _cached_options[filename]

def get_instruction(path):
    """
    Reads the content of a file (such as your system message or task instructions).
    """
    with open(path, "r", encoding="utf-8") as file:
        text = file.read()
    return text

def get_domain():
    # Uniformly sample a domain from the list of domains.
    options, weights = get_cached_options("domains.jsonl")
    return random.choices(options, weights=weights, k=1)[0] if weights else random.choice(options)

def get_tone():
    # Uniformly sample a tone from the list of tones.
    options, weights = get_cached_options("tones.jsonl")
    return random.choices(options, weights=weights, k=1)[0] if weights else random.choice(options)

def get_length_category():
    # Sample a length category from the list of length categories.
    options, weights = get_cached_options("length_categories.jsonl")
    return random.choices(options, weights=weights, k=1)[0] if weights else random.choice(options)

def get_question_length(mu=2.4, sigma=0.42, min_length=4):
    # Sample a question length from a log-normal distribution.
    length = int(np.round(np.random.lognormal(mean=mu, sigma=sigma)))
    return max(min_length, length)

def get_difficulty():
    # Weighted sample a difficulty from the list of difficulties.
    options, weights = get_cached_options("difficulties.jsonl")
    return random.choices(options, weights=weights, k=1)[0] if weights else random.choice(options)

def get_topic():
    # Uniformly sample a topic from the list of topics.
    options, weights = get_cached_options("topics.jsonl")
    return random.choices(options, weights=weights, k=1)[0] if weights else random.choice(options)

def get_language():
    # Uniformly sample a language from the list of languages.
    options, weights = get_cached_options("languages.jsonl")
    return random.choices(options, weights=weights, k=1)[0] if weights else random.choice(options)

def set_is_grounded():
    # Randomly sample whether the question is grounded or not.
    return random.choice([True, False])

    

