# AI RAG Evaluation

This repository contains a collection of Jupyter notebooks for evaluating a **RAG system**, with a focus on synthetic data generation and LLM evaluation.

## Usage

Work through the notebooks in numerical order to understand and implement different aspects of the RAG evaluation workflow:

1. Start with `00-intro-to-ai-evals.ipynb` for an overview
2. Follow the synthetic data generation and evaluation flow (notebooks 01-03)
3. Explore retrieval evaluation in notebook 04
4. Build and evaluate LLM judges (notebooks 05-07)
5. Evaluate your RAG system - upload results to AI Foundry in notebook 08

## Project Structure

### Notebooks
1. `00-intro-to-ai-evals.ipynb` - Introduction to RAG evaluation concepts
2. `01-generate-synthetic-data.ipynb` - Generate synthetic test data
3. `02-evaluate-synthetic-data.ipynb` - Evaluate quality of synthetic data
4. `03-prepare-synthetic-data.ipynb` - Prepare and format synthetic data
5. `04-evaluate-retrieval.ipynb` - Evaluate retrieval performance
6. `05-build-llm-judge-pe.ipynb` - Build LLM-based judge (prompt engineering)
7. `06-train-llm-judge-ft.ipynb` - Train LLM-based judge (fine-tuning)
8. `07-evalaute-llm-judge-ft.ipynb` - Evaluate fine-tuned LLM judge
9. `08-evaluate-RAG-on-azure.ipynb` - Evaluate your RAG system on Azure

### Directory Structure
- `configs/` - Configuration files
  - `prompts/` - Prompt templates
  - `settings/` - System settings
- `data/` - Data storage
  - `agent-output/` - AI agent evaluation outputs
  - `ft-judge/` - Fine-tuned judge data
  - `ft-sdg/` - Fine-tuned synthetic data generation
  - `index/` - Index files for retrieval
- `evaluators/` - Evaluation modules
  - `aoai/` - Azure OpenAI evaluators
- `media/` - Images and media files
- `utils/` - Utility functions and helper scripts

## Setup

1. Clone this repository
2. Install the required dependencies
3. Copy `sample.env` to `.env` and fill in your credentials 

## Requirements

See `requirements.txt` for the full list of dependencies.
