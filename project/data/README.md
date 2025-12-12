# Data Directory

This directory contains datasets and documents for the Research Assistant system.

## Current Datasets

### Anthropic HH-RLHF Dataset
**Source**: https://huggingface.co/datasets/Anthropic/hh-rlhf

**Description**:
- Human preference data about helpfulness and harmlessness
- Contains ~170K human-annotated conversations
- Useful for training and evaluating conversational AI systems
- Red-teaming data for safety research

**Usage in Week 7**:
This dataset will be used to demonstrate the RAG (Retrieval-Augmented Generation) system:
1. Load conversations from the dataset
2. Create embeddings and store in vector database (pgvector)
3. Implement similarity search for relevant conversations
4. Use retrieved context to improve research assistant responses

## Setup Instructions

### Option 1: Download via Hugging Face Datasets

```bash
pip install datasets
```

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("Anthropic/hh-rlhf")

# Save locally
dataset.save_to_disk("data/hh-rlhf")
```

### Option 2: Use the provided script

We'll create a script to download and prepare the dataset in Week 7.

## Directory Structure

```
data/
├── README.md              # This file
├── hh-rlhf/              # Anthropic HH-RLHF dataset (Week 7)
├── research_papers/       # Research papers for RAG (Week 7)
└── custom_documents/      # Your own documents (Week 7)
```

## Week 7 Preview

In Week 7, we'll:
1. Download and process the Anthropic HH-RLHF dataset
2. Create a document processing pipeline
3. Set up pgvector for vector storage
4. Implement embedding generation
5. Build the RAG retrieval system
6. Integrate with the Research Assistant

## Notes

- The dataset is large (~170K conversations)
- We may use a subset for faster experimentation
- Make sure you have enough disk space (~500MB-1GB)
