# NER Pipeline Documentation

A modular, swappable Named Entity Recognition (NER) and Entity Linking pipeline implemented in Python. This project provides a complete solution for extracting named entities from documents and linking them to entities in a knowledge base.

## Table of Contents

- [Overview](#overview)
- [Project Purpose](#project-purpose)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Documentation Index](#documentation-index)

## Overview

The NER Pipeline is a configurable system that processes documents through multiple stages to identify named entities and resolve them to a knowledge base. It features:

- **Modular Architecture**: Each pipeline stage is pluggable and can be swapped independently
- **Multiple Interfaces**: CLI, Python API, and Web UI (Gradio)
- **LELA Integration**: Advanced entity linking components based on the LELA research methodology
- **Flexible Input**: Supports text, PDF, DOCX, HTML, JSON, and JSONL documents
- **Caching**: Intelligent document caching for efficient reprocessing

## Project Purpose

This pipeline addresses the entity linking task: mapping ambiguous mentions of entities in natural language text to reference entities in a knowledge base. For example, given the text:

> "France hosted the 2024 Olympics in Paris."

The pipeline identifies "Paris" as a mention and links it to the correct entity (Paris, the capital city of France) from candidates like "Paris (city)", "Paris (novel)", or "Paris (Texas)".

### Key Use Cases

1. **Knowledge Graph Construction**: Extract and link entities to build structured knowledge
2. **Information Extraction**: Identify and disambiguate entities in documents
3. **Question Answering**: Provide entity context for QA systems
4. **Domain-Specific Entity Linking**: Adapt to custom knowledge bases without fine-tuning

## Architecture

The pipeline follows a multi-stage processing architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│                        Input Document                            │
│              (text, PDF, DOCX, HTML, JSON, JSONL)               │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                         LOADER                                   │
│     Parse document format and extract text content               │
│     Implementations: text, pdf, docx, html, json, jsonl          │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                        NER MODEL                                 │
│          Extract named entity mentions from text                 │
│  Implementations: simple, spacy, gliner, transformers,          │
│                   lela_gliner                                    │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                   CANDIDATE GENERATOR                            │
│        Find potential KB matches for each mention                │
│   Implementations: fuzzy, bm25, dense, lela_bm25, lela_dense    │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                       RERANKER                                   │
│         Reorder candidates by relevance (optional)               │
│    Implementations: none, cross_encoder, lela_embedder          │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     DISAMBIGUATOR                                │
│        Select the final entity from candidates                   │
│   Implementations: first, popularity, llm, lela_vllm            │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                         OUTPUT                                   │
│              JSONL with resolved entities                        │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
ner-pipeline/
├── ner_pipeline/              # Main Python package
│   ├── __init__.py            # Package exports
│   ├── types.py               # Data models (Document, Mention, Entity, etc.)
│   ├── config.py              # PipelineConfig for configuration parsing
│   ├── registry.py            # Component registries
│   ├── pipeline.py            # Main NERPipeline orchestrator
│   ├── context.py             # Context extraction utilities
│   ├── cli.py                 # CLI entry point
│   │
│   ├── loaders/               # Document input handlers
│   ├── ner/                   # NER implementations
│   ├── candidates/            # Candidate generation
│   ├── rerankers/             # Candidate reranking
│   ├── disambiguators/        # Final entity selection
│   ├── knowledge_bases/       # Entity knowledge bases
│   ├── lela/                  # LELA integration module
│   └── scripts/               # Utility scripts
│
├── app.py                     # Gradio web UI
├── tests/                     # Test suite (unit, integration, slow)
├── config/                    # Configuration examples
├── data/                      # Sample data and test files
├── docs/                      # Documentation
├── requirements.txt           # Dependencies
└── pyproject.toml             # Project metadata
```

## Quick Start

### Installation

```bash
# Clone the repository
cd ner-pipeline

# Install dependencies
pip install -r requirements.txt

# For spaCy models
python -m spacy download en_core_web_sm
```

### Basic Usage

**Using the CLI:**
```bash
python -m ner_pipeline.cli \
  --config config/lela_bm25_only.json \
  --input document.txt \
  --output results.jsonl
```

**Using the Python API:**
```python
from ner_pipeline.config import PipelineConfig
from ner_pipeline.pipeline import NERPipeline
import json

# Load configuration
with open("config.json") as f:
    config = PipelineConfig.from_dict(json.load(f))

# Create and run pipeline
pipeline = NERPipeline(config)
results = pipeline.run(["document.txt"], output_path="results.jsonl")
```

**Using the Web UI:**
```bash
python app.py --port 7860
# Open http://localhost:7860 in your browser
```

### Minimal Configuration

```json
{
  "loader": {"name": "text"},
  "ner": {"name": "simple", "params": {"min_len": 3}},
  "candidate_generator": {"name": "fuzzy", "params": {"top_k": 10}},
  "reranker": {"name": "none"},
  "disambiguator": {"name": "first"},
  "knowledge_base": {"name": "custom", "params": {"path": "kb.jsonl"}}
}
```

## Documentation Index

| Document | Description |
|----------|-------------|
| [API.md](API.md) | Python API reference, data types, and programmatic usage |
| [CLI.md](CLI.md) | Command-line interface documentation |
| [WEB_APP.md](WEB_APP.md) | Gradio web application guide |
| [PIPELINE.md](PIPELINE.md) | Detailed pipeline architecture and component documentation |
| [LELA.md](LELA.md) | LELA methodology and integration |

## Requirements

- Python 3.9+
- PyTorch 2.6.0+
- See `requirements.txt` for full dependency list

### Optional Dependencies

- **GPU Support**: CUDA 11.8+ for GPU acceleration
- **vLLM**: Required for LELA vLLM disambiguator
- **spaCy Models**: Required for spaCy NER
