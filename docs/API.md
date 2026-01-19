# Python API Reference

This document provides a comprehensive reference for the NER Pipeline Python API.

## Table of Contents

- [Core Classes](#core-classes)
- [Data Types](#data-types)
- [Configuration](#configuration)
- [Registry System](#registry-system)
- [Context Extraction](#context-extraction)
- [Usage Examples](#usage-examples)

## Core Classes

### NERPipeline

The main orchestrator class that manages the entire entity linking pipeline.

**Location:** `ner_pipeline/pipeline.py`

```python
from ner_pipeline.pipeline import NERPipeline
from ner_pipeline.config import PipelineConfig
```

#### Constructor

```python
NERPipeline(config: PipelineConfig)
```

**Parameters:**
- `config`: A `PipelineConfig` object containing pipeline configuration

**Initialization:**
- Instantiates all pipeline components (loader, NER, candidate generator, reranker, disambiguator, knowledge base)
- Sets up caching directory
- Configures batch processing

#### Methods

##### `process_document(doc: Document) -> Dict`

Process a single document through the entire pipeline.

**Parameters:**
- `doc`: A `Document` object to process

**Returns:**
```python
{
    "id": str,           # Document ID
    "text": str,         # Full document text
    "entities": List[{   # List of resolved entities
        "text": str,              # Mention text
        "start": int,             # Start character position
        "end": int,               # End character position
        "label": str,             # Entity type label
        "context": str,           # Surrounding context
        "entity_id": Optional[str],    # Resolved entity ID
        "entity_title": Optional[str], # Entity title
        "entity_description": Optional[str],  # Entity description
        "candidates": List[{      # Candidate list
            "entity_id": str,
            "score": float,
            "description": str
        }]
    }],
    "meta": Dict         # Document metadata
}
```

##### `run(paths: Iterable[str], output_path: Optional[str] = None) -> List[Dict]`

Process multiple files through the pipeline.

**Parameters:**
- `paths`: Iterable of file paths to process
- `output_path`: Optional path to write JSONL output

**Returns:**
- List of result dictionaries (same format as `process_document`)

**Example:**
```python
pipeline = NERPipeline(config)
results = pipeline.run(
    ["doc1.txt", "doc2.pdf", "doc3.html"],
    output_path="results.jsonl"
)
```

### PipelineConfig

Configuration dataclass for pipeline setup.

**Location:** `ner_pipeline/config.py`

```python
from ner_pipeline.config import PipelineConfig
```

#### Class Methods

##### `from_dict(config_dict: Dict) -> PipelineConfig`

Create a configuration from a dictionary.

**Parameters:**
- `config_dict`: Dictionary with configuration options

**Example:**
```python
config_dict = {
    "loader": {"name": "text"},
    "ner": {"name": "simple", "params": {"min_len": 3}},
    "candidate_generator": {"name": "fuzzy", "params": {"top_k": 10}},
    "reranker": {"name": "none"},
    "disambiguator": {"name": "first"},
    "knowledge_base": {"name": "custom", "params": {"path": "kb.jsonl"}},
    "cache_dir": ".ner_cache",
    "batch_size": 1
}
config = PipelineConfig.from_dict(config_dict)
```

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `loader` | Dict | Loader configuration |
| `ner` | Dict | NER model configuration |
| `candidate_generator` | Dict | Candidate generator configuration |
| `reranker` | Dict | Reranker configuration |
| `disambiguator` | Dict | Disambiguator configuration |
| `knowledge_base` | Dict | Knowledge base configuration |
| `cache_dir` | str | Directory for document caching |
| `batch_size` | int | Batch size for processing |

## Data Types

All core data types are defined in `ner_pipeline/types.py`.

### Document

Represents an input document.

```python
from ner_pipeline.types import Document

doc = Document(
    id="doc-001",
    text="Albert Einstein was born in Germany.",
    meta={"source": "wikipedia", "date": "2024-01-01"}
)
```

**Attributes:**
| Attribute | Type | Description |
|-----------|------|-------------|
| `id` | str | Unique document identifier |
| `text` | str | Document text content |
| `meta` | Dict | Optional metadata dictionary |

### Mention

Represents a raw NER output - an entity mention in text.

```python
from ner_pipeline.types import Mention

mention = Mention(
    start=0,
    end=15,
    text="Albert Einstein",
    label="PERSON",
    context="Albert Einstein was born in Germany."
)
```

**Attributes:**
| Attribute | Type | Description |
|-----------|------|-------------|
| `start` | int | Start character position in text |
| `end` | int | End character position in text |
| `text` | str | The mention text |
| `label` | str | Entity type label (e.g., "PERSON", "ORG") |
| `context` | Optional[str] | Surrounding context |

### Entity

Represents an entity in the knowledge base.

```python
from ner_pipeline.types import Entity

entity = Entity(
    id="Q937",
    title="Albert Einstein",
    description="German-born theoretical physicist",
    meta={"birth_year": 1879}
)
```

**Attributes:**
| Attribute | Type | Description |
|-----------|------|-------------|
| `id` | str | Unique entity identifier |
| `title` | str | Entity name/title |
| `description` | str | Entity description |
| `meta` | Dict | Optional metadata |

### Candidate

Represents a potential KB match for a mention.

```python
from ner_pipeline.types import Candidate

candidate = Candidate(
    entity_id="Q937",
    score=0.95,
    description="German-born theoretical physicist"
)
```

**Attributes:**
| Attribute | Type | Description |
|-----------|------|-------------|
| `entity_id` | str | Entity identifier in KB |
| `score` | float | Relevance score |
| `description` | str | Entity description |

### ResolvedMention

Represents a mention with its resolved entity and candidates.

```python
from ner_pipeline.types import ResolvedMention

resolved = ResolvedMention(
    mention=mention,
    entity=entity,  # Optional - resolved entity
    candidates=[candidate1, candidate2, candidate3]
)
```

**Attributes:**
| Attribute | Type | Description |
|-----------|------|-------------|
| `mention` | Mention | The original mention |
| `entity` | Optional[Entity] | The resolved entity |
| `candidates` | List[Candidate] | List of candidates |

## Configuration

### Configuration Format

Pipeline configuration uses JSON format with the following structure:

```json
{
  "loader": {
    "name": "<loader_name>",
    "params": {}
  },
  "ner": {
    "name": "<ner_name>",
    "params": {}
  },
  "candidate_generator": {
    "name": "<generator_name>",
    "params": {}
  },
  "reranker": {
    "name": "<reranker_name>",
    "params": {}
  },
  "disambiguator": {
    "name": "<disambiguator_name>",
    "params": {}
  },
  "knowledge_base": {
    "name": "<kb_name>",
    "params": {}
  },
  "cache_dir": ".ner_cache",
  "batch_size": 1
}
```

### Component Options

#### Loaders
| Name | Parameters | Description |
|------|------------|-------------|
| `text` | - | Plain text files |
| `pdf` | - | PDF documents |
| `docx` | - | Word documents |
| `html` | - | HTML pages |
| `json` | - | JSON files |
| `jsonl` | - | JSON Lines files |

#### NER Models
| Name | Parameters | Description |
|------|------------|-------------|
| `simple` | `min_len`, `context_mode` | Regex-based NER |
| `spacy` | `model`, `context_mode` | spaCy NER |
| `gliner` | `model_name`, `labels`, `threshold`, `context_mode` | GLiNER zero-shot NER |
| `transformers` | `model_name`, `context_mode` | HuggingFace transformers NER |
| `lela_gliner` | `model_name`, `labels`, `threshold`, `context_mode` | LELA GLiNER with defaults |

#### Candidate Generators
| Name | Parameters | Description |
|------|------------|-------------|
| `fuzzy` | `top_k` | RapidFuzz string matching |
| `bm25` | `top_k`, `use_context` | BM25 retrieval |
| `dense` | `model_name`, `top_k`, `use_context` | Dense retrieval with FAISS |
| `lela_bm25` | `top_k`, `use_context`, `stemmer_language` | bm25s with stemming |
| `lela_dense` | `model_name`, `top_k`, `base_url`, `port`, `use_context` | OpenAI-compatible embeddings |

#### Rerankers
| Name | Parameters | Description |
|------|------------|-------------|
| `none` | - | No reranking |
| `cross_encoder` | `model_name`, `top_k` | Cross-encoder reranking |
| `lela_embedder` | `model_name`, `top_k`, `base_url`, `port` | LELA embedding reranker |

#### Disambiguators
| Name | Parameters | Description |
|------|------------|-------------|
| `first` | - | Select first candidate |
| `popularity` | - | Select by highest score |
| `llm` | `model_name` | HuggingFace zero-shot classification |
| `lela_vllm` | `model_name`, `tensor_parallel_size`, `max_model_len`, ... | vLLM-based disambiguation |

#### Knowledge Bases
| Name | Parameters | Description |
|------|------------|-------------|
| `custom` | `path` | Custom JSONL KB |
| `lela_jsonl` | `path`, `title_field`, `description_field` | LELA-format JSONL KB |
| `wikipedia` | - | Wikipedia API |
| `wikidata` | - | Wikidata SPARQL |

## Registry System

Components are registered using a decorator-based registry pattern.

**Location:** `ner_pipeline/registry.py`

### Available Registries

```python
from ner_pipeline.registry import (
    loaders,
    ner_models,
    candidate_generators,
    rerankers,
    disambiguators,
    knowledge_bases
)
```

### Registering Custom Components

```python
from ner_pipeline.registry import ner_models

@ner_models.register("my_custom_ner")
class MyCustomNER:
    def __init__(self, param1: str = "default"):
        self.param1 = param1

    def extract(self, text: str) -> List[Mention]:
        # Implementation
        return mentions
```

### Retrieving Components

```python
# Get factory function
factory = ner_models.get("my_custom_ner")

# Instantiate with parameters
ner = factory(param1="custom_value")
```

## Context Extraction

Utilities for extracting context around mentions.

**Location:** `ner_pipeline/context.py`

### Functions

#### `extract_sentence_context(text, start, end, max_sentences=1)`

Extract surrounding sentences containing the mention.

```python
from ner_pipeline.context import extract_sentence_context

text = "First sentence. Albert Einstein was born in Germany. Third sentence."
context = extract_sentence_context(text, start=16, end=31, max_sentences=1)
# Returns: "Albert Einstein was born in Germany."
```

#### `extract_window_context(text, start, end, window_chars=150)`

Extract a fixed character window around the mention.

```python
from ner_pipeline.context import extract_window_context

context = extract_window_context(text, start=16, end=31, window_chars=100)
```

#### `extract_context(text, start, end, mode="sentence", **kwargs)`

General dispatcher for context extraction.

```python
from ner_pipeline.context import extract_context

# Sentence mode
context = extract_context(text, 16, 31, mode="sentence", max_sentences=2)

# Window mode
context = extract_context(text, 16, 31, mode="window", window_chars=150)
```

## Usage Examples

### Basic Pipeline Usage

```python
from ner_pipeline.config import PipelineConfig
from ner_pipeline.pipeline import NERPipeline
from ner_pipeline.types import Document
import json

# Load configuration
with open("config.json") as f:
    config = PipelineConfig.from_dict(json.load(f))

# Create pipeline
pipeline = NERPipeline(config)

# Process single document
doc = Document(
    id="test-doc",
    text="Albert Einstein was born in Germany and later moved to the United States.",
    meta={}
)
result = pipeline.process_document(doc)

# Print results
for entity in result["entities"]:
    print(f"Mention: {entity['text']}")
    print(f"  Label: {entity['label']}")
    print(f"  Resolved to: {entity.get('entity_title', 'N/A')}")
    print(f"  Candidates: {len(entity['candidates'])}")
```

### Processing Multiple Files

```python
# Process multiple files with output
results = pipeline.run(
    paths=["doc1.txt", "doc2.pdf", "doc3.html"],
    output_path="output/results.jsonl"
)

# Results are also returned
for result in results:
    print(f"Document {result['id']}: {len(result['entities'])} entities")
```

### Custom Component Integration

```python
from ner_pipeline.registry import ner_models
from ner_pipeline.types import Mention
from typing import List

@ner_models.register("custom_ner")
class CustomNER:
    def __init__(self, custom_param: str = "default"):
        self.custom_param = custom_param

    def extract(self, text: str) -> List[Mention]:
        # Custom NER logic
        mentions = []
        # ... implementation ...
        return mentions

# Use in configuration
config_dict = {
    "ner": {"name": "custom_ner", "params": {"custom_param": "value"}},
    # ... rest of config
}
```

### Working with Knowledge Bases

```python
from ner_pipeline.knowledge_bases.custom import CustomJSONLKnowledgeBase

# Load knowledge base
kb = CustomJSONLKnowledgeBase(path="entities.jsonl")

# Get entity by ID
entity = kb.get_entity("Q937")

# Search entities
results = kb.search("Einstein", top_k=10)

# Iterate all entities
for entity in kb.all_entities():
    print(f"{entity.id}: {entity.title}")
```

### LELA-Specific Configuration

```python
config_dict = {
    "loader": {"name": "text"},
    "ner": {
        "name": "lela_gliner",
        "params": {
            "model_name": "numind/NuNER_Zero-span",
            "labels": ["person", "organization", "location"],
            "threshold": 0.5
        }
    },
    "candidate_generator": {
        "name": "lela_bm25",
        "params": {"top_k": 64, "use_context": True}
    },
    "reranker": {
        "name": "lela_embedder",
        "params": {
            "model_name": "Qwen/Qwen3-Embedding-4B",
            "top_k": 10,
            "base_url": "http://localhost",
            "port": 8000
        }
    },
    "disambiguator": {
        "name": "lela_vllm",
        "params": {
            "model_name": "Qwen/Qwen3-8B",
            "tensor_parallel_size": 1,
            "add_none_candidate": True
        }
    },
    "knowledge_base": {
        "name": "lela_jsonl",
        "params": {"path": "entities.jsonl"}
    }
}
```

## Output Format

The pipeline outputs JSONL (JSON Lines) format, with one JSON object per line representing each processed document:

```json
{
  "id": "doc-001",
  "text": "Albert Einstein was born in Germany.",
  "entities": [
    {
      "text": "Albert Einstein",
      "start": 0,
      "end": 15,
      "label": "PERSON",
      "context": "Albert Einstein was born in Germany.",
      "entity_id": "Q937",
      "entity_title": "Albert Einstein",
      "entity_description": "German-born theoretical physicist",
      "candidates": [
        {
          "entity_id": "Q937",
          "score": 0.95,
          "description": "German-born theoretical physicist"
        },
        {
          "entity_id": "Q1234",
          "score": 0.45,
          "description": "Another Einstein"
        }
      ]
    },
    {
      "text": "Germany",
      "start": 28,
      "end": 35,
      "label": "GPE",
      "context": "Albert Einstein was born in Germany.",
      "entity_id": "Q183",
      "entity_title": "Germany",
      "entity_description": "Country in Central Europe",
      "candidates": [...]
    }
  ],
  "meta": {
    "source": "wikipedia"
  }
}
```
