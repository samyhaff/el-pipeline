# Pipeline Architecture and Components

This document provides detailed documentation of the NER Pipeline architecture and all individual components.

## Table of Contents

- [Pipeline Overview](#pipeline-overview)
- [Document Loaders](#document-loaders)
- [NER Models](#ner-models)
- [Candidate Generators](#candidate-generators)
- [Rerankers](#rerankers)
- [Disambiguators](#disambiguators)
- [Knowledge Bases](#knowledge-bases)
- [Context Extraction](#context-extraction)
- [Caching System](#caching-system)

## Pipeline Overview

### Processing Flow

```
Input Files → Loader → Documents → NER → Mentions → Candidate Generator →
Candidates → Reranker → Reranked Candidates → Disambiguator → Resolved Entities → Output
```

### Stage Responsibilities

| Stage | Input | Output | Purpose |
|-------|-------|--------|---------|
| Loader | File path | Document(s) | Parse file format, extract text |
| NER | Document text | Mentions | Identify entity mentions in text |
| Candidate Generator | Mention + KB | Candidates | Find potential KB matches |
| Reranker | Candidates | Ranked Candidates | Reorder by relevance |
| Disambiguator | Candidates | Entity | Select final entity |

### Component Protocols

Each component type implements a specific protocol (interface):

```python
# Document Loader
class DocumentLoader(Protocol):
    def load(self, path: str) -> Iterator[Document]: ...

# NER Model
class NERModel(Protocol):
    def extract(self, text: str) -> List[Mention]: ...

# Candidate Generator
class CandidateGenerator(Protocol):
    def generate(self, mention: Mention, doc: Document) -> List[Candidate]: ...

# Reranker
class Reranker(Protocol):
    def rerank(self, mention: Mention, candidates: List[Candidate], doc: Document) -> List[Candidate]: ...

# Disambiguator
class Disambiguator(Protocol):
    def disambiguate(self, mention: Mention, candidates: List[Candidate], doc: Document) -> Optional[Entity]: ...

# Knowledge Base
class KnowledgeBase(Protocol):
    def get_entity(self, entity_id: str) -> Optional[Entity]: ...
    def search(self, query: str, top_k: int) -> List[Entity]: ...
    def all_entities(self) -> Iterator[Entity]: ...
```

---

## Document Loaders

Loaders parse different file formats and extract text content.

**Location:** `ner_pipeline/loaders/`

### TextLoader

Handles plain text files.

**Registration:** `text`

**Parameters:** None

**Behavior:**
- Reads UTF-8 encoded text files
- Creates a single Document per file
- Sets document ID to filename

**Example:**
```python
loader = TextLoader()
for doc in loader.load("document.txt"):
    print(doc.text)
```

### JSONLoader

Handles JSON files containing document data.

**Registration:** `json`

**Parameters:** None

**Expected Format:**
```json
{
  "id": "doc-001",
  "text": "Document content here",
  "meta": {"source": "example"}
}
```

### JSONLLoader

Handles JSON Lines files with multiple documents.

**Registration:** `jsonl`

**Parameters:** None

**Expected Format (one JSON object per line):**
```jsonl
{"id": "doc-001", "text": "First document"}
{"id": "doc-002", "text": "Second document"}
```

### PDFLoader

Extracts text from PDF files.

**Registration:** `pdf`

**Parameters:** None

**Dependencies:** `pdfplumber`

**Behavior:**
- Extracts text from all pages
- Concatenates with newlines
- Handles multi-column layouts

### DocxLoader

Extracts text from Microsoft Word documents.

**Registration:** `docx`

**Parameters:** None

**Dependencies:** `python-docx`

**Behavior:**
- Extracts text from paragraphs
- Preserves paragraph structure
- Ignores formatting

### HTMLLoader

Extracts text from HTML pages.

**Registration:** `html`

**Parameters:** None

**Dependencies:** `beautifulsoup4`, `lxml`

**Behavior:**
- Removes script and style elements
- Extracts visible text content
- Preserves basic structure

---

## NER Models

NER models identify entity mentions in text.

**Location:** `ner_pipeline/ner/`

### SimpleRegexNER

Lightweight regex-based NER.

**Registration:** `simple`

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_len` | int | 3 | Minimum mention length in characters |
| `context_mode` | str | "sentence" | Context extraction mode |

**Behavior:**
- Uses regex: `\b([A-Z][a-zA-Z0-9_-]+(?:\s+[A-Z][a-zA-Z0-9_-]+)*)\b`
- Finds sequences of capitalized words
- Labels all mentions as "ENT"
- Fast, no model downloads required

**Example Matches:**
- "Albert Einstein" ✓
- "New York City" ✓
- "the" ✗ (lowercase)
- "AI" ✓ (but may be filtered by min_len)

### SpacyNER

spaCy-based NER using pre-trained models.

**Registration:** `spacy`

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | "en_core_web_sm" | spaCy model name |
| `context_mode` | str | "sentence" | Context extraction mode |

**Available Labels:**
- PERSON: People, including fictional
- NORP: Nationalities, religious/political groups
- ORG: Companies, agencies, institutions
- GPE: Countries, cities, states
- LOC: Non-GPE locations
- FAC: Buildings, airports, highways
- PRODUCT: Objects, vehicles, foods
- EVENT: Named hurricanes, battles, wars
- WORK_OF_ART: Titles of books, songs
- LAW: Named documents made into laws
- LANGUAGE: Any named language
- DATE, TIME, PERCENT, MONEY, QUANTITY, ORDINAL, CARDINAL

**Installation:**
```bash
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_md  # Medium
python -m spacy download en_core_web_lg  # Large
```

### GLiNERNER

Zero-shot NER using GLiNER model.

**Registration:** `gliner`

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | str | "urchade/gliner_large" | GLiNER model |
| `labels` | List[str] | ["person", "organization", "location"] | Entity types to detect |
| `threshold` | float | 0.3 | Detection threshold |
| `context_mode` | str | "sentence" | Context extraction mode |

**Features:**
- Arbitrary entity labels (no training required)
- Threshold controls precision/recall tradeoff
- Requires GPU for reasonable performance

**Example:**
```python
ner = GLiNERNER(
    labels=["company", "product", "technology"],
    threshold=0.4
)
```

### TransformersNER

HuggingFace transformers-based NER.

**Registration:** `transformers`

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | str | "dslim/bert-base-NER" | HuggingFace model |
| `context_mode` | str | "sentence" | Context extraction mode |

**Popular Models:**
- `dslim/bert-base-NER`: General English NER
- `dbmdz/bert-large-cased-finetuned-conll03-english`: CoNLL-2003
- `Jean-Baptiste/roberta-large-ner-english`: RoBERTa NER

### LELAGLiNERNER

LELA-optimized GLiNER configuration.

**Registration:** `lela_gliner`

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | str | "numind/NuNER_Zero-span" | LELA default model |
| `labels` | List[str] | LELA defaults | Entity types |
| `threshold` | float | 0.5 | Detection threshold |
| `context_mode` | str | "sentence" | Context extraction mode |

**LELA Default Labels:**
- person
- organization
- location
- event
- work of art
- product

**Features:**
- Optimized defaults for entity linking
- Detailed debug logging
- Lazy model loading

---

## Candidate Generators

Candidate generators find potential KB matches for mentions.

**Location:** `ner_pipeline/candidates/`

### FuzzyCandidateGenerator

Fast string matching using RapidFuzz.

**Registration:** `fuzzy`

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `top_k` | int | 20 | Maximum candidates to return |

**Algorithm:**
- Uses `rapidfuzz.process.extract()`
- Matches mention text against entity titles
- Returns scores 0-100 (normalized to 0-1)

**Use Cases:**
- Fast baseline
- Exact or near-exact matches
- No semantic understanding

### BM25CandidateGenerator

BM25-based retrieval on entity descriptions.

**Registration:** `bm25`

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `top_k` | int | 20 | Maximum candidates |
| `use_context` | bool | False | Include mention context in query |

**Algorithm:**
- Uses `rank-bm25.BM25Okapi`
- Tokenizes with simple split()
- Scores based on term frequency and document frequency

**Indexing:**
- Indexes entity descriptions at initialization
- Query: mention text (optionally + context)

### DenseCandidateGenerator

Dense retrieval using sentence embeddings and FAISS.

**Registration:** `dense`

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | str | "all-MiniLM-L6-v2" | Embedding model |
| `top_k` | int | 20 | Maximum candidates |
| `use_context` | bool | False | Include mention context |

**Algorithm:**
- Uses `sentence-transformers` for embeddings
- FAISS IndexFlatIP for similarity search
- L2 normalization for cosine similarity

**Performance:**
- Requires initial embedding computation
- Fast at query time with FAISS
- GPU acceleration available

### LELABM25CandidateGenerator

LELA-optimized BM25 using bm25s library.

**Registration:** `lela_bm25`

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `top_k` | int | 64 | Maximum candidates |
| `use_context` | bool | True | Include mention context |
| `stemmer_language` | str | "english" | Stemmer language |

**Features:**
- Uses `bm25s` with numba backend (faster)
- PyStemmer for language-specific stemming
- Handles empty tokenization gracefully

### LELADenseCandidateGenerator

LELA dense retrieval using OpenAI-compatible API.

**Registration:** `lela_dense`

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | str | LELA default | Embedding model |
| `top_k` | int | 64 | Maximum candidates |
| `base_url` | str | "http://localhost" | API endpoint |
| `port` | int | 8000 | API port |
| `use_context` | bool | True | Include mention context |

**Query Format:**
```
Instruct: {RETRIEVER_TASK}
Query: {mention_text}
```

**Requirements:**
- Running vLLM or compatible embedding server
- OpenAI API client

---

## Rerankers

Rerankers reorder candidates by relevance.

**Location:** `ner_pipeline/rerankers/`

### NoOpReranker

Passthrough reranker that returns candidates unchanged.

**Registration:** `none`

**Parameters:** None

**Use Cases:**
- Baseline
- When candidate generator ordering is sufficient
- Performance-critical applications

### CrossEncoderReranker

Cross-encoder reranking using sentence-transformers.

**Registration:** `cross_encoder`

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | str | "cross-encoder/ms-marco-MiniLM-L-6-v2" | Model name |
| `top_k` | int | 10 | Candidates to keep |

**Algorithm:**
- Creates pairs: (mention context, candidate description)
- Scores each pair with cross-encoder
- Sorts by score and returns top_k

**Performance:**
- Slower than bi-encoder (processes pairs)
- Higher accuracy for reranking
- GPU recommended

### LELAEmbedderReranker

LELA reranking using cosine similarity with marked mentions.

**Registration:** `lela_embedder`

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | str | LELA default | Embedding model |
| `top_k` | int | 10 | Candidates to keep |
| `base_url` | str | "http://localhost" | API endpoint |
| `port` | int | 8000 | API port |

**Query Format:**
```
Instruct: {RERANKER_TASK}
Query: {text with [marked] mention}
```

**Features:**
- Marks mention with `[` and `]` in full text
- Uses task-specific instruction
- Cosine similarity scoring

---

## Disambiguators

Disambiguators select the final entity from candidates.

**Location:** `ner_pipeline/disambiguators/`

### FirstCandidateDisambiguator

Selects the first candidate in the list.

**Registration:** `first`

**Parameters:** None

**Use Cases:**
- Baseline
- When candidates are well-ordered
- Fast processing

### PopularityDisambiguator

Selects the candidate with the highest score.

**Registration:** `popularity`

**Parameters:** None

**Algorithm:**
- Returns candidate with max score
- Falls back to 0.0 for None scores
- Simple but effective baseline

### LLMDisambiguator

HuggingFace zero-shot classification for disambiguation.

**Registration:** `llm`

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | str | "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli" | Model |

**Algorithm:**
- Uses NLI (Natural Language Inference)
- Creates hypothesis from candidate descriptions
- Selects by entailment score

### LELAvLLMDisambiguator

LELA disambiguation using vLLM for fast batched inference.

**Registration:** `lela_vllm`

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | str | "Qwen/Qwen3-8B" | LLM model |
| `tensor_parallel_size` | int | 1 | GPU parallelism |
| `max_model_len` | int | None | Max context length |
| `add_none_candidate` | bool | True | Add "None" option |
| `add_descriptions` | bool | True | Include descriptions |
| `disable_thinking` | bool | False | Disable reasoning |
| `system_prompt` | str | LELA default | Custom system prompt |
| `generation_config` | Dict | {} | vLLM generation settings |
| `self_consistency_k` | int | 1 | Voting samples |

**Prompt Format:**
```
System: You are an expert designed to disambiguate entities in text...

User: Input: "France hosted the 2024 Olympics in [Paris]."

Candidates:
0. None of the candidates
1. Paris (city): Capital city of France
2. Paris (novel): 1897 novel by Emile Zola
...
```

**Features:**
- Tournament-style disambiguation (from LELA methodology)
- Self-consistency voting with multiple samples
- Structured output parsing
- Thinking mode control for reasoning models

---

## Knowledge Bases

Knowledge bases store entity information.

**Location:** `ner_pipeline/knowledge_bases/`

### CustomJSONLKnowledgeBase

Custom JSONL format knowledge base.

**Registration:** `custom`

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | str | Required | Path to JSONL file |

**Format:**
```jsonl
{"id": "Q937", "title": "Albert Einstein", "description": "Theoretical physicist"}
{"id": "Q183", "title": "Germany", "description": "Country in Central Europe"}
```

**Methods:**
- `get_entity(id)`: Retrieve by ID
- `search(query, top_k)`: Fuzzy search on titles
- `all_entities()`: Iterate all entities

### LELAJSONLKnowledgeBase

LELA-format JSONL knowledge base.

**Registration:** `lela_jsonl`

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | str | Required | Path to JSONL file |
| `title_field` | str | "title" | Field name for title |
| `description_field` | str | "description" | Field name for description |

**Format:**
```jsonl
{"title": "Albert Einstein", "description": "Theoretical physicist"}
{"title": "Germany", "description": "Country in Central Europe"}
```

**Note:** Uses title as entity ID (LELA convention).

**Additional Methods:**
- `get_descriptions_dict()`: Returns {title: description} mapping

### WikipediaKB

Wikipedia API-based knowledge base.

**Registration:** `wikipedia`

**Parameters:** None

**Features:**
- Live Wikipedia lookups
- Uses wptools library
- Returns Wikipedia article summaries

### WikidataKB

Wikidata SPARQL-based knowledge base.

**Registration:** `wikidata`

**Parameters:** None

**Features:**
- SPARQL queries to Wikidata
- Returns structured entity data
- Includes Wikidata IDs (Q-numbers)

---

## Context Extraction

Context extraction utilities for mention context.

**Location:** `ner_pipeline/context.py`

### Sentence Mode

Extracts complete sentences containing the mention.

```python
extract_sentence_context(text, start, end, max_sentences=1)
```

**Algorithm:**
- Splits on `.!?` and paragraph breaks
- Finds sentence containing mention
- Returns surrounding sentences

### Window Mode

Extracts fixed character window around mention.

```python
extract_window_context(text, start, end, window_chars=150)
```

**Algorithm:**
- Takes chars before and after mention
- Aligns to word boundaries
- Cleaner for embedding models

### General Function

```python
extract_context(text, start, end, mode="sentence", **kwargs)
```

**Modes:**
- `"sentence"`: Use sentence extraction
- `"window"`: Use window extraction

---

## Caching System

The pipeline includes document caching for efficiency.

### Cache Key Generation

```python
key = SHA256(f"{path}-{mtime}-{size}".encode())
```

### Cache Location

Default: `.ner_cache/` (configurable via `cache_dir`)

### Cache Files

Format: `.ner_cache/{key}.pkl`
Storage: Python pickle

### Cache Behavior

1. Compute hash from file path, modification time, size
2. Check if cache file exists
3. If exists: load cached Document list
4. If not: load from disk, cache, return

### Cache Invalidation

- Automatic on file modification (mtime change)
- Automatic on file size change
- Manual: delete `.ner_cache/` directory

### Disabling Cache

Set `cache_dir` to `None` in configuration.
