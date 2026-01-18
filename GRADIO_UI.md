# Gradio UI for NER Pipeline

Interactive web interface for the modular NER pipeline with component swapping and parameter tuning.

## Features

- 📝 **Dual Input**: Text input or file upload (txt, pdf, docx, html)
- 🔧 **Modular Components**: Swap NER, candidate generation, reranking, and disambiguation
- ⚙️ **Dynamic Parameters**: Component-specific settings appear based on selection
- 📊 **Visual Output**: Highlighted entities with labels and full JSON results
- 🎯 **Knowledge Base**: Upload custom JSONL knowledge bases

## Installation

```bash
cd ner-pipeline
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# If using SpaCy NER
python -m spacy download en_core_web_sm
```

## Quick Start

```bash
python app.py
```

Then open your browser to `http://localhost:7860`

### Command Line Options

```bash
python app.py --port 8080              # Custom port
python app.py --share                  # Create public share link
python app.py --log DEBUG              # Set logging level
```

## Usage

### 1. Upload Knowledge Base

Provide a JSONL file with entities. Each line should be a JSON object with:
- `id` (required): Unique entity identifier
- `title` (required): Entity name/title
- `description` (optional): Entity description
- Additional fields: Any metadata (e.g., `popularity`)

**Example** (`sample_kb.jsonl`):
```json
{"id": "Q1", "title": "Albert Einstein", "description": "German-born theoretical physicist", "popularity": 100}
{"id": "Q2", "title": "Marie Curie", "description": "Polish-French physicist and chemist", "popularity": 90}
```

### 2. Input Text

Either:
- Type directly in the text box, or
- Upload a document (txt, pdf, docx, html)

### 3. Configure Pipeline

#### Loader
- **text**: Plain text files
- **pdf**: PDF documents (requires pdfplumber)
- **docx**: Word documents (requires python-docx)
- **html**: HTML files (requires beautifulsoup4)

Auto-detected from file extension when uploading.

#### NER (Named Entity Recognition)
- **simple**: Regex-based (lightweight, no downloads needed)
  - `min_len`: Minimum entity length (default: 3)
- **spacy**: SpaCy NER (requires model download)
  - `model`: SpaCy model name (e.g., `en_core_web_sm`)
- **gliner**: Zero-shot GLiNER (requires model download)
  - `model_name`: HuggingFace model (e.g., `urchade/gliner_large`)
  - `labels`: Comma-separated entity types (e.g., `person, organization, location`)

#### Candidate Generation
- **fuzzy**: Fuzzy string matching on titles (fast, good for small KBs)
- **bm25**: BM25 retrieval on descriptions (better for large KBs)
- **dense**: Dense embedding search (requires sentence-transformers)

All support `top_k` parameter (default: 10).

#### Reranking (Optional)
- **none**: No reranking
- **cross_encoder**: Semantic reranking (requires sentence-transformers)

#### Disambiguation
- **first**: Pick first candidate (simple baseline)
- **popularity**: Use `popularity` field from KB metadata
- **llm**: LLM-based disambiguation (experimental, requires LLM)
- **none**: No disambiguation

### 4. Run Pipeline

Click "Run Pipeline" and view results:
- **Highlighted Text**: Entities with color-coded labels
- **JSON Output**: Full results with candidates and scores

## Testing

Run component tests:
```bash
python test_gradio_app.py
```

Or test with sample files:
```bash
# In the UI, use:
# - KB: data/test/sample_kb.jsonl
# - Text: Load data/test/sample_doc.txt
```

## Example Configurations

### Lightweight (No Heavy Models)
- **NER**: simple
- **Candidates**: fuzzy
- **Disambiguator**: first
- Good for: Small KBs, quick tests

### Accurate (With Models)
- **NER**: spacy (en_core_web_sm)
- **Candidates**: bm25
- **Reranker**: cross_encoder
- **Disambiguator**: popularity
- Good for: Production use, larger KBs

### Zero-shot
- **NER**: gliner
- **Candidates**: dense
- **Disambiguator**: popularity
- Good for: Custom entity types, domain adaptation

## Troubleshooting

### "Model not found" (SpaCy)
```bash
python -m spacy download en_core_web_sm
```

### CUDA warnings
If you see CUDA compatibility warnings but don't have a compatible GPU, the pipeline will fall back to CPU. This is expected and won't affect results, just speed.

### "Knowledge base file required"
Make sure to upload a JSONL file with the proper format before running.

### Empty results
- Check that your KB contains relevant entities
- Try the "simple" NER if other models aren't detecting entities
- Verify your input text contains capitalized entity mentions

## Development

The app uses:
- **Gradio Blocks** for flexible layout
- **Dynamic component visibility** based on dropdown selection
- **Progress tracking** during pipeline execution
- **Error handling** with user-friendly messages

Key files:
- `app.py`: Main Gradio application
- `ner_pipeline/`: Core pipeline logic
- `data/test/`: Sample files for testing

