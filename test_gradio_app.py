"""
Simple test script to verify the Gradio app components work correctly.
This doesn't launch the UI, but tests the core pipeline function.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app import run_pipeline, format_highlighted_text


class MockFile:
    """Mock file object for testing."""
    def __init__(self, path: str):
        self.name = path


def test_basic_pipeline():
    """Test basic pipeline with sample data."""
    print("Testing basic pipeline with sample files...")
    
    kb_file = MockFile("data/test/sample_kb.jsonl")
    
    text_input = "Albert Einstein was born in Germany. Marie Curie was a pioneering scientist."
    
    try:
        highlighted, result = run_pipeline(
            text_input=text_input,
            file_input=None,
            kb_file=kb_file,
            loader_type="text",
            ner_type="simple",
            spacy_model="en_core_web_sm",
            gliner_model="urchade/gliner_large",
            gliner_labels="",
            simple_min_len=3,
            cand_type="fuzzy",
            cand_top_k=10,
            reranker_type="none",
            disambig_type="first",
        )
        
        print("\n✅ Pipeline executed successfully!")
        print(f"\nFound {len(result['entities'])} entities:")
        for entity in result['entities'][:5]:
            print(f"  - {entity['text']} → {entity.get('entity_title', 'None')}")
        
        print(f"\nHighlighted text segments: {len(highlighted)}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_format_highlighted():
    """Test the highlighted text formatting."""
    print("\nTesting highlighted text formatting...")
    
    sample_result = {
        "text": "Albert Einstein was born in Germany.",
        "entities": [
            {
                "text": "Albert Einstein",
                "start": 0,
                "end": 15,
                "label": "PERSON",
                "entity_title": "Albert Einstein",
            },
            {
                "text": "Germany",
                "start": 28,
                "end": 35,
                "label": "GPE",
                "entity_title": "Berlin",
            }
        ]
    }
    
    highlighted = format_highlighted_text(sample_result)
    
    print(f"✅ Formatted {len(highlighted)} segments")
    for text, label in highlighted:
        if label:
            print(f"  - [{label}] {text}")
    
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("NER Pipeline Gradio App - Component Tests")
    print("=" * 60)
    
    success = True
    
    success = test_format_highlighted() and success
    success = test_basic_pipeline() and success
    
    print("\n" + "=" * 60)
    if success:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed")
    print("=" * 60)
    
    sys.exit(0 if success else 1)

