from ner_pipeline.registry import knowledge_bases
from ner_pipeline.knowledge_bases.custom import CustomJSONLKnowledgeBase


@knowledge_bases.register("wikipedia")
class WikipediaKB(CustomJSONLKnowledgeBase):
    """Wikipedia adapter backed by a provided JSONL dump."""

    def __init__(self, path: str):
        super().__init__(path=path)

