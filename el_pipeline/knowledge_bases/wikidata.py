from el_pipeline.registry import knowledge_bases
from el_pipeline.knowledge_bases.custom import CustomJSONLKnowledgeBase


@knowledge_bases.register("wikidata")
class WikidataKB(CustomJSONLKnowledgeBase):
    """Wikidata adapter backed by a provided JSONL dump."""

    def __init__(self, path: str):
        super().__init__(path=path)

