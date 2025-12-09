from typing import List, Optional

from ner_pipeline.registry import disambiguators
from ner_pipeline.types import Candidate, Document, Entity, Mention
from ner_pipeline.knowledge_bases.base import KnowledgeBase


@disambiguators.register("first")
class FirstCandidateDisambiguator:
    """Returns the first candidate in the list."""

    def __init__(self, kb: KnowledgeBase):
        self.kb = kb

    def disambiguate(
        self, mention: Mention, candidates: List[Candidate], doc: Document
    ) -> Optional[Entity]:
        if not candidates:
            return None
        return self.kb.get_entity(candidates[0].entity_id)

