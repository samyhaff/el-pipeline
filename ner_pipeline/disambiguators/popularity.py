from typing import List, Optional

from ner_pipeline.registry import disambiguators
from ner_pipeline.types import Candidate, Document, Entity, Mention
from ner_pipeline.knowledge_bases.base import KnowledgeBase


@disambiguators.register("popularity")
class PopularityDisambiguator:
    """Chooses highest-scored candidate (falling back to KB lookup)."""

    def __init__(self, kb: KnowledgeBase):
        self.kb = kb

    def disambiguate(
        self, mention: Mention, candidates: List[Candidate], doc: Document
    ) -> Optional[Entity]:
        if not candidates:
            return None
        top = max(candidates, key=lambda c: c.score or 0.0)
        return self.kb.get_entity(top.entity_id)

