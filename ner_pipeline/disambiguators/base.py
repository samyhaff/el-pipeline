from typing import List, Optional, Protocol

from ner_pipeline.types import Candidate, Document, Entity, Mention


class Disambiguator(Protocol):
    """Selects a final entity."""

    def disambiguate(
        self, mention: Mention, candidates: List[Candidate], doc: Document
    ) -> Optional[Entity]:
        ...

