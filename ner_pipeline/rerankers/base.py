from typing import List, Protocol

from ner_pipeline.types import Candidate, Document, Mention


class Reranker(Protocol):
    """Reorders candidates."""

    def rerank(self, mention: Mention, candidates: List[Candidate], doc: Document) -> List[Candidate]:
        ...

