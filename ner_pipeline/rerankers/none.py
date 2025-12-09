from typing import List

from ner_pipeline.registry import rerankers
from ner_pipeline.types import Candidate, Document, Mention


@rerankers.register("none")
class NoOpReranker:
    """Keeps candidates as-is."""

    def rerank(self, mention: Mention, candidates: List[Candidate], doc: Document):
        return candidates

