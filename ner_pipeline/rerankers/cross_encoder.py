from typing import List

from sentence_transformers import CrossEncoder

from ner_pipeline.registry import rerankers
from ner_pipeline.types import Candidate, Document, Mention


@rerankers.register("cross_encoder")
class CrossEncoderReranker:
    """Cross-encoder reranking of candidates."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k: int = 10,
    ):
        self.model = CrossEncoder(model_name)
        self.top_k = top_k

    def rerank(self, mention: Mention, candidates: List[Candidate], doc: Document):
        if not candidates:
            return candidates
        pairs = [
            (f"{mention.text} | {doc.text}", c.description or c.entity_id)
            for c in candidates
        ]
        scores = self.model.predict(pairs)
        scored = list(zip(candidates, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[: self.top_k]
        return [
            Candidate(
                entity_id=c.entity_id,
                score=float(score),
                description=c.description,
            )
            for c, score in top
        ]

