from typing import List

from rank_bm25 import BM25Okapi

from ner_pipeline.registry import candidate_generators
from ner_pipeline.types import Candidate, Document, Entity, Mention
from ner_pipeline.knowledge_bases.base import KnowledgeBase


def _tokenize(text: str) -> List[str]:
    return text.lower().split()


@candidate_generators.register("bm25")
class BM25CandidateGenerator:
    """BM25 over entity descriptions."""

    def __init__(self, kb: KnowledgeBase, top_k: int = 20, use_context: bool = True):
        if kb is None:
            raise ValueError("BM25 requires a knowledge base.")
        self.kb = kb
        descriptions = [e.description or "" for e in kb.all_entities()]
        self.entities = list(kb.all_entities())
        corpus = [_tokenize(d) for d in descriptions]
        self.bm25 = BM25Okapi(corpus)
        self.top_k = top_k
        self.use_context = use_context

    def generate(self, mention: Mention, doc: Document) -> List[Candidate]:
        # Use context if available and enabled for better retrieval
        if self.use_context and mention.context:
            query_text = f"{mention.text} {mention.context}"
        else:
            query_text = mention.text
        
        scores = self.bm25.get_scores(_tokenize(query_text))
        paired = list(zip(self.entities, scores))
        paired.sort(key=lambda x: x[1], reverse=True)
        top = paired[: self.top_k]
        return [
            Candidate(entity_id=ent.id, score=float(score), description=ent.description)
            for ent, score in top
        ]

