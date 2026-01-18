from typing import List

from rapidfuzz import process

from ner_pipeline.registry import candidate_generators
from ner_pipeline.types import Candidate, Document, Mention
from ner_pipeline.knowledge_bases.base import KnowledgeBase


@candidate_generators.register("fuzzy")
class FuzzyCandidateGenerator:
    """Fuzzy string matching against KB titles."""

    def __init__(self, kb: KnowledgeBase, top_k: int = 20):
        if kb is None:
            raise ValueError("Fuzzy matching requires a knowledge base.")
        self.kb = kb
        self.entities = list(kb.all_entities())
        self.titles = [e.title for e in self.entities]
        self.top_k = top_k

    def generate(self, mention: Mention, doc: Document) -> List[Candidate]:
        # Fuzzy matching uses mention text only (context not useful for string matching)
        results = process.extract(mention.text, self.titles, limit=self.top_k)
        candidates: List[Candidate] = []
        for title, score, idx in results:
            ent = self.entities[idx]
            candidates.append(
                Candidate(entity_id=ent.id, score=float(score), description=ent.description)
            )
        return candidates

