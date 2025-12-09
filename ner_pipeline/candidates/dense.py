from typing import List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from ner_pipeline.registry import candidate_generators
from ner_pipeline.types import Candidate, Document, Mention
from ner_pipeline.knowledge_bases.base import KnowledgeBase


@candidate_generators.register("dense")
class DenseCandidateGenerator:
    """Dense retrieval over entity descriptions."""

    def __init__(
        self,
        kb: KnowledgeBase,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        top_k: int = 20,
    ):
        if kb is None:
            raise ValueError("Dense retrieval requires a knowledge base.")
        self.kb = kb
        self.model = SentenceTransformer(model_name)
        self.top_k = top_k

        self.entities = list(kb.all_entities())
        corpus = [e.description or e.title for e in self.entities]
        embeddings = self.model.encode(corpus, convert_to_numpy=True, normalize_embeddings=True)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings.astype(np.float32))

    def generate(self, mention: Mention, doc: Document):
        query = self.model.encode(
            [mention.text], convert_to_numpy=True, normalize_embeddings=True
        )
        scores, idx = self.index.search(query.astype(np.float32), self.top_k)
        results: List[Candidate] = []
        for score, i in zip(scores[0], idx[0]):
            ent = self.entities[int(i)]
            results.append(
                Candidate(entity_id=ent.id, score=float(score), description=ent.description)
            )
        return results

