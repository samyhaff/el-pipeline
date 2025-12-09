import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

# Ensure component registration by importing modules with registry decorators.
from ner_pipeline import loaders as _loaders_pkg  # noqa: F401
from ner_pipeline import ner as _ner_pkg  # noqa: F401
from ner_pipeline import candidates as _cands_pkg  # noqa: F401
from ner_pipeline import rerankers as _rerankers_pkg  # noqa: F401
from ner_pipeline import disambiguators as _disamb_pkg  # noqa: F401
from ner_pipeline import knowledge_bases as _kb_pkg  # noqa: F401

from .config import PipelineConfig
from .registry import (
    candidate_generators,
    disambiguators,
    knowledge_bases,
    loaders,
    ner_models,
    rerankers,
)
from .types import Candidate, Document, Entity, Mention, ResolvedMention


class NERPipeline:
    """Orchestrates the modular pipeline."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        kb_factory = None
        if config.knowledge_base:
            kb_factory = knowledge_bases.get(config.knowledge_base.name)
            self.kb = kb_factory(**config.knowledge_base.params)
        else:
            self.kb = None

        loader_factory = loaders.get(config.loader.name)
        self.loader = loader_factory(**config.loader.params)

        ner_factory = ner_models.get(config.ner.name)
        self.ner_model = ner_factory(**config.ner.params)

        cand_factory = candidate_generators.get(config.candidate_generator.name)
        self.candidate_generator = cand_factory(
            kb=self.kb, **config.candidate_generator.params
        )

        self.reranker = None
        if config.reranker:
            reranker_factory = rerankers.get(config.reranker.name)
            self.reranker = reranker_factory(**config.reranker.params)

        self.disambiguator = None
        if config.disambiguator:
            disamb_factory = disambiguators.get(config.disambiguator.name)
            self.disambiguator = disamb_factory(
                kb=self.kb, **config.disambiguator.params
            )

    def _cache_key(self, path: str) -> str:
        stat = os.stat(path)
        raw = f"{path}-{stat.st_mtime}-{stat.st_size}".encode()
        return hashlib.sha256(raw).hexdigest()

    def _load_with_cache(self, path: str) -> Iterator[Document]:
        key = self._cache_key(path)
        cache_file = self.cache_dir / f"{key}.pkl"
        if cache_file.exists():
            with cache_file.open("rb") as f:
                cached = pickle.load(f)
            for doc in cached:
                yield doc
            return

        docs = list(self.loader.load(path))
        with cache_file.open("wb") as f:
            pickle.dump(docs, f)
        for doc in docs:
            yield doc

    def process_document(self, doc: Document) -> Dict:
        mentions = self.ner_model.extract(doc.text)
        resolved: List[ResolvedMention] = []

        for mention in mentions:
            candidates = self.candidate_generator.generate(mention, doc)
            if self.reranker:
                candidates = self.reranker.rerank(mention, candidates, doc)

            entity: Optional[Entity] = None
            if self.disambiguator:
                entity = self.disambiguator.disambiguate(mention, candidates, doc)

            resolved.append(
                ResolvedMention(
                    mention=mention,
                    entity=entity,
                    candidates=candidates,
                )
            )

        return {
            "id": doc.id,
            "text": doc.text,
            "entities": [
                {
                    "text": rm.mention.text,
                    "start": rm.mention.start,
                    "end": rm.mention.end,
                    "label": rm.mention.label,
                    "entity_id": rm.entity.id if rm.entity else None,
                    "entity_title": rm.entity.title if rm.entity else None,
                    "entity_description": (
                        rm.entity.description if rm.entity else None
                    ),
                    "candidates": [
                        {
                            "entity_id": c.entity_id,
                            "score": c.score,
                            "description": c.description,
                        }
                        for c in rm.candidates
                    ],
                }
                for rm in resolved
            ],
            "meta": doc.meta,
        }

    def run(self, paths: Iterable[str], output_path: Optional[str] = None) -> List[Dict]:
        results: List[Dict] = []
        writer = None
        if output_path:
            writer = Path(output_path).open("w", encoding="utf-8")

        try:
            for path in paths:
                for doc in self._load_with_cache(path):
                    result = self.process_document(doc)
                    if writer:
                        writer.write(json.dumps(result) + "\n")
                    results.append(result)
        finally:
            if writer:
                writer.close()

        return results

