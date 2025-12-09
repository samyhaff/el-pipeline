from typing import List, Optional

from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from ner_pipeline.registry import disambiguators
from ner_pipeline.types import Candidate, Document, Entity, Mention
from ner_pipeline.knowledge_bases.base import KnowledgeBase


@disambiguators.register("llm")
class LLMDisambiguator:
    """Lightweight LLM-based disambiguation using sequence classification."""

    def __init__(
        self,
        kb: KnowledgeBase,
        model_name: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
    ):
        self.kb = kb
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)

    def disambiguate(
        self, mention: Mention, candidates: List[Candidate], doc: Document
    ) -> Optional[Entity]:
        if not candidates:
            return None

        prompts = []
        for cand in candidates:
            desc = cand.description or ""
            text = (
                f"Context: {doc.text}\nMention: {mention.text}\n"
                f"Candidate: {cand.entity_id}\nDescription: {desc}"
            )
            prompts.append(text)

        outputs = self.pipe(prompts)
        scored = list(zip(candidates, outputs))
        scored.sort(key=lambda x: x[1].get("score", 0.0), reverse=True)
        best = scored[0][0]
        return self.kb.get_entity(best.entity_id)

