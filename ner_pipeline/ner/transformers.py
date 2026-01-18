from typing import List

from transformers import pipeline

from ner_pipeline.registry import ner_models
from ner_pipeline.types import Mention
from ner_pipeline.context import extract_context


@ner_models.register("transformers")
class TransformersNER:
    """HuggingFace transformers NER."""

    def __init__(self, model_name: str = "dslim/bert-base-NER", context_mode: str = "sentence") -> None:
        self.pipe = pipeline("ner", model=model_name, aggregation_strategy="simple")
        self.context_mode = context_mode

    def extract(self, text: str) -> List[Mention]:
        outputs = self.pipe(text)
        mentions: List[Mention] = []
        for out in outputs:
            start, end = int(out["start"]), int(out["end"])
            context = extract_context(text, start, end, mode=self.context_mode)
            mentions.append(
                Mention(
                    start=start,
                    end=end,
                    text=out["word"],
                    label=out.get("entity_group"),
                    context=context,
                )
            )
        return mentions

