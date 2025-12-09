from typing import List

from transformers import pipeline

from ner_pipeline.registry import ner_models
from ner_pipeline.types import Mention


@ner_models.register("transformers")
class TransformersNER:
    """HuggingFace transformers NER."""

    def __init__(self, model_name: str = "dslim/bert-base-NER") -> None:
        self.pipe = pipeline("ner", model=model_name, aggregation_strategy="simple")

    def extract(self, text: str) -> List[Mention]:
        outputs = self.pipe(text)
        mentions: List[Mention] = []
        for out in outputs:
            mentions.append(
                Mention(
                    start=int(out["start"]),
                    end=int(out["end"]),
                    text=out["word"],
                    label=out.get("entity_group"),
                )
            )
        return mentions

