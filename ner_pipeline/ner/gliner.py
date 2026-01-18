from typing import List

from gliner import GLiNER

from ner_pipeline.registry import ner_models
from ner_pipeline.types import Mention
from ner_pipeline.context import extract_context


DEFAULT_GLINER_LABELS = ["person", "organization", "location", "event", "product"]


@ner_models.register("gliner")
class GLiNERNER:
    """Zero-shot GLiNER NER."""

    def __init__(self, model_name: str = "urchade/gliner_large", labels=None, context_mode: str = "sentence"):
        self.model = GLiNER.from_pretrained(model_name)
        self.labels = labels if labels is not None else DEFAULT_GLINER_LABELS
        self.context_mode = context_mode

    def extract(self, text: str) -> List[Mention]:
        predictions = self.model.predict_entities(text, labels=self.labels, threshold=0.3)
        mentions = []
        for pred in predictions:
            start, end = pred["start"], pred["end"]
            context = extract_context(text, start, end, mode=self.context_mode)
            mentions.append(
                Mention(
                    start=start,
                    end=end,
                    text=pred["text"],
                    label=pred.get("label"),
                    context=context,
                )
            )
        return mentions

