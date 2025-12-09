from gliner import GLiNER

from ner_pipeline.registry import ner_models
from ner_pipeline.types import Mention


@ner_models.register("gliner")
class GLiNERNER:
    """Zero-shot GLiNER NER."""

    def __init__(self, model_name: str = "urchade/gliner_large", labels=None):
        self.model = GLiNER.from_pretrained(model_name)
        self.labels = labels

    def extract(self, text: str):
        predictions = self.model.predict_entities(text, labels=self.labels, threshold=0.3)
        mentions = []
        for pred in predictions:
            mentions.append(
                Mention(
                    start=pred["start"],
                    end=pred["end"],
                    text=pred["text"],
                    label=pred.get("label"),
                )
            )
        return mentions

