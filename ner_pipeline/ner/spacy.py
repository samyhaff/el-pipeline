import spacy
from ner_pipeline.registry import ner_models
from ner_pipeline.types import Mention


@ner_models.register("spacy")
class SpacyNER:
    """spaCy NER component."""

    def __init__(self, model: str = "en_core_web_sm") -> None:
        self.nlp = spacy.load(model)

    def extract(self, text: str):
        doc = self.nlp(text)
        return [
            Mention(
                start=ent.start_char,
                end=ent.end_char,
                text=ent.text,
                label=ent.label_,
            )
            for ent in doc.ents
        ]

