import spacy
from typing import List

from ner_pipeline.registry import ner_models
from ner_pipeline.types import Mention
from ner_pipeline.context import extract_context


@ner_models.register("spacy")
class SpacyNER:
    """spaCy NER component."""

    def __init__(self, model: str = "en_core_web_sm", context_mode: str = "sentence") -> None:
        self.nlp = spacy.load(model)
        self.context_mode = context_mode

    def extract(self, text: str) -> List[Mention]:
        doc = self.nlp(text)
        mentions = []
        for ent in doc.ents:
            context = extract_context(text, ent.start_char, ent.end_char, mode=self.context_mode)
            mentions.append(
                Mention(
                    start=ent.start_char,
                    end=ent.end_char,
                    text=ent.text,
                    label=ent.label_,
                    context=context,
                )
            )
        return mentions

