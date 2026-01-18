import re
from typing import List

from ner_pipeline.registry import ner_models
from ner_pipeline.types import Mention
from ner_pipeline.context import extract_context


@ner_models.register("simple")
class SimpleRegexNER:
    """Lightweight regex-based NER for quick tests (no external deps)."""

    def __init__(self, min_len: int = 3, context_mode: str = "sentence"):
        self.pattern = re.compile(
            r"\b([A-Z][a-zA-Z0-9_-]+(?:\s+[A-Z][a-zA-Z0-9_-]+)*)\b"
        )
        self.min_len = min_len
        self.context_mode = context_mode

    def extract(self, text: str) -> List[Mention]:
        mentions: List[Mention] = []
        for match in self.pattern.finditer(text):
            span = match.group(1)
            if len(span) < self.min_len:
                continue
            start, end = match.start(1), match.end(1)
            context = extract_context(text, start, end, mode=self.context_mode)
            mentions.append(
                Mention(
                    start=start,
                    end=end,
                    text=span,
                    label="ENT",
                    context=context,
                )
            )
        return mentions

