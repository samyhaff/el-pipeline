from typing import List, Protocol

from ner_pipeline.types import Mention


class NERModel(Protocol):
    """Extracts mentions from raw text."""

    def extract(self, text: str) -> List[Mention]:
        ...

