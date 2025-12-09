from typing import List, Protocol

from ner_pipeline.types import Candidate, Document, Mention


class CandidateGenerator(Protocol):
    """Generates candidates for a mention."""

    def generate(self, mention: Mention, doc: Document) -> List[Candidate]:
        ...

