"""Unit tests for LELACrossEncoderVLLMRerankerComponent."""

import math
from unittest.mock import MagicMock, patch

import pytest
import spacy
from spacy.tokens import Span

from el_pipeline.types import Candidate
from el_pipeline.utils import ensure_candidates_extension


@pytest.fixture
def nlp():
    return spacy.blank("en")


@pytest.fixture
def sample_candidates() -> list:
    return [
        Candidate(entity_id="E1", description="Description 1"),
        Candidate(entity_id="E2", description="Description 2"),
        Candidate(entity_id="E3", description="Description 3"),
        Candidate(entity_id="E4", description="Description 4"),
        Candidate(entity_id="E5", description="Description 5"),
    ]


def _make_generate_output(yes_probability: float):
    """Create a mock vLLM generate output with yes/no logprobs."""
    no_probability = 1.0 - yes_probability
    yes_logprob = math.log(max(yes_probability, 1e-10))
    no_logprob = math.log(max(no_probability, 1e-10))

    yes_token = MagicMock()
    yes_token.decoded_token = "yes"
    yes_token.logprob = yes_logprob

    no_token = MagicMock()
    no_token.decoded_token = "no"
    no_token.logprob = no_logprob

    generation = MagicMock()
    generation.text = "yes" if yes_probability > 0.5 else "no"
    generation.logprobs = [{0: yes_token, 1: no_token}]

    output = MagicMock()
    output.outputs = [generation]
    return output


class TestLELACrossEncoderVLLMRerankerComponent:
    """Tests for LELACrossEncoderVLLMRerankerComponent."""

    @patch("el_pipeline.spacy_components.rerankers.release_vllm")
    @patch("el_pipeline.spacy_components.rerankers.get_vllm_instance")
    @patch("el_pipeline.spacy_components.rerankers._get_vllm")
    def test_rerank_returns_candidates(
        self, mock_get_vllm_mod, mock_get_instance, mock_release, sample_candidates, nlp
    ):
        mock_model = MagicMock()
        mock_model.generate.return_value = [
            _make_generate_output(0.1),
            _make_generate_output(0.5),
            _make_generate_output(0.9),
            _make_generate_output(0.3),
            _make_generate_output(0.7),
        ]
        mock_get_instance.return_value = (mock_model, False)

        mock_vllm_mod = MagicMock()
        mock_get_vllm_mod.return_value = mock_vllm_mod

        from el_pipeline.spacy_components.rerankers import LELACrossEncoderVLLMRerankerComponent
        ensure_candidates_extension()

        reranker = LELACrossEncoderVLLMRerankerComponent(nlp=nlp, top_k=3)

        doc = nlp("Obama was the president.")
        doc.ents = [Span(doc, 0, 1, label="ENTITY")]
        doc.ents[0]._.candidates = sample_candidates
        doc = reranker(doc)

        result = doc.ents[0]._.candidates
        assert len(result) == 3
        assert all(isinstance(c, Candidate) for c in result)

    @patch("el_pipeline.spacy_components.rerankers.release_vllm")
    @patch("el_pipeline.spacy_components.rerankers.get_vllm_instance")
    @patch("el_pipeline.spacy_components.rerankers._get_vllm")
    def test_rerank_sorts_by_score(
        self, mock_get_vllm_mod, mock_get_instance, mock_release, sample_candidates, nlp
    ):
        mock_model = MagicMock()
        mock_model.generate.return_value = [
            _make_generate_output(0.1),  # E1
            _make_generate_output(0.5),  # E2
            _make_generate_output(0.9),  # E3 - highest
            _make_generate_output(0.3),  # E4
            _make_generate_output(0.7),  # E5
        ]
        mock_get_instance.return_value = (mock_model, False)

        mock_vllm_mod = MagicMock()
        mock_get_vllm_mod.return_value = mock_vllm_mod

        from el_pipeline.spacy_components.rerankers import LELACrossEncoderVLLMRerankerComponent
        ensure_candidates_extension()

        reranker = LELACrossEncoderVLLMRerankerComponent(nlp=nlp, top_k=3)

        doc = nlp("Obama was the president.")
        doc.ents = [Span(doc, 0, 1, label="ENTITY")]
        doc.ents[0]._.candidates = sample_candidates
        doc = reranker(doc)

        result = doc.ents[0]._.candidates
        assert result[0].entity_id == "E3"
        assert result[1].entity_id == "E5"
        assert result[2].entity_id == "E2"

    @patch("el_pipeline.spacy_components.rerankers.release_vllm")
    @patch("el_pipeline.spacy_components.rerankers.get_vllm_instance")
    @patch("el_pipeline.spacy_components.rerankers._get_vllm")
    def test_skips_reranking_when_candidates_below_top_k(
        self, mock_get_vllm_mod, mock_get_instance, mock_release, nlp
    ):
        """Should not call .generate() if all entities have <= top_k candidates."""
        mock_model = MagicMock()
        mock_get_instance.return_value = (mock_model, False)

        mock_vllm_mod = MagicMock()
        mock_get_vllm_mod.return_value = mock_vllm_mod

        from el_pipeline.spacy_components.rerankers import LELACrossEncoderVLLMRerankerComponent
        ensure_candidates_extension()

        reranker = LELACrossEncoderVLLMRerankerComponent(nlp=nlp, top_k=5)

        candidates = [Candidate(entity_id=f"E{i}", description=f"Desc {i}") for i in range(3)]
        doc = nlp("Obama was the president.")
        doc.ents = [Span(doc, 0, 1, label="ENTITY")]
        doc.ents[0]._.candidates = candidates
        doc = reranker(doc)

        mock_model.generate.assert_not_called()
        assert len(doc.ents[0]._.candidates) == 3

    @patch("el_pipeline.spacy_components.rerankers.release_vllm")
    @patch("el_pipeline.spacy_components.rerankers.get_vllm_instance")
    @patch("el_pipeline.spacy_components.rerankers._get_vllm")
    def test_prompt_uses_cross_encoder_template(
        self, mock_get_vllm_mod, mock_get_instance, mock_release, sample_candidates, nlp
    ):
        """Prompt should use the Qwen3-Reranker template with marked mention and document."""
        mock_model = MagicMock()
        mock_model.generate.return_value = [_make_generate_output(0.5)] * 5
        mock_get_instance.return_value = (mock_model, False)

        mock_vllm_mod = MagicMock()
        mock_get_vllm_mod.return_value = mock_vllm_mod

        from el_pipeline.spacy_components.rerankers import LELACrossEncoderVLLMRerankerComponent
        ensure_candidates_extension()

        reranker = LELACrossEncoderVLLMRerankerComponent(nlp=nlp, top_k=3)

        doc = nlp("Obama was the president.")
        doc.ents = [Span(doc, 0, 1, label="ENTITY")]
        doc.ents[0]._.candidates = sample_candidates
        doc = reranker(doc)

        call_args = mock_model.generate.call_args
        prompts = call_args[0][0]
        assert len(prompts) == 5
        assert "[Obama]" in prompts[0]
        assert "<Instruct>" in prompts[0]
        assert "<Query>" in prompts[0]
        assert "<Document>" in prompts[0]
        assert "E1 (Description 1)" in prompts[0]

    @patch("el_pipeline.spacy_components.rerankers.release_vllm")
    @patch("el_pipeline.spacy_components.rerankers.get_vllm_instance")
    @patch("el_pipeline.spacy_components.rerankers._get_vllm")
    def test_preserves_descriptions(
        self, mock_get_vllm_mod, mock_get_instance, mock_release, sample_candidates, nlp
    ):
        mock_model = MagicMock()
        mock_model.generate.return_value = [
            _make_generate_output(float(i) / 5) for i in range(5)
        ]
        mock_get_instance.return_value = (mock_model, False)

        mock_vllm_mod = MagicMock()
        mock_get_vllm_mod.return_value = mock_vllm_mod

        from el_pipeline.spacy_components.rerankers import LELACrossEncoderVLLMRerankerComponent
        ensure_candidates_extension()

        reranker = LELACrossEncoderVLLMRerankerComponent(nlp=nlp, top_k=3)

        doc = nlp("Obama was the president.")
        doc.ents = [Span(doc, 0, 1, label="ENTITY")]
        doc.ents[0]._.candidates = sample_candidates
        doc = reranker(doc)

        for candidate in doc.ents[0]._.candidates:
            original = next(o for o in sample_candidates if o.entity_id == candidate.entity_id)
            assert candidate.description == original.description

    @patch("el_pipeline.spacy_components.rerankers.release_vllm")
    @patch("el_pipeline.spacy_components.rerankers.get_vllm_instance")
    @patch("el_pipeline.spacy_components.rerankers._get_vllm")
    def test_releases_vllm_after_use(
        self, mock_get_vllm_mod, mock_get_instance, mock_release, sample_candidates, nlp
    ):
        mock_model = MagicMock()
        mock_model.generate.return_value = [_make_generate_output(0.5)] * 5
        mock_get_instance.return_value = (mock_model, False)

        mock_vllm_mod = MagicMock()
        mock_get_vllm_mod.return_value = mock_vllm_mod

        from el_pipeline.spacy_components.rerankers import LELACrossEncoderVLLMRerankerComponent
        ensure_candidates_extension()

        reranker = LELACrossEncoderVLLMRerankerComponent(nlp=nlp, top_k=3)

        doc = nlp("Obama was the president.")
        doc.ents = [Span(doc, 0, 1, label="ENTITY")]
        doc.ents[0]._.candidates = sample_candidates
        reranker(doc)

        mock_release.assert_called_once_with(reranker.model_name)

    def test_initialization_with_custom_params(self, nlp):
        from el_pipeline.spacy_components.rerankers import LELACrossEncoderVLLMRerankerComponent
        reranker = LELACrossEncoderVLLMRerankerComponent(
            nlp=nlp,
            model_name="custom-reranker-model",
            top_k=5,
        )
        assert reranker.model_name == "custom-reranker-model"
        assert reranker.top_k == 5
        assert reranker.model is None

    @patch("el_pipeline.spacy_components.rerankers.release_vllm")
    @patch("el_pipeline.spacy_components.rerankers.get_vllm_instance")
    @patch("el_pipeline.spacy_components.rerankers._get_vllm")
    def test_loads_vllm_without_task(
        self, mock_get_vllm_mod, mock_get_instance, mock_release, sample_candidates, nlp
    ):
        """Model should be loaded as default causal LM (no task parameter)."""
        mock_model = MagicMock()
        mock_model.generate.return_value = [_make_generate_output(0.5)] * 5
        mock_get_instance.return_value = (mock_model, False)

        mock_vllm_mod = MagicMock()
        mock_get_vllm_mod.return_value = mock_vllm_mod

        from el_pipeline.spacy_components.rerankers import LELACrossEncoderVLLMRerankerComponent
        ensure_candidates_extension()

        reranker = LELACrossEncoderVLLMRerankerComponent(nlp=nlp, top_k=3)

        doc = nlp("Obama was the president.")
        doc.ents = [Span(doc, 0, 1, label="ENTITY")]
        doc.ents[0]._.candidates = sample_candidates
        reranker(doc)

        mock_get_instance.assert_called_once_with(
            model_name=reranker.model_name,
        )

    def test_extract_yes_probability_both_tokens(self):
        """Should compute softmax(yes, no) when both tokens present."""
        from el_pipeline.spacy_components.rerankers import LELACrossEncoderVLLMRerankerComponent

        output = _make_generate_output(0.8)
        prob = LELACrossEncoderVLLMRerankerComponent._extract_yes_probability(output)
        assert abs(prob - 0.8) < 0.01

    def test_extract_yes_probability_no_logprobs(self):
        """Should fall back to text comparison when no logprobs."""
        from el_pipeline.spacy_components.rerankers import LELACrossEncoderVLLMRerankerComponent

        output = MagicMock()
        output.outputs = [MagicMock(text="yes", logprobs=None)]
        assert LELACrossEncoderVLLMRerankerComponent._extract_yes_probability(output) == 1.0

        output.outputs = [MagicMock(text="no", logprobs=None)]
        assert LELACrossEncoderVLLMRerankerComponent._extract_yes_probability(output) == 0.0
