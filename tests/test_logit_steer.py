from __future__ import annotations

import torch

from introspection.logit_steer import (
    FRAMING_MESSAGES,
    INVERTED_DETECTION_QUESTION,
    TRIAL_MARKER,
    build_detection_messages,
    build_factual_messages,
    build_requests,
)
from introspection.constants import FACTUAL_NO_QUESTIONS, FACTUAL_YES_QUESTIONS


def test_detection_messages_structure() -> None:
    messages = build_detection_messages()
    assert len(messages) == 3
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"] == "Ok."
    assert messages[2]["role"] == "user"
    assert TRIAL_MARKER in messages[2]["content"]
    assert "YES or NO" in messages[2]["content"]


def test_factual_messages_structure() -> None:
    question = "Is the Earth flat?"
    messages = build_factual_messages(question)
    assert len(messages) == 3
    assert messages[0] == FRAMING_MESSAGES[0]
    assert messages[1] == FRAMING_MESSAGES[1]
    assert TRIAL_MARKER in messages[2]["content"]
    assert question in messages[2]["content"]
    assert "YES or NO" in messages[2]["content"]


def test_detection_and_factual_share_framing() -> None:
    detection = build_detection_messages()
    factual = build_factual_messages("Is the Earth flat?")
    # First two messages (framing + Ok) must be identical
    assert detection[0] == factual[0]
    assert detection[1] == factual[1]
    # Third message differs only in the question content
    assert detection[2]["role"] == factual[2]["role"] == "user"
    # Both have the Trial 1 marker
    assert TRIAL_MARKER in detection[2]["content"]
    assert TRIAL_MARKER in factual[2]["content"]


def test_factual_questions_all_present() -> None:
    assert len(FACTUAL_NO_QUESTIONS) == 10
    for q in FACTUAL_NO_QUESTIONS:
        assert q.endswith("?")


def test_build_requests_count() -> None:
    concepts = ["oceans", "snow", "dust"]
    layers = [5, 10, 15]
    strengths = [3.5, 4.0]
    requests = build_requests(concepts, layers, strengths)
    assert len(requests) == len(concepts) * len(layers) * len(strengths)  # 3*3*2 = 18


def test_build_requests_content() -> None:
    concepts = ["oceans"]
    layers = [10]
    strengths = [4.0]
    requests = build_requests(concepts, layers, strengths)
    assert len(requests) == 1
    r = requests[0]
    assert r.concept == "oceans"
    assert r.layers == [10]
    assert r.strength == 4.0
    assert r.layer_label == "10"


def test_build_requests_ordering() -> None:
    """Verify requests iterate concepts outermost, then layers, then strengths."""
    concepts = ["a", "b"]
    layers = [1, 2]
    strengths = [0.5, 1.0]
    requests = build_requests(concepts, layers, strengths)
    expected = [
        ("a", 1, 0.5),
        ("a", 1, 1.0),
        ("a", 2, 0.5),
        ("a", 2, 1.0),
        ("b", 1, 0.5),
        ("b", 1, 1.0),
        ("b", 2, 0.5),
        ("b", 2, 1.0),
    ]
    for req, (c, layer, s) in zip(requests, expected):
        assert req.concept == c
        assert req.layers == [layer]
        assert req.strength == s


def test_extract_logit_diffs_mock() -> None:
    """Test extract_logit_diffs with a mock model that returns known logits."""
    from unittest.mock import MagicMock
    from introspection.logit_steer import extract_logit_diffs
    from introspection.types import PromptSetup

    vocab_size = 100
    yes_id = 10
    no_id = 20
    batch_size = 3

    # Create mock logits: (batch, seq_len, vocab_size)
    mock_logits = torch.zeros(batch_size, 5, vocab_size)
    # Set known values at last position for YES and NO tokens
    mock_logits[0, -1, yes_id] = 2.0
    mock_logits[0, -1, no_id] = -1.0
    mock_logits[1, -1, yes_id] = -0.5
    mock_logits[1, -1, no_id] = 3.0
    mock_logits[2, -1, yes_id] = 1.0
    mock_logits[2, -1, no_id] = 1.0

    # Build mock model
    mock_output = MagicMock()
    mock_output.logits = mock_logits
    mock_model = MagicMock()
    mock_model.return_value = mock_output

    input_ids = torch.zeros(1, 5, dtype=torch.long)
    attention_mask = torch.ones(1, 5, dtype=torch.long)
    prompt = PromptSetup(
        input_ids=input_ids,
        attention_mask=attention_mask,
        formatted_prompt="test",
        injection_index=0,
    )

    results = extract_logit_diffs(mock_model, prompt, batch_size, yes_id, no_id)

    assert len(results) == 3
    assert abs(results[0]["logit_diff"] - 3.0) < 1e-5  # 2.0 - (-1.0)
    assert abs(results[1]["logit_diff"] - (-3.5)) < 1e-5  # -0.5 - 3.0
    assert abs(results[2]["logit_diff"] - 0.0) < 1e-5  # 1.0 - 1.0
    assert abs(results[0]["logit_yes"] - 2.0) < 1e-5
    assert abs(results[0]["logit_no"] - (-1.0)) < 1e-5


def test_inverted_detection_messages_structure() -> None:
    messages = build_detection_messages(inverted=True)
    assert len(messages) == 3
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"] == "Ok."
    assert messages[2]["role"] == "user"
    assert TRIAL_MARKER in messages[2]["content"]
    assert "YES or NO" in messages[2]["content"]
    assert "control trial" in messages[2]["content"]


def test_inverted_detection_question_text() -> None:
    assert "Was this a control trial" in INVERTED_DETECTION_QUESTION
    assert TRIAL_MARKER in INVERTED_DETECTION_QUESTION


def test_factual_yes_questions_count() -> None:
    assert len(FACTUAL_YES_QUESTIONS) == 10
    for q in FACTUAL_YES_QUESTIONS:
        assert q.endswith("?")


def test_inverted_and_normal_share_framing() -> None:
    normal = build_detection_messages(inverted=False)
    inverted = build_detection_messages(inverted=True)
    assert normal[0] == inverted[0]
    assert normal[1] == inverted[1]
    assert normal[2]["content"] != inverted[2]["content"]


def test_build_detection_messages_default_not_inverted() -> None:
    default = build_detection_messages()
    explicit = build_detection_messages(inverted=False)
    assert default == explicit


def test_factual_yes_questions_are_inverses() -> None:
    assert len(FACTUAL_YES_QUESTIONS) == len(FACTUAL_NO_QUESTIONS)
