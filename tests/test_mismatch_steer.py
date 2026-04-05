from __future__ import annotations

import torch

from introspection.logit_steer import FRAMING_MESSAGES
from introspection.mismatch_steer import (
    build_mismatch_batch,
    build_mismatch_messages,
    compute_mismatch_pairs,
)


def _make_steering_vectors(
    concepts: list[str], dim: int = 8
) -> dict[str, dict[int, torch.Tensor]]:
    """Create fake steering vectors for testing."""
    vectors: dict[str, dict[int, torch.Tensor]] = {}
    for i, concept in enumerate(concepts):
        v = torch.zeros(dim)
        v[i % dim] = 1.0  # orthogonal-ish directions
        vectors[concept] = {0: v, 5: v * 2, 10: v * 3}
    return vectors


def test_compute_mismatch_pairs_returns_k_dissimilar() -> None:
    concepts = ["a", "b", "c", "d", "e"]
    sv = _make_steering_vectors(concepts)
    pairs, _sim_matrix, order = compute_mismatch_pairs(sv, reference_layer=0, k=2)

    assert set(order) == set(concepts)
    assert len(pairs) == 5
    for concept, partners in pairs.items():
        assert len(partners) == 2
        assert concept not in partners


def test_compute_mismatch_pairs_excludes_self() -> None:
    concepts = ["x", "y", "z"]
    sv = _make_steering_vectors(concepts)
    pairs, _, _ = compute_mismatch_pairs(sv, reference_layer=0, k=2)

    for concept, partners in pairs.items():
        assert concept not in partners


def test_compute_mismatch_pairs_clamps_k() -> None:
    concepts = ["a", "b", "c"]
    sv = _make_steering_vectors(concepts)
    # K=10 > N-1=2, should clamp to 2
    pairs, _, _ = compute_mismatch_pairs(sv, reference_layer=0, k=10)

    for partners in pairs.values():
        assert len(partners) == 2


def test_compute_mismatch_pairs_similarity_matrix_shape() -> None:
    concepts = ["a", "b", "c", "d"]
    sv = _make_steering_vectors(concepts)
    _, sim_matrix, _concept_order = compute_mismatch_pairs(sv, reference_layer=0, k=2)

    assert len(sim_matrix) == 4
    assert all(len(row) == 4 for row in sim_matrix)
    # Diagonal should be ~1.0
    for i in range(4):
        assert abs(sim_matrix[i][i] - 1.0) < 1e-5


def test_compute_mismatch_pairs_picks_most_dissimilar() -> None:
    # Create vectors where "a" is very similar to "b" but distant from "c"
    sv: dict[str, dict[int, torch.Tensor]] = {
        "a": {0: torch.tensor([1.0, 0.0, 0.0])},
        "b": {0: torch.tensor([0.99, 0.1, 0.0])},
        "c": {0: torch.tensor([0.0, 0.0, 1.0])},
    }
    pairs, _, _ = compute_mismatch_pairs(sv, reference_layer=0, k=1)
    # "a" should pick "c" as most dissimilar (orthogonal), not "b" (nearly parallel)
    assert pairs["a"] == ["c"]
    # "b" should also pick "c"
    assert pairs["b"] == ["c"]


def test_build_mismatch_messages_structure() -> None:
    messages = build_mismatch_messages("oceans")
    assert len(messages) == 3
    assert messages[0] == FRAMING_MESSAGES[0]
    assert messages[1] == FRAMING_MESSAGES[1]
    assert messages[2]["role"] == "user"
    assert "oceans" in messages[2]["content"]
    assert "Trial 1" in messages[2]["content"]
    assert "YES or NO" in messages[2]["content"]


def test_build_mismatch_messages_contains_concept_twice() -> None:
    messages = build_mismatch_messages("lightning")
    content = messages[2]["content"]
    # Concept should appear in both the claim and the question
    assert content.count("lightning") >= 2


def test_build_mismatch_batch_sizes() -> None:
    concept_order = ["a", "b", "c", "d"]
    sim_matrix = [[1.0, 0.5, 0.2, 0.1]] * 4  # dummy
    partners = ["c", "d"]  # K=2

    requests, conditions = build_mismatch_batch(
        named_concept="a",
        partners=partners,
        layer=10,
        strength=4.0,
        similarity_matrix=sim_matrix,
        concept_order=concept_order,
    )
    # 1 congruent + 2 incongruent + 1 baseline = 4
    assert len(requests) == 4
    assert len(conditions) == 4


def test_build_mismatch_batch_conditions() -> None:
    concept_order = ["a", "b", "c"]
    sim_matrix = [
        [1.0, 0.5, 0.1],
        [0.5, 1.0, 0.3],
        [0.1, 0.3, 1.0],
    ]
    partners = ["c"]  # K=1

    batch_requests, conditions = build_mismatch_batch(
        named_concept="a",
        partners=partners,
        layer=5,
        strength=3.5,
        similarity_matrix=sim_matrix,
        concept_order=concept_order,
    )

    # Order: congruent, incongruent, baseline
    assert len(batch_requests) == len(conditions)
    assert conditions[0].condition == "congruent"
    assert conditions[0].named_concept == "a"
    assert conditions[0].injected_concept == "a"
    assert conditions[0].cosine_similarity == 1.0

    assert conditions[1].condition == "incongruent"
    assert conditions[1].named_concept == "a"
    assert conditions[1].injected_concept == "c"
    assert abs(conditions[1].cosine_similarity - 0.1) < 1e-5  # type: ignore[operator]

    assert conditions[2].condition == "baseline"
    assert conditions[2].named_concept == "a"
    assert conditions[2].injected_concept is None
    assert conditions[2].cosine_similarity is None


def test_build_mismatch_batch_baseline_strength_zero() -> None:
    concept_order = ["a", "b"]
    sim_matrix = [[1.0, 0.5], [0.5, 1.0]]

    batch_requests, conditions = build_mismatch_batch(
        named_concept="a",
        partners=["b"],
        layer=10,
        strength=4.0,
        similarity_matrix=sim_matrix,
        concept_order=concept_order,
    )

    # Find the baseline request (last one)
    baseline_idx = next(
        i for i, c in enumerate(conditions) if c.condition == "baseline"
    )
    assert batch_requests[baseline_idx].strength == 0.0


def test_build_mismatch_batch_congruent_uses_named_concept() -> None:
    concept_order = ["x", "y", "z"]
    sim_matrix = [[1.0, 0.5, 0.1]] * 3

    batch_requests, conditions = build_mismatch_batch(
        named_concept="x",
        partners=["z"],
        layer=10,
        strength=4.0,
        similarity_matrix=sim_matrix,
        concept_order=concept_order,
    )

    # Congruent should inject the named concept's vector
    congruent_idx = next(
        i for i, c in enumerate(conditions) if c.condition == "congruent"
    )
    assert batch_requests[congruent_idx].concept == "x"
    assert batch_requests[congruent_idx].strength == 4.0


def test_build_mismatch_batch_incongruent_uses_partner_concept() -> None:
    concept_order = ["x", "y", "z"]
    sim_matrix = [[1.0, 0.5, 0.1]] * 3

    batch_requests, conditions = build_mismatch_batch(
        named_concept="x",
        partners=["y", "z"],
        layer=10,
        strength=4.0,
        similarity_matrix=sim_matrix,
        concept_order=concept_order,
    )

    incongruent = [
        (batch_requests[i], conditions[i])
        for i in range(len(conditions))
        if conditions[i].condition == "incongruent"
    ]
    assert len(incongruent) == 2
    injected_concepts = {req.concept for req, _ in incongruent}
    assert injected_concepts == {"y", "z"}
    for req, _ in incongruent:
        assert req.strength == 4.0
