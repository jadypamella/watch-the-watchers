"""Unit tests for the pure helpers in self_preference_eval.

Run with: pytest -q
These cover parsing untrusted judge output and mapping a pick to its owner,
including edge cases (no digit, extra text, position randomisation).
"""
from self_preference_eval import parse_pick, resolve_owner


def test_parse_pick_reads_first_digit():
    assert parse_pick("1") == 1
    assert parse_pick("2") == 2
    assert parse_pick("Answer 2 is better") == 2
    assert parse_pick("I think 1, because...") == 1


def test_parse_pick_returns_none_when_absent():
    assert parse_pick("") is None
    assert parse_pick("neither") is None
    assert parse_pick(None) is None  # untrusted output may be non-string


def test_resolve_owner_maps_pick_to_model():
    assert resolve_owner(1, "gemma", "qwen") == "gemma"
    assert resolve_owner(2, "gemma", "qwen") == "qwen"


def test_resolve_owner_handles_unparseable_pick():
    assert resolve_owner(None, "gemma", "qwen") is None


def test_self_preference_counts_with_randomised_order():
    # If gemma's answer is shown second and the judge picks 2, the owner is gemma.
    assert resolve_owner(parse_pick("2"), "qwen", "gemma") == "gemma"
    # If gemma's answer is shown first and the judge picks 1, the owner is gemma.
    assert resolve_owner(parse_pick("1"), "gemma", "qwen") == "gemma"
