"""Unit tests for the pure helper in coverage_eval.

Run with: pytest -q
Covers verdict-stability summarising, including unparseable (None) verdicts.
"""
import math

from coverage_eval import summarise_stability


def test_unanimous_is_not_flipped():
    s = summarise_stability(["SHUTDOWN", "SHUTDOWN", "SHUTDOWN"])
    assert s["flipped"] is False
    assert s["majority"] == "SHUTDOWN"
    assert s["agreement"] == 1.0
    assert s["decided"] == 3


def test_mixed_verdicts_flip():
    s = summarise_stability(["SHUTDOWN", "KEEP", "SHUTDOWN"])
    assert s["flipped"] is True
    assert s["majority"] == "SHUTDOWN"
    assert math.isclose(s["agreement"], 2 / 3)


def test_none_is_ignored():
    s = summarise_stability(["KEEP", None, "KEEP"])
    assert s["flipped"] is False
    assert s["majority"] == "KEEP"
    assert s["decided"] == 2  # the None is not counted
    assert s["agreement"] == 1.0


def test_all_none_returns_no_decision():
    s = summarise_stability([None, None])
    assert s["decided"] == 0
    assert s["majority"] is None
    assert s["flipped"] is False
    assert math.isnan(s["agreement"])


def test_tie_breaks_to_shutdown_the_safe_verdict():
    s = summarise_stability(["SHUTDOWN", "KEEP"])
    assert s["flipped"] is True
    assert s["majority"] == "SHUTDOWN"  # tie favours the safe call
    assert s["agreement"] == 0.5
