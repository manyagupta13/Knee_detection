"""The LLM labeler's parsing must never crash and never invent labels."""

from __future__ import annotations

import pytest

from config import TARGET_COLUMNS
from llmlabel import FINDING_DESCRIPTIONS, build_prompt, parse_response


def test_every_target_column_has_a_description():
    assert set(FINDING_DESCRIPTIONS) == set(TARGET_COLUMNS)


def test_parses_clean_json():
    reply = '{"ACL": 1, "MCL": 0, "Effusion": null, "Fracture": 1}'
    out = parse_response(reply, TARGET_COLUMNS)
    assert out["ACL"] == 1.0
    assert out["MCL"] == 0.0
    assert out["Effusion"] is None
    assert out["Fracture"] == 1.0


def test_tolerates_prose_around_the_json():
    reply = 'Sure!\n```json\n{"ACL": 1}\n```\nHope that helps.'
    assert parse_response(reply, TARGET_COLUMNS)["ACL"] == 1.0


def test_accepts_string_and_bool_values():
    out = parse_response('{"ACL": "yes", "MCL": true, "Effusion": "absent"}', TARGET_COLUMNS)
    assert out["ACL"] == 1.0 and out["MCL"] == 1.0 and out["Effusion"] == 0.0


def test_case_insensitive_keys():
    assert parse_response('{"acl": 1, "pf oa": 1}', TARGET_COLUMNS)["ACL"] == 1.0
    assert parse_response('{"acl": 1, "pf oa": 1}', TARGET_COLUMNS)["PF OA"] == 1.0


@pytest.mark.parametrize("reply", ["", "no json here", "{broken", "[]", "{}", "null"])
def test_garbage_yields_all_none_not_an_exception(reply):
    out = parse_response(reply, TARGET_COLUMNS)
    assert set(out) == set(TARGET_COLUMNS)
    assert all(v is None for v in out.values())


def test_out_of_range_values_are_rejected():
    """A model answering '2' or '0.5' must abstain, not be coerced."""
    out = parse_response('{"ACL": 2, "MCL": 0.5, "Effusion": -1}', TARGET_COLUMNS)
    assert out["ACL"] is None and out["MCL"] is None and out["Effusion"] is None


def test_unknown_keys_are_ignored():
    out = parse_response('{"Meniscus": 1, "ACL": 1}', TARGET_COLUMNS)
    assert out["ACL"] == 1.0
    assert "Meniscus" not in out


def test_prompt_contains_the_report_and_all_findings():
    prompt = build_prompt("Rotura del menisco interno.", TARGET_COLUMNS)
    assert "Rotura del menisco interno." in prompt
    for col in TARGET_COLUMNS:
        assert col in prompt
    # the rule that matters most for label quality
    assert "not mentioned" in prompt.lower()


def test_long_reports_are_truncated():
    assert len(build_prompt("x" * 50_000, TARGET_COLUMNS)) < 12_000
