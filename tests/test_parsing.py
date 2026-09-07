import pytest

from data.parser import extract_topic, split_into_sections
from ml.condition_parser import _regex_extract, parse_condition_nl


def test_regex_extracts_complete_condition():
    result = _regex_extract(
        "CO=40%, H2=8%, N2=52%. Burden is 70% sinter and 30% ore.".lower()
    )

    assert result == {
        "co": 40.0,
        "h2": 8.0,
        "n2": 52.0,
        "sinter": 70.0,
        "ore": 30.0,
        "pellet": 0.0,
        "other": 0.0,
    }


def test_condition_parser_preserves_warnings_for_low_co():
    result = parse_condition_nl(
        "CO=20%, H2=10%, N2=70%. Burden is 100% pellet.",
        tokenizer=None,
        model=None,
    )

    assert result["test_type"] == "P"
    assert result["num_components"] == 1
    assert any("below 30" in warning for warning in result["warnings"])


def test_section_parser_discards_short_and_reference_sections():
    text = (
        "TOPIC: reducibility\n"
        "SECTION: overview\n"
        "This section contains enough technical words to remain indexed.\n"
        "SECTION: references\n"
        "A short citation.\n"
    )

    sections = split_into_sections(text.replace("TOPIC: reducibility\n", ""))

    assert [section["section"] for section in sections] == ["overview"]
    assert extract_topic(text, "fallback") == "reducibility"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("100% pellet burden", 100.0),
        ("sinter only", 100.0),
    ],
)
def test_regex_extracts_single_material_burden(text, expected):
    assert _regex_extract(text)["pellet" if "pellet" in text else "sinter"] == expected