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


def test_section_parser_preserves_prose_before_first_explicit_marker():
    sections = split_into_sections(
        "This introductory warning applies to all operating conditions.\n"
        "SECTION: mechanism\n"
        "The reduction mechanism uses carbon monoxide under controlled conditions."
    )

    assert [section["section"] for section in sections] == ["introduction", "mechanism"]
    assert "introductory warning" in sections[0]["content"]


@pytest.mark.parametrize(
    ("heading", "content"),
    [
        ("threshold", "Threshold: 1.6 to 2.2."),
        ("definition", "Basicity is CaO divided by SiO2."),
        ("warning", "Warning: excess hydrogen increases heat demand."),
        ("overview", "A short meaningful sentence."),
    ],
)
def test_section_parser_keeps_short_technical_sections(heading, content):
    sections = split_into_sections(f"SECTION: {heading}\n{content}")

    assert sections == [{"section": heading, "content": content}]


def test_section_parser_still_discards_junk_sections():
    sections = split_into_sections(
        "SECTION: overview\nA short meaningful sentence.\n"
        "SECTION: references\nA short citation."
    )

    assert sections == [{"section": "overview", "content": "A short meaningful sentence."}]


def test_legacy_heading_parser_preserves_preamble_and_short_sections():
    sections = split_into_sections(
        "An introductory warning applies throughout this document.\n"
        "OVERVIEW\n"
        "Basicity is CaO divided by SiO2.\n"
        "REFERENCES\n"
        "A short citation."
    )

    assert [section["section"] for section in sections] == ["introduction", "overview"]
    assert "introductory warning" in sections[0]["content"]
    assert sections[1]["content"] == "Basicity is CaO divided by SiO2."


def test_legacy_heading_parser_does_not_treat_uppercase_sentence_as_heading():
    sections = split_into_sections(
        "OVERVIEW\n"
        "THIS IS AN ORDINARY SENTENCE, NOT A SECTION HEADING.\n"
        "Basicity describes a ratio of basic to acidic oxides."
    )

    assert len(sections) == 1
    assert "THIS IS AN ORDINARY SENTENCE" in sections[0]["content"]


def test_legacy_heading_parser_preserves_long_unpunctuated_uppercase_prose():
    sentence = "CO REDUCTION BECOMES FASTER AT HIGHER TEMPERATURE"
    sections = split_into_sections(
        f"OVERVIEW\n{sentence}\nThe reaction rate depends on the gas atmosphere."
    )

    assert sections == [{
        "section": "overview",
        "content": f"{sentence} The reaction rate depends on the gas atmosphere.",
    }]


def test_legacy_heading_parser_supports_markdown_headings():
    sections = split_into_sections(
        "## Mechanism\n"
        "A short technical note is preserved."
    )

    assert sections == [{
        "section": "mechanism",
        "content": "A short technical note is preserved.",
    }]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("100% pellet burden", 100.0),
        ("sinter only", 100.0),
    ],
)
def test_regex_extracts_single_material_burden(text, expected):
    assert _regex_extract(text)["pellet" if "pellet" in text else "sinter"] == expected