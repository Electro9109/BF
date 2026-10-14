"""
data/parser.py - Splits raw .txt file content into semantic chunks.

REPAIR NOTES (from the previous broken edit)
─────────────────────────────────────────────
The previous version switched _split_into_sections() to split on
"SECTION: <name>" markers instead of ALL-CAPS headings. That's a valid
approach IF every .txt file in docs/ has been restructured to use
explicit SECTION: markers (consistent with the corpus-restructuring
advice given earlier in this project).

This version keeps the SECTION:-marker approach but:
  1. Re-applies the junk filter (_is_junk_section), which the rewrite
     silently dropped - REFERENCES/LIMITATIONS-style sections would
     otherwise re-enter the index.
  2. Falls back to the old heading-based splitter for any file that
     does NOT yet use SECTION: markers, so you can migrate files
     incrementally rather than all-at-once.

If you've already converted every file in docs/ to SECTION: markers,
the fallback path is simply never used - no harm in keeping it.
"""

import re

from config.retrieval import JUNK_HEADINGS

# Matches lines that look like section headings (legacy format):
#   ALL CAPS LINES  (e.g. "EFFECT ON Tm-Ts")
#   Markdown headers  (e.g. "## Overview")
_MARKDOWN_HEADING_RE = re.compile(r"^#{1,4}\s+\S.+$")
_UPPERCASE_HEADING_RE = re.compile(r"^[A-Z][A-Z0-9 \-/]{3,}$")
_KNOWN_HEADING_RE = re.compile(
    r"^(?:overview|key mechanisms|advantages|limitations|references|"
    r"effect on T(?:s|m|m-Ts))(?:\s*[\(:-].*)?$",
    re.IGNORECASE,
)

# Matches "SECTION: <name>" markers (new format)
_SECTION_MARKER_RE = re.compile(r"^SECTION:\s*(.+?)\s*$", re.MULTILINE | re.IGNORECASE)


def _is_junk_section(section_label: str) -> bool:
    """Return True if this section label is in the junk-heading set."""
    label = section_label.lower().strip()
    return any(label == j or label.startswith(j) for j in JUNK_HEADINGS)


def _is_heading_line(line: str) -> bool:
    stripped = line.strip()
    return bool(
        _MARKDOWN_HEADING_RE.fullmatch(stripped)
        or (
            _UPPERCASE_HEADING_RE.fullmatch(stripped)
            and len(stripped) <= 48
            and len(stripped.split()) <= 5
        )
        or _KNOWN_HEADING_RE.fullmatch(stripped)
    )


def _append_section(sections: list[dict], section_label: str, content: str) -> None:
    clean_content = " ".join(content.split())
    if clean_content and not _is_junk_section(section_label):
        sections.append({"section": section_label, "content": clean_content})


def _split_by_section_markers(text: str) -> list[dict]:
    """
    Split text on "SECTION: <name>" markers.

    Returns a list of {"section": <label>, "content": <body>} dicts,
    excluding junk sections and empty bodies.
    """
    parts = _SECTION_MARKER_RE.split(text)
    # parts = [preamble, name1, body1, name2, body2, ...]

    sections = []
    if parts[0].strip():
        _append_section(sections, "introduction", parts[0])
    for i in range(1, len(parts), 2):
        section_name = parts[i].strip().lower()
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""

        _append_section(sections, section_name, content)

    return sections


def _split_by_headings(text: str) -> list[dict]:
    """
    Legacy splitter: split on ALL-CAPS / markdown headings rather than
    SECTION: markers. Used as a fallback for files not yet migrated to
    the SECTION: format.

    Returns a list of {"section": <label>, "content": <body>} dicts.
    """
    boundaries = [
        match.start()
        for match in re.finditer(r"^.*$", text, re.MULTILINE)
        if _is_heading_line(match.group())
    ]

    if not boundaries:
        # No heading structure: preserve each non-empty paragraph as body.
        sections = []
        for para in re.split(r"\n\s*\n", text):
            _append_section(sections, "body", para)
        return sections

    sections = []
    if text[:boundaries[0]].strip():
        _append_section(sections, "introduction", text[:boundaries[0]])

    boundaries.append(len(text))
    for i in range(len(boundaries) - 1):
        block = text[boundaries[i]: boundaries[i + 1]].strip()
        if not block:
            continue

        first_line = block.splitlines()[0].strip().lower().lstrip("# ")
        section_label = first_line[:40]
        body = "\n".join(block.splitlines()[1:]).strip()
        _append_section(sections, section_label, body)

    return sections


def split_into_sections(text: str) -> list[dict]:
    """
    Split file text into semantic sections.

    Tries SECTION:-marker format first. If the file has no SECTION:
    markers at all, falls back to legacy heading-based splitting so
    older files keep working during migration.

    Returns a list of {"section": <label>, "content": <body>} dicts,
    with junk headings such as REFERENCES already removed. Short non-empty
    sections are retained because thresholds, definitions, and warnings can
    be meaningful evidence.
    """
    if _SECTION_MARKER_RE.search(text):
        return _split_by_section_markers(text)
    return _split_by_headings(text)


def extract_topic(text: str, filename_stem: str) -> str:
    """
    Extract the canonical topic name for a file.

    Priority:
      1. Explicit  TOPIC: <name>  line in the first 10 lines of the file.
         Example:  TOPIC: sinter
      2. Filename stem as fallback (e.g. 'sinter' from 'sinter.txt').

    NOTE: The filename fallback is fragile - a rename silently breaks
    topic detection for that file. Always prefer the explicit header.
    """
    for line in text.splitlines()[:10]:
        m = re.match(r"^TOPIC[:\s]+(.+)$", line.strip(), re.IGNORECASE)
        if m:
            return m.group(1).strip().lower()
    return filename_stem.lower()


def strip_topic_header(text: str) -> str:
    """Remove the leading 'TOPIC: <name>' line (if present) before chunking."""
    return re.sub(
        r"^TOPIC[:\s]+.+?\n",
        "",
        text,
        count=1,
        flags=re.IGNORECASE,
    )