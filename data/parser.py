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
_HEADING_RE = re.compile(r"^(#{1,4} .+|[A-Z][A-Z0-9 \-\/]{3,})$", re.MULTILINE)

# Matches "SECTION: <name>" markers (new format)
_SECTION_MARKER_RE = re.compile(r"^SECTION:\s*(.+?)\s*$", re.MULTILINE | re.IGNORECASE)


def _is_junk_section(section_label: str) -> bool:
    """Return True if this section label is in the junk-heading set."""
    label = section_label.lower().strip()
    return any(label == j or label.startswith(j) for j in JUNK_HEADINGS)


def _split_by_section_markers(text: str) -> list[dict]:
    """
    Split text on "SECTION: <name>" markers.

    Returns a list of {"section": <label>, "content": <body>} dicts,
    excluding junk sections and trivially short bodies.
    """
    parts = _SECTION_MARKER_RE.split(text)
    # parts = [preamble, name1, body1, name2, body2, ...]

    sections = []
    for i in range(1, len(parts), 2):
        section_name = parts[i].strip().lower()
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""

        if _is_junk_section(section_name):
            continue
        if len(content.split()) <= 8:
            continue

        sections.append({"section": section_name, "content": content})

    return sections


def _split_by_headings(text: str) -> list[dict]:
    """
    Legacy splitter: split on ALL-CAPS / markdown headings rather than
    SECTION: markers. Used as a fallback for files not yet migrated to
    the SECTION: format.

    Returns a list of {"section": <label>, "content": <body>} dicts.
    """
    boundaries = [m.start() for m in _HEADING_RE.finditer(text)]

    if len(boundaries) < 2:
        # No clear heading structure at all - treat whole file as one
        # section per non-trivial paragraph, labelled "body".
        sections = []
        for para in re.split(r"\n\s*\n", text):
            clean = " ".join(para.split())
            if len(clean.split()) > 8:
                sections.append({"section": "body", "content": clean})
        return sections

    boundaries.append(len(text))
    sections = []
    for i in range(len(boundaries) - 1):
        block = text[boundaries[i]: boundaries[i + 1]].strip()
        if not block:
            continue

        first_line = block.splitlines()[0].strip().lower().lstrip("# ")
        section_label = first_line[:40]
        body = "\n".join(block.splitlines()[1:]).strip()
        body = " ".join(body.split())

        if _is_junk_section(section_label):
            continue
        if len(body.split()) <= 8:
            continue

        sections.append({"section": section_label, "content": body})

    return sections


def split_into_sections(text: str) -> list[dict]:
    """
    Split file text into semantic sections.

    Tries SECTION:-marker format first. If the file has no SECTION:
    markers at all, falls back to legacy heading-based splitting so
    older files keep working during migration.

    Returns a list of {"section": <label>, "content": <body>} dicts,
    with junk sections (REFERENCES, LIMITATIONS-for-explanatory, etc.
    handled separately by the reranker - only true junk headings like
    REFERENCES are dropped here) already removed.
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