"""Guard tests against stale or removed module paths appearing in documentation."""

import re
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

# Markdown files to guard
DOC_PATHS = [
    ROOT_DIR / "README.md",
    ROOT_DIR / "BF_BASELINE.md",
    ROOT_DIR / "PARSE_ARCHITECTURE.md",
    ROOT_DIR / "KAGGLE_TRAINING.md",
]

# Obsolete / moved module paths that must not be referenced as active modules
OBSOLETE_MODULE_PATTERNS = [
    r"\bml/feature_processing\.py\b",
    r"\bml\.feature_processing\b(?!\s*\(formerly)",
    r"\bdata/sinter_schemas\.py\b",
    r"\bdata\.sinter_schemas\b(?!\s*\(formerly)",
    r"\bconfig/sinter\.py\b",
    r"\bconfig\.sinter\b(?!\s*\(formerly)",
]


def test_documentation_does_not_reference_obsolete_module_paths():
    """Verify markdown docs do not point users to moved or removed modules."""
    violations = []
    for doc in DOC_PATHS:
        if not doc.exists():
            continue
        text = doc.read_text(encoding="utf-8")
        for line_no, line in enumerate(text.splitlines(), start=1):
            for pattern in OBSOLETE_MODULE_PATTERNS:
                if re.search(pattern, line):
                    violations.append(f"{doc.name}:{line_no}: {line.strip()}")

    assert not violations, f"Obsolete module paths referenced in documentation:\n" + "\n".join(violations)
