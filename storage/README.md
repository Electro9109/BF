# PARSE storage scaffold

This directory separates original inputs, normalized content, catalogue
metadata, and derived artifacts. `parse.source_catalogue.SourceCatalogue`
registers local text and tabular files in a SQLite catalogue, preserves
original bytes, records processing issues, and writes normalized artifacts
without storing dataset rows in the catalogue.

## Layout

- `private/originals/` — byte-preserved internal and operational sources.
- `public/originals/` — byte-preserved public research, datasets, and standards
  selected during source registration.
- `catalogue/` — SQLite registry and resource metadata.
- `normalized/` — extracted or normalized content generated from a source while
  preserving the original resource separately.
- `derived/` — analysis outputs, indexes, and other reproducible artifacts.

The catalogue keeps a local, SHA-256-addressed original copy and retains
submitted source locations as aliases. Registered `.txt` artifacts are included by the existing RAG engine builder;
CSV and Excel uploads are normalized to CSV and remain available through the
catalogue for analysis. Runtime source files and databases are local and
ignored by Git. Public web crawling, shared/cloud storage, and permissions
management are not implemented.
