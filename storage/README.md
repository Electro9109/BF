# PARSE storage scaffold

This directory establishes the first storage placeholders needed for a shared,
context-oriented resource catalogue. The current implementation stays intentionally
small: it reserves space for private operational data, public knowledge, shared
metadata manifests, and derived artifacts without requiring a full retrieval or
indexing pipeline.

## Layout

- `private/` — approved internal operational resources and controlled plant data.
- `public/` — externally sourced or public reference resources that are not copied
  into the repository as original content.
- `catalogue/` — shared manifest metadata and registry-style references.
- `normalized/` — extracted or normalized content generated from a source while
  preserving the original resource separately.
- `derived/` — analysis outputs, indexes, and other reproducible artifacts.

This scaffold documents the intended boundaries. It does not yet implement
unified retrieval, automatic public-resource ingestion, or a full shared
database.
