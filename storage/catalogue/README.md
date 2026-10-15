# Shared catalogue and metadata

`sources.sqlite3` is the local SQLite catalogue created by
`parse.core.source_catalogue.SourceCatalogue`. It stores compact
`ResourceManifest` records, source-location aliases, processing issues, and
references to derived `KnowledgeItem` artifacts. It does not store source
contents or dataframe payloads.

Original bytes are retained under `storage/private/originals/`; normalized
documents and tabular data are written under `storage/normalized/`. The
catalogue reuses PARSE Core contracts rather than defining another source
schema. Runtime databases are local and ignored by Git.
