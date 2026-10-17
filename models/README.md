# Versioned ML bundles

The current workflow loads a complete bundle by an explicit `--bundle` path.
Training status and measured quality: [training report](../reports/training_report.md).
Installation and export: [ML handoff](../docs/handoffs/ml-to-inference.md).

The adjacent JSON files and `checksums.sha256` describe the historical
`db81e578` artifact from the initial import. They are not the metadata for
the new development bundle or the preserved v3 fallback. Use the sidecars
inside the chosen bundle together with its graph.

Model weights are distributed outside ordinary Git. The export command
creates a new directory, verifies every artifact and never replaces an
existing bundle.
