# Transcript-length normalization reference

Generated with R 4.5.3 and DESeq2 1.50.2 from the existing `datasets/synthetic/` counts and metadata, unchanged from upstream `caf027f`.
`generate_reference.R` constructs deterministic positive lengths matching the test's `varying_transcript_lengths` helper and calls DESeq2 for ratio and poscounts normalization.
The CSVs contain gene-by-sample normalization factors and normalized counts; the Python test transposes them and compares with `rtol=1e-8, atol=1e-8`.
Regenerate from the repository root with `Rscript tests/data/transcript_length/generate_reference.R`; ordinary tests do not require R.
These fixtures cover normalization, not independent dispersion or Wald-result parity. No external dataset is bundled.
