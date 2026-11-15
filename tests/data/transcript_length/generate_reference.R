# Run from the repository root with R 4.5.3 and DESeq2 1.50.2.
suppressPackageStartupMessages(library(DESeq2))
counts <- as.matrix(read.csv("datasets/synthetic/test_counts.csv", row.names = 1))
metadata <- read.csv("datasets/synthetic/test_metadata.csv", row.names = 1)
# Match varying_transcript_lengths(): deterministic sample-by-gene length changes.
lengths <- sweep(
    1 + 0.2 * outer(seq(-1, 1, length.out = nrow(counts)),
                    seq(-1, 1, length.out = ncol(counts))),
    1, 800 + 100 * (seq_len(nrow(counts)) - 1), "*"
)
dimnames(lengths) <- dimnames(counts)
for (method in c("ratio", "poscounts")) {
    dds <- DESeqDataSetFromTximport(
        list(counts = counts, length = lengths, countsFromAbundance = "no"),
        colData = metadata, design = ~condition
    )
    dds <- estimateSizeFactors(dds, type = method)
    prefix <- file.path("tests/data/transcript_length", paste0("r_", method))
    write.csv(normalizationFactors(dds), paste0(prefix, "_factors.csv"))
    write.csv(counts(dds, normalized = TRUE), paste0(prefix, "_counts.csv"))
}
