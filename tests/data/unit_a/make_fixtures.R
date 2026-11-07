# DESeq2 reference outputs for tests/test_unit_data*.py.
# Run from this directory, after make_counts.py: Rscript make_fixtures.R
suppressMessages(library(DESeq2))
cat("R", R.version$major, ".", R.version$minor, ", DESeq2 ",
    as.character(packageVersion("DESeq2")), "\n", sep = "")

read_dds <- function(prefix) {
	counts <- as.matrix(read.csv(paste0(prefix, "_counts.csv"), row.names = 1, check.names = FALSE))
	coldata <- read.csv(paste0(prefix, "_metadata.csv"), row.names = 1, stringsAsFactors = TRUE)
	DESeqDataSetFromMatrix(t(counts), coldata, ~condition)
}
write_samples_by_genes <- function(m, file) {
	write.csv(signif(t(m), 10), file, quote = FALSE)
}
fmt <- function(x) formatC(x, digits = 17, format = "g")

# vst: blind parametric, blind mean, and with the design (blind = FALSE).
dds <- read_dds("vst")
for (fit_type in c("parametric", "mean")) {
	vsd <- varianceStabilizingTransformation(dds, blind = TRUE, fitType = fit_type)
	write_samples_by_genes(assay(vsd), paste0("vst_blind_", fit_type, ".csv"))
}
full <- estimateDispersions(estimateSizeFactors(dds), fitType = "parametric", quiet = TRUE)
vsd <- varianceStabilizingTransformation(full, blind = FALSE)
write_samples_by_genes(assay(vsd), "vst_design_parametric.csv")

# sparse: poscounts and iterative size factors.
dds <- read_dds("sparse")
pos <- sizeFactors(estimateSizeFactors(dds, type = "poscounts"))
it <- sizeFactors(estimateSizeFactors(dds, type = "iterate"))
write.csv(data.frame(poscounts = fmt(pos), iterate = fmt(it), row.names = colnames(dds)),
          "sparse_size_factors.csv", quote = FALSE)
