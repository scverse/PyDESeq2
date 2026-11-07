# Dispersion estimates of DESeq2 for one case directory of tests/data/unit_b.
#
# Usage, from the case directory: Rscript ../generate.R "<design formula>"
# Reads counts.csv (genes x samples) and metadata.csv; writes r_dispersions.csv
# (one row per gene: gene-wise, trend, MAP and final dispersions, iterations and
# outlier flags) and r_dispersion_fit.csv (size factors, trend coefficients,
# varLogDispEsts and the prior variance). The first line of each output names the
# DESeq2 and R versions. See README.md for how the inputs were made.
suppressMessages(library(DESeq2))
design <- as.formula(commandArgs(trailingOnly = TRUE)[1])
counts <- as.matrix(read.csv("counts.csv", row.names = 1, check.names = FALSE))
coldata <- read.csv("metadata.csv", row.names = 1)
coldata$condition <- factor(coldata$condition)
dds <- DESeqDataSetFromMatrix(counts, coldata, design)
dds <- estimateSizeFactors(dds)
dds <- estimateDispersions(dds, fitType = "parametric", quiet = TRUE)
m <- mcols(dds)
f <- dispersionFunction(dds)
header <- sprintf("# DESeq2 %s, %s", packageVersion("DESeq2"), R.version.string)

write_with_header <- function(df, path) {
	writeLines(header, path)
	suppressWarnings(write.table(df, path, sep = ",", append = TRUE, quote = FALSE,
		col.names = NA, row.names = TRUE))
}

write_with_header(data.frame(
	dispGeneEst = m$dispGeneEst,
	dispGeneIter = m$dispGeneIter,
	dispFit = m$dispFit,
	dispMAP = m$dispMAP,
	dispIter = m$dispIter,
	dispOutlier = m$dispOutlier,
	dispersion = m$dispersion,
	row.names = rownames(dds)
), "r_dispersions.csv")

coefs <- attr(f, "coefficients")
fit <- data.frame(
	value = c(sizeFactors(dds), coefs[["asymptDisp"]], coefs[["extraPois"]],
		attr(f, "varLogDispEsts"), attr(f, "dispPriorVar")),
	row.names = c(paste0("size_factor_", colnames(dds)), "a0", "a1",
		"varLogDispEsts", "dispPriorVar")
)
write_with_header(fit, "r_dispersion_fit.csv")
cat(header, "\n")
