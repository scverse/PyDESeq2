# Datasets

This directory stores example data which can be retrieved using the `load_example_data` function from
`pydeseq2.utils`. Only synthetic data is provided for now, but new datasets might be available in the future.

The `tests_clinical.csv` and `test_counts.csv` files were generated using a custom modification of the
`makeExampleDESeqDataSet` method of [DESeq2](https://bioconductor.org/packages/release/bioc/html/DESeq2.html),
to handle several factors.


## Folder organisation

```
PyDESeq2
│
└───datasets
        │
        └───synthetic
            │       test_clinical.csv
            │       test_counts.csv
```
