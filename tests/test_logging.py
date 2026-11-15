import logging

import pytest

from pydeseq2.datasets import load_example_data
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

LOGGER = "scverse.pydeseq2"


def run_analysis(quiet=False):
    counts_df = load_example_data(
        modality="raw_counts",
        dataset="synthetic",
        debug=False,
    )

    metadata = load_example_data(
        modality="metadata",
        dataset="synthetic",
        debug=False,
    )

    dds = DeseqDataSet(counts=counts_df, metadata=metadata, design="~condition", quiet=quiet)
    dds.deseq2()
    ds = DeseqStats(dds, contrast=["condition", "B", "A"], quiet=quiet)
    ds.summary()


def test_default_output(caplog, capsys):
    """Test that only the results are shown at the default verbosity."""
    run_analysis()

    assert [record for record in caplog.records if record.name == LOGGER] == []
    assert capsys.readouterr().out.startswith("Log2 fold change & Wald test p-value: condition B vs A")


@pytest.mark.parametrize("quiet", [False, True])
def test_status_updates(caplog, capsys, quiet):
    """Test that status updates are logged at the info level, and that quiet suppresses all output."""
    with caplog.at_level(logging.INFO, logger=LOGGER):
        run_analysis(quiet=quiet)

    messages = [record.getMessage() for record in caplog.records if record.name == LOGGER]
    if quiet:
        assert messages == []
        assert capsys.readouterr().out == ""
    else:
        assert "Fitting size factors..." in messages
        assert capsys.readouterr().out != ""
