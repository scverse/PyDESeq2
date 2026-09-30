import numpy as np
import pandas as pd
import pytest

from pydeseq2.dds import DeseqDataSet


@pytest.mark.parametrize("levels", [("A", "B", "C"), (0, 1, 2)])
def test_formula_reference_level(levels):
    samples = [f"sample{i}" for i in range(6)]
    conditions = np.repeat(levels, 2)
    metadata = pd.DataFrame({"condition": conditions}, index=samples)
    counts = pd.DataFrame({"gene1": [10, 12, 20, 22, 30, 32]}, index=samples)
    reference = levels[1]

    dds = DeseqDataSet(
        counts=counts,
        metadata=metadata,
        design=f"~ C(condition, contr.treatment(base={reference!r}))",
        n_cpus=1,
    )

    np.testing.assert_array_equal(
        dds.obsm["design_matrix"],
        np.column_stack([np.ones(6), conditions == levels[0], conditions == levels[2]]),
    )
    # The third level versus the chosen reference is a single fitted coefficient.
    np.testing.assert_array_equal(
        dds.contrast(column="condition", baseline=reference, group_to_compare=levels[2]),
        [0, 0, 1],
    )


def test_ref_level_warning_points_to_formula():
    samples = [f"sample{i}" for i in range(4)]
    counts = pd.DataFrame({"gene1": [10, 12, 20, 22]}, index=samples)
    metadata = pd.DataFrame({"condition": ["A", "A", "B", "B"]}, index=samples)

    with pytest.warns(DeprecationWarning, match=r"design=.*contr\.treatment"):
        dds = DeseqDataSet(
            counts=counts,
            metadata=metadata,
            design="~condition",
            ref_level=["condition", "B"],
            n_cpus=1,
        )

    # The deprecated argument still does not change the design.
    assert dds.obsm["design_matrix"].columns.tolist() == ["Intercept", "condition[T.B]"]
