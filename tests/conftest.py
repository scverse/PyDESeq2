from pathlib import Path

import pytest

import pydeseq2
from pydeseq2.default_inference import DefaultInference
from tests import helpers


def pytest_report_header(config):
    """Show which PyDESeq2 is under test and the ``n_cpus`` the unit tests use."""
    return [
        f"pydeseq2: {pydeseq2.__version__} from {Path(pydeseq2.__file__).parent}",
        f"{helpers.CPUS_ENV_VAR}: {helpers.get_test_cpus()}",
    ]


@pytest.fixture
def n_cpus():
    """``n_cpus`` for PyDESeq2 calls: ``PYDESEQ2_TEST_CPUS``, default 1."""
    return helpers.get_test_cpus()


@pytest.fixture
def inference(n_cpus):
    """A fresh ``DefaultInference`` at the test ``n_cpus``."""
    return DefaultInference(n_cpus=n_cpus)
