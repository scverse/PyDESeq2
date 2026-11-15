"""PyDESeq2's logger, the ``scverse.pydeseq2`` child of the shared scverse logger.

One setting controls PyDESeq2 and every other package on the shared logger::

    from scverse_misc.logging import config

    config.verbosity = "info"  # default "warning"
"""

from scverse_misc.logging import get_logger

logger = get_logger("pydeseq2")
