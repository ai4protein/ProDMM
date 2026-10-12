"""Run DMS scoring from a source checkout.

The implementation lives in :mod:`prodmm.scoring`, which is also exposed
through the installed ``prodmm-score-dms`` command.
"""

from prodmm.scoring import main, read_seq, score_mutant

__all__ = ["main", "read_seq", "score_mutant"]


if __name__ == "__main__":
    main()
