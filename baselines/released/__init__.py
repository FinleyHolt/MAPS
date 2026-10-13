"""Released CityNav checkpoints run under their own evaluators.

Unlike the seam baselines beside this package, a released checkpoint flies
its authors' own code, action space, and observation crops. This package
holds what that needs on this side of the fence: the pinned upstream
identities (``upstream.py``), the Cal City export a checkpoint can read
(``calcity_export.py`` and ``describe_instructions.py``), the launcher that
runs inside each method's own environment (``upstream_eval/launch.py``), and
the conversion of native outputs to ledger rows (``outcome_rows.py``).
``README.md`` states what is held to upstream and what is ours.
"""
