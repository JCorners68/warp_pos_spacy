# Code Review

Review scope: migrate the Super-Neg `sbd_accelerator` Rust/PyO3 prototype into `warp_pos_spacy` as a standalone public crate.

## Findings Addressed

1. Original crate had no Rust unit tests.

   Added tests for deterministic hashing, boundary padding, invalid anchors, metric calculations, and parallel count merging.

2. Invalid anchors were implicitly padded.

   Padding is useful for window boundaries, but an anchor outside the sentence is bad input. `hash_window` and `accumulate_and_score` now reject empty POS sequences and out-of-range anchors with Python `ValueError`.

3. Public positioning could overclaim against spaCy.

   README now states the important distinction: spaCy still does POS tagging and parsing. Rust accelerates POS-window hashing, accumulation, and diversity scoring.

4. PyO3 macro expansion triggers a known clippy `useless_conversion` warning for Python result wrappers.

   Documented a crate-level allow for that one lint because the warning is emitted through the PyO3 macro expansion. The rest of the warning set still runs under `clippy -D warnings`.

5. The original README was too thin for reuse.

   Added architecture, API, build, example, performance, and development sections.

## Residual Risks

1. The benchmark claim is workload-specific.

   The 1,500x speedup should be repeated on a fixed public benchmark before publishing a formal release announcement.

2. Dominant species hash is stable but opaque.

   This is fine for fast gating, but user-facing diagnostics may eventually need a reverse map from hash to POS-window string.

3. Python package metadata is intentionally minimal.

   Before a PyPI release, add project URLs, license files, and a formal maturin release workflow.

## Recommended Next Steps

1. Add a microbenchmark script with a fixed synthetic POS corpus.
2. Add Python tests that import the built extension through `pytest`.
3. Add GitHub release automation with `PyO3/maturin-action`.
4. Consider a helper that returns both `dominant_hash` and the corresponding POS window when callers provide a reverse lookup map.
