# Provenance

These are the original, unmodified scripts as executed to produce the numbers
reported in the paper. They are kept byte-for-byte as they ran, including the
hard-coded absolute path used to share machinery between them.

They are **not** the supported entry points. Use `scripts/` instead, which is a
refactor of these files with identical numerical behaviour. The regression tests
in `tests/` pin the outputs of `scripts/` to the values reported in the paper.
