# BEATs upstream source and local integration

The `original/` directory contains code derived from:
https://github.com/microsoft/unilm/tree/master/beats

The upstream-derived files are `BEATs.py`, `backbone.py`, `modules.py`, `Tokenizers.py`, and `quantizer.py`.
Their original filenames are retained.
See `original/LICENSE` and `original/NOTICE.md` for licensing and notices.

Import statements in these files were modified to use package-relative imports.
`extract_features_from_fbank` was added to BEATs.py.
