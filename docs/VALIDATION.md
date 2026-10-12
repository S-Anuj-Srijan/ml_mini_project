# Verification record

Verified on 8 October 2026 with Python 3.13.9 and the direct dependency versions in requirements-tested.txt.

- Download: 144 upstream files verified against pinned Git blob hashes; SHA-256 recorded in data/source_manifest.json.
- Data: 71 single-terrain recordings, 4,155 windows; train/validation/test recording IDs disjoint.
- Training: completed all three models plus majority baseline; validation-selected model is cnn_features.
- Evaluation: 930 test windows; hybrid accuracy 0.8634408602, macro-F1 0.8454988125.
- Saved inference: predictions for all 50 demonstration windows match saved test predictions for all three models.
- Tests: python -m pytest -q -> 8 passed in 32.57 seconds.
- App tests: all models, terrain switching, and held-out sample selection pass without app exceptions.
- PDF check: one-page alternative has exactly 1 page; main write-up has exactly 2 pages. Every rendered page was visually inspected for clipping, overlap and readability.
- Source syntax: compileall completed successfully.
- Original supplied PDFs were retained unchanged.
- No remote repository was created/pushed and no slide deck was produced.

The final app check removed an unnecessary Pandas Styler import that caused very slow Matplotlib initialization in a Streamlit script thread. Numeric formatting now uses rounded plain DataFrames.

These checks establish functionality on this environment and split, not reproduction of the tactile paper, statistical significance or real robot deployment readiness.
