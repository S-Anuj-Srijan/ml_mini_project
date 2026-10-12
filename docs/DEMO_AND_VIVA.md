# Live demonstration and review preparation

## Five-minute demonstration

1. Explain the problem: infer terrain from measured robot motion and wheel response.
2. State the scope: inspired by the supplied tactile paper, implemented on real BorealTC IMU/wheel data because the original SAIL-R archive was not supplied or located.
3. Start: python -m streamlit run app.py
4. In Live prediction, select a terrain, held-out example and model. Explain the recorded/predicted labels and provenance ID. Show both correct and incorrect predictions when available.
5. Switch among angular velocity, acceleration, current and wheel-speed plots. Each sample contains 1.7 seconds of real measurements.
6. Open Evaluation. Explain macro-F1, the majority baseline, confusion matrices and why validation performance selects the default model.
7. Show terrain/data.py (whole-run split), terrain/models.py (CNN) and terrain/train.py (training-only normalization and evaluation).
8. Show results/metrics.json. Explain limitations instead of promising deployment accuracy.

Offline fallback: python -m terrain.predict --model svm --sample 0

Saved models and 50 real held-out examples work without downloading the dataset again.

## Questions to prepare

**Why this dataset?** It contains public real robot/terrain interaction measurements and preserves the classification problem. Its sensor modality differs from the tactile paper.

**Why a 1D CNN?** Temporal convolutions detect local patterns with shared weights. Pooling reduces sequence length, and final layers classify the representation.

**What features?** Mean, standard deviation, min, max, RMS and first-difference RMS for each of ten channels: 60 features.

**What changes in the hybrid?** Its 240 learned features are concatenated with 60 standardized statistics before a 64-unit hidden layer and five logits.

**Why an SVM?** A useful nonlinear, relatively low-data baseline. Neural networks do not automatically outperform it.

**Why macro-F1?** Classes contribute equally, unlike accuracy, which can be dominated by common classes or long recordings. Balanced accuracy and per-class precision/recall are also reported.

**What leakage is prevented?** Runs stay in one partition; windows do not overlap; scaling is fitted on training data; hyperparameters/checkpoints use validation only. Prediction never receives the terrain label.

**Are test windows independent?** Not fully: windows from one recording share conditions. Run separation does not replace grouped cross-validation.

**Why can validation and test differ?** Different recording durations, conditions and per-class support. Asphalt has only five validation windows.

**Is this real-time robot control?** No. It is offline fixed-window classification demonstrated interactively. Deployment timing and gait adaptation have not been evaluated.

**Why not claim the paper's accuracy?** Different robots, inputs, labels and protocols make results non-comparable.

**What comes next?** Multiple seeds, grouped cross-validation, sensor ablations, calibration and genuine tactile data.

## Individual contributions

The course evaluates individual contribution. Each teammate should record their actual work and understand the submitted implementation. Do not invent equal contributions or claim tasks not performed. Names, student IDs, section and team number were not supplied and have not been fabricated.

## User-owned submission steps

- Create the presentation using this project's measured results.
- Create a **private** GitHub repository, push the prepared files and grant faculty/TA access.
- Add any faculty-required identifying details.
- Submit the PDF length faculty expects: the brief says "One-Page" in its heading but requests two pages in the body. Both versions are provided.
- The brief's deadline is 10 October 2026, 11:59 PM.
