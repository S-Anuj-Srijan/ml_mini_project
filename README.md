# Terrain classification from real robot sensor data

UE24CS352A Machine Learning mini-project, inspired by **Terrain Classification for Small Legged Robots Using Deep Learning on Tactile Data** (Choi and Thomasson).

**Scope:** a real-data adaptation on BorealTC. Inputs are IMU and wheel measurements from a Husky wheeled robot, not the original SAIL-R tactile sensors. No synthetic data or copied paper accuracy is used. See [dataset provenance](docs/DATASET.md).


## README navigation

- [Run the demo](#start-the-live-demo)
- [Reproduce the experiment](#reproduce-all-results)
- [Measured results](#actual-held-out-results)
- [Deliverables](#deliverables)
- [Full build process and technical record](#detailed-build-history-and-technical-record)
- [Failures, diagnostics and fixes](#11-failures-diagnostics-and-changes-made)
- [Testing coverage](#12-testing-progression-and-exact-coverage)

## Start the live demo

Run commands from this project directory. Saved trained models and 50 held-out real examples are included, so the demo does not require another dataset download.

    python -m venv .venv
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt
    .\.venv\Scripts\python.exe -m streamlit run app.py

Open the localhost URL printed by Streamlit. Select a model, terrain and held-out window. The app shows predictions, recorded labels, signal traces, evaluation tables and confusion matrices. No GPU or online account is required.

If using the existing Python environment in which this project was verified:

    python -m streamlit run app.py

Command-line fallback:

    python -m terrain.predict --model cnn_features --sample 0
    python -m terrain.predict --model svm --sample 25

For your own prepared input: python -m terrain.predict --input your_windows.npz --sample 0

The input NPZ must contain a finite float array X of shape (N, 10, 170), with the channel order in terrain/data.py, the same units as upstream CSVs and 100 Hz sampling. It should come from comparable sensor recordings; arbitrary sensors are not supported.

## Reproduce all results

Install dependencies as above, then use that environment's Python:

    python scripts/download_data.py
    python -m terrain.data
    python -m terrain.train --epochs 60 --seed 42
    python scripts/build_report.py
    python -m pytest -q

Downloading requires internet and approximately 90.5 MB for the selected single-terrain subset. The downloader pins the upstream revision and verifies each Git blob hash. Existing correct files are reused. Raw data is stored locally under data/raw/borealtc; prepared windows under data/processed. CPU training in the verified environment took approximately 82 seconds; other hardware/environments may differ.

The tested Python/package versions are in requirements-tested.txt and results/metrics.json. The requirements file installs the tested direct dependency versions. Python 3.13 is recommended, especially when loading the included scikit-learn checkpoint.

## Actual held-out results

One fixed recording-wise split and seed 42:

| Model | Validation macro-F1 | Test accuracy | Test macro-F1 |
|---|---:|---:|---:|
| Majority baseline | 0.042 | 23.8% | 0.077 |
| RBF SVM | 0.688 | 84.1% | 0.817 |
| 1D CNN | 0.690 | 79.6% | 0.770 |
| 1D CNN + statistical features | **0.726** | **86.3%** | **0.845** |

The hybrid model is selected by **validation macro-F1**, not test performance. Its test balanced accuracy is 86.1%. Compared with the SVM, its observed test accuracy is 2.3 percentage points higher; this is not a statistical significance claim.

Complete metrics, per-class reports, confusion matrices, selected CNN epochs and package versions are in [results/metrics.json](results/metrics.json). Training histories and the SVM validation search are also saved.

## Dataset and methodology

- 71 real recordings; five classes: asphalt, flooring, ice, loam and snow.
- 10 channels: three angular velocities, three accelerations, two motor currents and two wheel speeds.
- 170 samples per window at 100 Hz (1.7 seconds); 4,155 non-overlapping windows.
- Train: 2,654 windows; validation: 571; test: 930.
- Whole recording IDs remain separate across splits. Approximately 60/20/20% of recording counts are allocated per terrain; durations cause unequal window proportions.
- Wheel data is aligned by holding the latest preceding value on the IMU grid within common coverage.
- Both channel normalization and feature scaling are fitted on training data only.
- SVM: 60 channel statistics, balanced class weights, eight validation hyperparameter combinations.
- CNN: four Conv1d/ReLU/MaxPool/Dropout blocks, 24 channels, kernel size 5, dropout 0.3; flattened 240 features; 64 hidden units; five logits.
- Hybrid CNN: adds 60 standardized statistics to the learned representation.
- Training: Adam, learning rate 0.001, weight decay 0.0001, batch size 64, weighted cross-entropy, patience 12, maximum 60 epochs. Best epochs: raw CNN 26, hybrid 39.
- The 60 features are mean, standard deviation, minimum, maximum, RMS and first-difference RMS for each channel.

The source folder SANDY_LOAM is preserved, although the dataset publication describes silty loam. See [data/UPSTREAM_DATASET_README.md](data/UPSTREAM_DATASET_README.md) for the authors' dataset description and CC0 statement.

## Deliverables

| Path | Purpose |
|---|---|
| terrain/ | Preprocessing, training, model and inference source |
| app.py | Local live demonstration |
| scripts/download_data.py | Pinned, integrity-verified real dataset downloader |
| scripts/build_report.py | Regenerate PDF summaries from actual results |
| models/ | Trained models and training-only normalization |
| results/ | Metrics, predictions, search history and charts |
| data/demo_samples.npz | 50 held-out real sensor examples, with provenance |
| output/pdf/project_writeup_two_pages.pdf | Main write-up, following the brief's body text |
| output/pdf/project_writeup_one_page.pdf | Alternate version, following the brief's heading |
| docs/WRITEUP_SOURCE.md | Editable write-up content |
| docs/DEMO_AND_VIVA.md | Demonstration walkthrough and Q&A preparation |
| tests/test_pipeline.py | Leakage, features, model, checkpoint and app checks |

The brief contradicts itself on write-up length, so both PDF lengths are provided. Original PDFs are retained unchanged. No PPT was created and no GitHub repository was created or pushed, as requested.

## Limitations and interpretation

This is **not an exact reproduction** of the tactile paper or BorealTC's published benchmark. It uses different sensors/labels from SAIL-R and its own time-domain architecture and split.

Only one seed/split is evaluated. Asphalt has just five validation windows, limiting selection reliability. Test windows within a recording are correlated; they are not 930 independent field trials. Sensor-specific offsets, speeds and collection conditions may influence classification. No live robot control, inference latency benchmark, confidence calibration or cross-robot generalization is claimed. CNN softmax scores in the demo are explicitly uncalibrated.

Future extensions: grouped cross-validation, multiple seeds, sensor ablations and evaluation on true SAIL-R tactile recordings.

## Private GitHub submission

You will create and push the repository. Keep it **private** and invite faculty/TAs as required by the brief. The provided .gitignore excludes raw data, processed full windows and temporary files; the downloader and manifest recreate them. Saved models, demo samples and result files are small enough for normal Git.

Add actual team names/IDs if your faculty requires them. Do not invent individual contributions; each member should explain the submitted code.

See [docs/DEMO_AND_VIVA.md](docs/DEMO_AND_VIVA.md) for the review flow and submission checklist.

## References

1. Hojung Choi and Rachel Thomasson. *Terrain Classification for Small Legged Robots Using Deep Learning on Tactile Data*. Stanford project report, 2020. https://cs229.stanford.edu/proj2020spr/report/Choi_Thomasson.pdf
2. Damien LaRocque et al. *Proprioception Is All You Need: Terrain Classification for Boreal Forests*. IROS 2024, pp. 11686–11693. https://doi.org/10.1109/IROS58592.2024.10801407
3. BorealTC dataset, revision 0146dcd9fa08c34a9075a80448ac04c0a947b568. https://github.com/norlab-ulaval/BorealTC

## Detailed build history and technical record

This section records the work performed on 8 October 2026: requirement interpretation, implementation decisions, unsuccessful attempts, fixes and validation. It was reconstructed from the conversation, current source, result files and available diagnostic logs. It is a build account, not a claim that every failed tool invocation produced a permanent log. The project was built with AI assistance; teammates should record their own contributions accurately.

### 1. Starting point and assignment interpretation

The working folder contained an existing .git directory and three supplied PDFs:

| Supplied file | What it contained | How it was used |
|---|---|---|
| Guidelines and Instructions_Mini Project Assignment.pdf | Course deliverables, review requirements and evaluation criteria | Determined the source-code, README, PDF and live-demo deliverables |
| Choi_Thomasson.pdf | Seven-page terrain-classification research report | Established the motivation, tactile-sensor setup, SVM/CNN comparison and feature-fusion idea |
| Choi_Thomasson (1).pdf | One-page research poster | Cross-checked the method and terrain grouping |

There was no project implementation or raw sensor dataset in the initial directory listing. The user reserved presentation creation and GitHub repository creation/pushing for themselves. Therefore, source files were prepared locally without creating a remote repository, pushing commits, or generating a slide deck. No student names, IDs or contribution claims were invented.

The assignment's "One-Page Write-up" heading conflicts with its instruction for a "two-page summary." Both lengths were produced rather than silently choosing one interpretation. The main two-page version follows the body instruction; the alternate follows the heading. Both include the problem, dataset, approach, implementation and conclusions. The brief lists a submission deadline of 10 October 2026 at 11:59 PM.

### 2. Reading the supplied research

PDF text was extracted locally. The source report's second page was also rendered and viewed to verify the terrain grouping in Figure 3, which is less clear in extracted text:

- High friction: concrete.
- Low friction: waxed tile and laminate wood.
- Deformable: grass, wood chips, gravel and pebble.
- Granular: sand.

The paper describes six tactile channels, 69 time samples per step, 33 hand-designed features, a PUK-kernel SVM and a four-layer temporal CNN. Some of the 33 features depend on gait controls or motor measurements and cannot honestly be reconstructed from taxels alone.

Its textual architecture description also contains an ambiguity: repeated half-length pooling starting at 69 samples does not straightforwardly produce the reported 1,368 flattened features with 24 channels. The implementation therefore documents its own exact tensor dimensions instead of asserting an exact reproduction.

The supplied paper reports these original results, included here only as background:

| Original-paper input/model | Four-class accuracy | Eight-terrain accuracy |
|---|---:|---:|
| SVM | 84% | 70% |
| CNN, raw tactile data | 87% | 69% |
| CNN, raw data + all features | 97% | 80% |
| CNN, raw data + control inputs | 93% | 76% |

Those are the authors' results from Table 2, not measurements from this project. The current project's five-class numbers cannot be compared directly with them.

### 3. Dataset search, abandoned direction and final choice

The PDFs did not contain a raw dataset archive or an actionable SAIL-R download link. The report acknowledges receiving data from collaborators. Searches used the paper title, authors, SAIL-R and GitHub-related queries. They located the Stanford report/poster, a related sequential-learning report and alternative robotics datasets, but did not locate a public archive for this experiment. This is a limited search result, not proof that the original data cannot be obtained.

Before the user's dataset clarification, an initial demonstration design was drafted around synthetic six-channel signals with 69 samples, eight labels, 33 paper-inspired features and a PUK-kernel baseline. The large tool invocation carrying that draft failed to parse before it executed. That implementation was not installed, trained or used for results. The user then explicitly requested a suitable real dataset to be selected and downloaded. The final code contains no synthetic-data generator or synthetic training records. Small constant/random tensors in unit tests are test fixtures only.

BorealTC was selected because it provides real robot sensor time series and explicit terrain labels, preserving the practical terrain-classification task. Its input modality is different: a wheeled Husky's IMU and wheel-service measurements rather than SAIL-R leg-mounted tactile sensors.

The adaptation decisions were:

| Design choice | Reason |
|---|---|
| Five upstream terrain labels | Use measured labels without inventing tactile-class equivalents |
| Ten sensor channels | Use the six IMU and four wheel-service channels actually available |
| 170-sample windows | 1.7 seconds on the provided 100 Hz IMU grid |
| Sixty statistical features | Define reproducible features for the new sensors; do not fabricate the paper's unavailable gait/taxel features |
| RBF SVM | A standard nonlinear baseline implemented directly in scikit-learn; not a claimed PUK reproduction |
| Raw CNN and CNN + statistics | Preserve the research question of learned temporal features versus feature fusion |
| Whole-recording holdouts | Reduce leakage caused by adjacent windows from the same recording |

### 4. Download, version pinning and integrity checks

The GitHub tree was inspected first, rather than cloning the complete code/checkpoint repository. The upstream revision was pinned to:

    0146dcd9fa08c34a9075a80448ac04c0a947b568

The initial tree contained 146 BorealTC files totaling 105,538,299 bytes, including the MIXED recording. The final selected subset contains 144 files totaling **90,534,541 bytes**: 142 paired sensor CSVs, the dataset README and its LICENSE file. The earlier approximately 105 MB estimate described the broader inventory.

The selected recording counts are:

| Terrain folder | Recordings |
|---|---:|
| ASPHALT | 7 |
| FLOORING | 10 |
| ICE | 13 |
| SANDY_LOAM | 14 |
| SNOW | 27 |
| Total | 71 |

MIXED was excluded because the pipeline assumes a single terrain label for each recording. The dataset README describes loam as silty loam, whereas the source folder is SANDY_LOAM. Folder labels are preserved throughout processing for traceability.

The downloader:

1. Reads the saved source manifest, or obtains the pinned tree if no manifest exists.
2. Resolves each selected upstream file to data/raw/borealtc.
3. Uses six download workers, a 90-second per-request timeout and up to four attempts; retry waits are 1, 2 and 4 seconds.
4. Reuses existing file bytes when a local file exists.
5. Reconstructs the Git blob digest using SHA-1 over the bytes of "blob ", the decimal byte count, a zero byte and file content.
6. Requires that digest to match the upstream tree hash.
7. Calculates a separate SHA-256 and records it in data/source_manifest.json.

An existing corrupted file causes an integrity error; the script does not silently accept or automatically replace it. The user would need to remove that specific failed local file and retry. Network failures are retried; integrity failures are not.

The dataset README states CC0 1.0. The dataset LICENSE file at the pinned revision is empty. The upstream README was copied to data/UPSTREAM_DATASET_README.md to preserve the actual license statement. This is distinct from the upstream repository's code license.

### 5. Data validation, alignment and window creation

The preprocessing implementation is in terrain/data.py.

**CSV contract.** Each terrain recording has an imu_XX.csv file and a matching pro_XX.csv file. Required columns are checked. Measurements must be finite, and each timestamp sequence must increase strictly. Invalid values are rejected rather than imputed silently.

**Channel order.** The model requires this fixed order:

| Index | Name | Measurement family |
|---:|---|---|
| 0 | wx | Angular velocity |
| 1 | wy | Angular velocity |
| 2 | wz | Angular velocity |
| 3 | ax | Acceleration |
| 4 | ay | Acceleration |
| 5 | az | Acceleration |
| 6 | curL | Left motor current |
| 7 | curR | Right motor current |
| 8 | velL | Left wheel speed |
| 9 | velR | Right wheel speed |

Numeric values retain upstream units. The code does not apply a separate unit conversion or force calibration.

**Alignment.** The IMU is trimmed to the wheel data's time coverage. Its adjacent timestamps must match a 0.01-second increment within the implementation's tolerance. Pandas merge_asof with direction="backward" attaches the most recent wheel observation at or before each IMU timestamp. This is a causal zero-order hold, not interpolation using a future wheel observation. Any missing aligned measurement is an error. No maximum permissible wheel-data age is enforced; that is a potential future robustness check.

**Splitting.** A NumPy generator seeded with 42 permutes the sorted recordings separately within each terrain. For each class, n_holdout = max(1, round(0.2 * number_of_runs)). The first n_holdout recordings become test, the next n_holdout become validation and the remainder become training. The allocation occurs before any windows are generated. Composite group IDs such as ASPHALT/00 prevent same-number recordings in different terrain folders from colliding.

**Windowing.** Start indices increase by 170. A full slice of 170 rows is transposed from time-by-channel into channel-by-time. Trailing rows that cannot form a complete window are discarded. Windows never cross recording boundaries and do not overlap. At 100 Hz, 170 samples represent a 1.7-second sample interval; the first-to-last timestamp spacing within the window is 1.69 seconds.

The result is 43 training, 14 validation and 14 test recordings. Window counts are:

| Terrain | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| ASPHALT | 275 | 5 | 51 | 331 |
| FLOORING | 617 | 259 | 378 | 1,254 |
| ICE | 1,049 | 67 | 221 | 1,337 |
| SANDY_LOAM | 211 | 83 | 87 | 381 |
| SNOW | 502 | 157 | 193 | 852 |
| Total | 2,654 | 571 | 930 | 4,155 |

Unequal recording durations explain why a roughly 60/20/20 recording allocation is not a 60/20/20 window allocation. Asphalt's five validation windows are a limitation of this fixed split, not a corrected or hidden defect. The split was not changed to increase the observed score.

The processed NPZ contains:

| Array | Shape/type | Meaning |
|---|---|---|
| X | (4155, 10, 170), float32 | Unnormalized sensor windows |
| features | (4155, 60), float32 | Statistics extracted from original measurements |
| y | (4155,), int64 | Integer labels 0 through 4 in CLASSES order |
| groups | (4155,), strings | Composite recording IDs |
| split | (4155,), strings | train, val or test |
| start_seconds | (4155,), floating point | Window start time within the upstream recording |

All five classes must be present in every partition. The summary JSON records per-class counts and every run's partition, usable row count and window count.

### 6. Feature extraction and normalization

For each channel sequence x[0], ..., x[T-1], with T=170, six features are computed:

| Feature | Definition |
|---|---|
| Mean | sum(x[t]) / T |
| Standard deviation | sqrt(mean((x[t] - mean(x))^2)); population convention, ddof=0 |
| Minimum | min(x[t]) |
| Maximum | max(x[t]) |
| RMS | sqrt(mean(x[t]^2)) |
| First-difference RMS | sqrt(mean((x[t+1] - x[t])^2)) over 169 differences |

First-difference RMS measures adjacent-sample variation. It is not divided by the sampling interval and therefore is not a derivative in units per second.

Feature order is statistic-major: positions 0-9 are means, 10-19 standard deviations, 20-29 minima, 30-39 maxima, 40-49 RMS values and 50-59 difference RMS values. Within each block, channel order follows the table above.

CNN channel normalization uses a separate mean and standard deviation for each channel across **all training windows and all their time positions**. Stored shapes are (1, 10, 1). Standard deviations are clipped to at least 0.000001. The transform is:

    normalized_X = (X - training_channel_mean) / training_channel_std

A StandardScaler is fitted to the 60 training features. Its means and scales are saved for the hybrid CNN. The SVM instead contains its own fitted StandardScaler in its saved pipeline. Statistics are extracted before channel normalization; inference uses the same sequence.

Validation and test arrays are transformed with these already-fitted parameters. They do not influence fitted normalization. Entire recording sets are explicitly checked for overlap before training.

### 7. Models and parameter counts

**Majority baseline.** DummyClassifier(strategy="most_frequent") learns the most frequent training label, which is ICE. It provides a basic reference. Its 23.8% test accuracy reflects test ICE frequency, not a learned terrain distinction.

**SVM.** A StandardScaler -> SVC pipeline uses an RBF kernel and balanced class weights. Eight combinations were fitted on training features and scored on validation macro-F1:

| C | gamma | Validation macro-F1 |
|---:|---|---:|
| 0.1 | scale | 0.512883 |
| 0.1 | 0.01 | 0.518419 |
| 1 | scale | 0.576969 |
| 1 | 0.01 | 0.583356 |
| 10 | scale | 0.676068 |
| 10 | 0.01 | 0.639798 |
| 100 | scale | **0.688136** |
| 100 | 0.01 | 0.665426 |

The selected SVM uses C=100 and gamma="scale". Gamma="scale" is scikit-learn's feature-count/variance-based setting, not a manually fixed gamma number. The RBF kernel has the form exp(-gamma * squared_distance). Larger C places a stronger penalty on training classification errors. No SVM probability fitting was enabled; its inference output is a class label without probabilities. The selected model was not refitted on train+validation.

**CNN architecture.** Both neural models share this encoder:

| Stage | Operation | Output shape for batch B | Trainable parameters |
|---|---|---|---:|
| Input | Ten channels, 170 samples | B x 10 x 170 | 0 |
| Block 1 | Conv1d(10,24,k=5,pad=2), ReLU, MaxPool1d(2), Dropout(0.3) | B x 24 x 85 | 1,224 |
| Block 2 | Conv1d(24,24,k=5,pad=2), ReLU, MaxPool1d(2), Dropout(0.3) | B x 24 x 42 | 2,904 |
| Block 3 | Same 24-channel block | B x 24 x 21 | 2,904 |
| Block 4 | Same 24-channel block | B x 24 x 10 | 2,904 |
| Flatten | Concatenate channel/time positions | B x 240 | 0 |

Convolutions use stride 1, biases and zero padding. There is no batch normalization, residual path, recurrent layer, attention mechanism or pretrained network.

The raw CNN uses Linear(240,64), ReLU, Dropout(0.3), Linear(64,5), for **25,685 trainable parameters** in total. The hybrid concatenates 60 statistics to get B x 300, then uses Linear(300,64), ReLU, Dropout(0.3), Linear(64,5), for **29,525 parameters**. No activation is applied after the last linear layer during training, because cross-entropy expects logits. Softmax is applied only when obtaining inference scores.

### 8. Training and checkpoint selection

All models were trained on CPU. Python random, NumPy and PyTorch seeds were set to 42; CNN initialization is reset to that seed for each variant. DataLoader shuffling uses a separately seeded PyTorch generator. PyTorch runs with four CPU threads and deterministic algorithms requested. These controls improve repeatability in the tested environment; they are not a guarantee of bit-identical results across versions or hardware.

CNN configuration:

| Setting | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 0.001 |
| Weight decay | 0.0001 |
| Batch size | 64 |
| Loss | Class-weighted cross-entropy |
| Per-class weight | N_train / (5 * training_count_for_class) |
| Maximum epochs | 60 |
| Selection metric | Validation macro-F1 |
| Improvement tolerance | More than 0.000001 above the best score |
| Early-stopping patience | 12 consecutive epochs without such improvement |
| Learning-rate scheduler | None |
| Gradient clipping | None |

For each batch: zero gradients, calculate logits/loss, backpropagate and update parameters. Each epoch then switches to evaluation mode, disables gradients, evaluates the entire validation partition and records metrics. A deep copy of the best state dictionary is retained. Patience resets only on a qualifying validation macro-F1 improvement.

The raw CNN ran **38 epochs** and restored the checkpoint from **epoch 26**. The hybrid ran **51 epochs** and restored **epoch 39**. Neither reached the maximum of 60 epochs because early stopping triggered.

The plotted training loss is the sample-count-weighted average of batch weighted-cross-entropy values. Because class composition changes by batch, this is not exactly the same aggregation as evaluating one globally weighted loss over the full training partition. It is a training-progress diagnostic. Validation loss is computed on the complete validation tensor.

Final selection compares the three trained candidates' best validation macro-F1 values. The hybrid wins with 0.725570. Test predictions are generated for reporting, but test scores do not enter hyperparameter, checkpoint or default-model selection. The script logs each completed model's test result as it goes; its selection expressions use validation values exclusively.

The recorded training timer is 82.279 seconds. It starts inside main and ends when assembling metrics, so it excludes downloading, preprocessing, Python imports, final chart generation, PDF building and debugging. It is not a full project runtime or an inference-latency benchmark.

### 9. Evaluation and what the results mean

Accuracy is correct predictions divided by the number of windows. Balanced accuracy is mean per-class recall. Macro-F1 is the unweighted mean of per-class F1. Undefined precision/recall/F1 calculations use zero in the scikit-learn reports.

| Model | Test accuracy | Test balanced accuracy | Test macro-F1 |
|---|---:|---:|---:|
| Majority | 0.237634 | 0.200000 | 0.076803 |
| SVM | 0.840860 | 0.825479 | 0.816693 |
| Raw CNN | 0.795699 | 0.805536 | 0.769729 |
| Hybrid CNN | 0.863441 | 0.860666 | 0.845499 |

The hybrid correctly classifies 803 of 930 test windows; the SVM 782 and raw CNN 740. The hybrid's gain over the SVM is 21 windows, or about 2.26 percentage points. The raw CNN underperforms the SVM on test accuracy despite narrowly exceeding its validation macro-F1. This is a measured result, not an implementation failure to conceal. Feature fusion helps on this split, but a single experiment does not prove universal superiority.

Hybrid per-class test performance:

| Class | Precision | Recall | F1 | Test windows |
|---|---:|---:|---:|---:|
| ASPHALT | 0.657143 | 0.901961 | 0.760331 | 51 |
| FLOORING | 0.889169 | 0.933862 | 0.910968 | 378 |
| ICE | 0.878049 | 0.814480 | 0.845070 | 221 |
| SANDY_LOAM | 0.939759 | 0.896552 | 0.917647 | 87 |
| SNOW | 0.834286 | 0.756477 | 0.793478 | 193 |

Its confusion matrix, with true labels on rows and predicted labels on columns, is:

| True / predicted | Asphalt | Flooring | Ice | Loam | Snow |
|---|---:|---:|---:|---:|---:|
| Asphalt | 46 | 1 | 3 | 0 | 1 |
| Flooring | 3 | 353 | 1 | 1 | 20 |
| Ice | 0 | 35 | 180 | 0 | 6 |
| Loam | 4 | 3 | 0 | 78 | 2 |
| Snow | 17 | 5 | 21 | 4 | 146 |

Examples of remaining errors are ice confused with flooring and snow confused with ice/asphalt. These counts show where errors occur; they do not by themselves establish a physical explanation for each error.

### 10. Saved artifacts and inference behavior

- models/svm.joblib contains the selected scaler/SVC pipeline.
- models/cnn.pt and models/cnn_features.pt contain state_dict, extra feature count and selected epoch; they are not resumable optimizer checkpoints.
- models/normalization.npz stores channel mean/std and feature mean/scale.
- results/metrics.json stores selection, all reports, data summary, seed and selected environment versions.
- results/svm_search.json records all eight validation configurations.
- results/training_history.json records per-epoch losses, accuracy, balanced accuracy and macro-F1.
- results/test_predictions.npz stores original test indices, true labels and predictions from all four models.
- results/confusion_matrices.png and results/training_curves.png provide evaluation plots.
- data/demo_samples.npz contains the first ten test windows of each class in existing dataset order, with recording IDs, start times and original indices.

The 50 demo examples are chosen by class/order, not by whether a model predicts them correctly. They are a compact demonstration subset, not the evaluation population. All reported test metrics use 930 windows.

Inference validates sensor shape/finite values, recomputes features and selects the saved default if no model name is given. SVM inference uses its pipeline. CNN inference loads saved scaling, restores weights on CPU with weights_only=True, switches to eval mode and runs with no gradients. Dropout is disabled. The raw CNN receives an empty auxiliary-feature tensor; the hybrid receives 60 normalized statistics.

The predictor sets four PyTorch CPU threads. It reloads artifacts for each call rather than maintaining a model cache. This is adequate for the demonstration but is not an optimized deployment service. The three models plus normalization occupy 674,647 bytes; the demo NPZ occupies 210,203 bytes in this build.

The Streamlit app has three tabs:

1. **Live prediction:** model selection, terrain/example selection, predicted/true labels, match status, provenance, sensor-group plot, optional neural softmax chart and CSV export.
2. **Evaluation:** measured model metrics and the saved charts.
3. **Method & data:** processing explanation, modality differences and limitations.

The CSV download exports the currently selected sensor family for the selected window, not all ten channels. Ground-truth labels choose/display a recorded example; they are not passed into predict. Softmax bars are explicitly uncalibrated. The UI supports bundled examples; custom NPZ input is supported through the CLI.

### 11. Failures, diagnostics and changes made

The following includes development-environment failures as well as application changes. Not every failure was caused by project code.

| Stage | Observed failure or limitation | Action/change | Outcome and certainty |
|---|---|---|---|
| Initial folder inspection | PowerShell/rg reads did not return; later the old execution-cell ID no longer existed | Retried read-only inspection in a subsequent turn | Initial inspection was incomplete; no project files were recovered from that attempt |
| Shell process startup | The execution helper reported helper_unknown_error: setup refresh had errors, including on a later pytest attempt | Used the available Node runtime for local filesystem reads/writes and hidden Python subprocesses | Local work continued; the execution helper itself was not repaired |
| PDF dependency inspection | pypdf was absent; pdfplumber and PyMuPDF were installed | Used installed PyMuPDF for extraction/rendering and ReportLab for generation | No pypdf installation was required |
| First PDF extraction | Windows cp1252 output raised UnicodeEncodeError on a Unicode arrow | Reran Python with -X utf8 | Full text extraction succeeded |
| PDF import name | The fitz alias emitted a deprecation warning | Switched subsequent code to import pymupdf | Removed reliance on the deprecated alias |
| Original dataset search | No SAIL-R archive was supplied or located | Selected BorealTC after the user's real-dataset instruction | Scope explicitly became an adaptation; original data availability remains unresolved |
| Initial synthetic draft | A large embedded-JavaScript authoring call raised SyntaxError: Invalid or unexpected token before execution | Abandoned that dataset direction and changed code-writing approach | No synthetic model or result was produced |
| Later document-authoring call | Another large JavaScript authoring request failed to parse | Split the authoring into smaller calls and serialized file contents before writing | Files were written successfully; malformed drafts did not partially execute |
| GitHub metadata fetch | Node fetch returned fetch failed | Used the permissioned network command to retrieve metadata, then the approved downloader for data | Downloads succeeded; exact cause of the fetch failure was not established |
| Data labeling/license review | Loam naming differs between folders and publication; dataset LICENSE file is empty | Preserved folder names and copied the README's CC0 statement | Ambiguities are documented, not silently rewritten |
| First training run | OSError: No username set in the environment, reached through PyTorch's default cache-directory lookup; the trace also contained missing pwd on Windows | Set TORCHINDUCTOR_CACHE_DIR to the project's tmp/torch_cache before importing PyTorch | Training completed; no model mathematics changed |
| Initial checks | Six data/model checks passed, but the added full app check stalled | Isolated app testing and wrote logs to files | Core model correctness was kept separate from UI startup diagnosis |
| Long tool execution | A Node execution timed out and reset its kernel during combined checks | Recreated bindings and launched subprocesses with persistent log files | Logs survived tool timeouts; this was orchestration recovery |
| Process inspection | Get-CimInstance returned Access denied | Continued using logs from the processes already launched | Broad process enumeration was not necessary and was not fixed |
| Diagnostic faulthandler run | Timed out and emitted Windows fatal exception: access violation; exit code 3221225477 | Switched to standalone and import-time probes | The access violation's root cause was not proven; it is not presented as a fixed model bug |
| Standalone AppTest | ModuleNotFoundError: No module named 'terrain' when launched from a helper under tmp | Added the resolved app directory to sys.path before local imports in app.py | Local-package imports became robust to that launch context |
| Debug subprocess management | One attempt to reuse a subprocess binding raised Cannot read properties of undefined (reading 'kill') | Used a new named probe/log rather than assuming the old binding remained usable | Diagnostic tooling issue; no source/model state was changed by the failed call |
| Streamlit startup | AppTest exceeded its timeout; import profiling showed roughly 95.5 seconds in matplotlib._image loading through Pandas Styler | Replaced DataFrame.style.format with a plain rounded DataFrame in the evaluation table | Final all-model app test passed; the unnecessary import was removed, without claiming the underlying system import delay was universally fixed |
| Inference configuration | CPU thread count was initially left at the runtime default | Set four threads in the prediction function | Configuration matches the modest CPU demo; no formal speedup benchmark was performed |
| Dependency reproducibility | Initial requirements used broad minimum versions | Captured installed direct versions in requirements-tested.txt and made requirements.txt include it | Reduces saved-checkpoint compatibility risk; it is not a complete transitive dependency lock |
| Documentation-time model probe | Importing terrain.models directly in a fresh process raised OMP Error #15: duplicate libiomp5md.dll initialization | Retried the parameter-count probe after importing NumPy and pandas, matching the app's numerical-library import order | Probe then returned 25,685 and 29,525 parameters; no duplicate-runtime bypass was enabled, and the underlying environment conflict was not fully repaired |
| README audit | Initial download estimate retained the broader 105 MB inventory figure | Corrected the selected subset to 90,534,541 bytes and explained the estimate | Documentation correction only |

The JavaScript writing failures were avoided by treating file contents as data, using serialized strings and smaller calls. Literal Markdown/code delimiters inside large nested JavaScript strings had made the original calls fragile. This affected the authoring tools, not Python model behavior.

The final Streamlit fix intentionally kept Matplotlib available for offline training charts. It removed only a needless runtime styling dependency from the app table; the saved images still display in the demo.

### 12. Testing progression and exact coverage

First, six tests covering split behavior, features, invalid input, two CNN configurations and real-data partition integrity passed in 11.08 seconds. Additional saved-model and UI checks were then added. After the UI/import fixes, the final suite reported:

    python -m pytest -q
    8 passed in 32.57s

There are seven test functions and eight cases because the CNN shape/gradient check is parameterized over two auxiliary-feature dimensions.

| Test | What it establishes |
|---|---|
| test_recordings_never_cross_partitions | Seeded mapping is repeatable and every class receives all three partitions on a constructed run inventory |
| test_features_have_analytic_values | Constant inputs yield the expected means, zero standard deviations, RMS and zero difference RMS; feature shape is correct |
| test_bad_windows_rejected | Wrong sensor dimensions and NaN inputs are rejected |
| test_network_shape_and_gradients, extra=0 | Raw CNN emits B x 5 logits and gradients reach the first convolution |
| test_network_shape_and_gradients, extra=60 | Hybrid CNN does the same with feature concatenation |
| test_actual_data_split_is_disjoint_and_complete | Actual run sets do not overlap, all classes are present and window starts do not overlap within runs |
| test_training_normalization_and_saved_predictions | Saved channel mean/std match training-only recomputation; all three models reproduce saved predictions on all 50 demo windows |
| test_demo_app_all_models | App runs without captured exceptions, switches among all three models, selects snow and changes the example slider |

The actual-data/checkpoint tests can skip when their local artifacts are missing. Therefore, a clean checkout without raw preprocessing may not run all eight cases to completion until the reproduction commands are followed. The recorded eight-pass run used the complete prepared local project.

The suite does not directly test every malformed CSV, every downloader retry path, all feature-scaler values, every UI control or every PDF layout property. The saved-prediction test covers 50 demo windows, not a recomputation of all 930 test predictions. AppTest verifies application execution and interactions; no full browser visual-regression suite was run.

Additional checks performed:

- Source compilation with compileall for the package, scripts, app and tests.
- A command-line prediction from the saved hybrid checkpoint; the demonstrated asphalt example was predicted as asphalt.
- PDF page-count checks and visual inspection of every rendered output page.
- Git status inspection to identify prepared local files and confirm no tracked raw-data files.
- Git diff --check returned no output. Many new files were untracked at that point, so this alone is not a complete whitespace audit of all authored files.

### 13. PDF generation and visual review

scripts/build_report.py reads metrics.json rather than inserting invented scores. It builds both PDFs with ReportLab using A4 pages, Helvetica fonts, consistent teal headings, result tables, footers and page numbers. The two-page version includes the generated confusion-matrix figure. The one-page version condenses the same problem, method, result and limitations.

The PDF workflow's artifact-start marker was run before authoring the PDF outputs. This was an authoring-environment step; it is not needed by a student running the report builder.

PyMuPDF reopened both files to check their lengths: exactly one and two pages. Every page was rendered to PNG at 1.3 scale and visually reviewed for clipping, overlap, table alignment, chart readability and page structure. No visual defect requiring a further PDF revision was identified in that inspection. Generated text was also saved to docs/WRITEUP_SOURCE.md for editing.

The original supplied PDFs were not modified. Temporary page images are kept under the ignored tmp/pdfs directory. PDF regeneration overwrites the two stable output filenames.

### 14. Environment, organization and reproduction limits

The build used the existing Windows Python 3.13.9 environment with these installed direct dependencies:

| Package | Tested version | Purpose |
|---|---|---|
| NumPy | 2.3.5 | Arrays, statistics, NPZ files |
| pandas | 2.3.3 | CSV reading/alignment and demo tables |
| scikit-learn | 1.7.2 | Scaling, SVM, baseline and metrics |
| PyTorch | 2.11.0 | CNN training/inference |
| Matplotlib | 3.10.6 | Offline result figures |
| joblib | 1.5.2 | SVM pipeline serialization |
| Streamlit | 1.51.0 | Local interactive demonstration |
| ReportLab | 5.0.0 | PDF write-ups |
| pytest | 8.4.2 | Automated checks |
| PyMuPDF | 1.28.2 | PDF inspection/rendering |

During this README expansion, a direct model-only import exposed an OpenMP runtime initialization conflict in the shared environment. Importing NumPy and pandas first allowed the parameter-count probe to pass, matching the app's import order. This was a diagnostic finding, not a new model training run or proof that every import order works. No KMP_DUPLICATE_LIB_OK override was added.

A brand-new virtual-environment install was documented but was not independently executed during this build. Direct packages are pinned; transitive dependencies, the OS and hardware are not fully locked. No CUDA path was used. Saved scikit-learn models are version-sensitive; use the tested environment for the supplied checkpoint.

The root resolves from source-file locations, not an assumed system username. Training caches and diagnostics live under tmp. Raw data, full processed windows, virtual environments, Python caches, pytest caches and tmp are ignored by Git. Small checkpoints, the demo subset, result figures/JSON and PDFs are prepared for inclusion.

Re-running preprocessing overwrites its processed arrays/summary. Re-running training overwrites models, metrics, predictions, demo samples and figures. Rebuilding the reports overwrites their PDFs and Markdown source. Preserve an experiment separately before deliberately changing its configuration.

One subtle seed detail: the data-preparation CLI currently uses prepare() with its default seed 42. The training --seed argument controls training randomness but does not re-split the already prepared dataset. A different split requires explicitly calling prepare(seed=...) and then retraining; changing only --seed on training is not enough.

The diagnostic logs in tmp are local debugging evidence and are not required for running the project. Because they are gitignored, the failure account above is included in this README so the build history remains available in the submitted source.

### 15. Remaining limitations and final handoff

Completed: real-data acquisition, preprocessing, three trained candidates and baseline, held-out evaluation, saved inference, local demo, setup instructions, tests, two PDF write-ups, provenance documentation and review/Q&A guidance.

Reserved for the user: create the presentation, create/push the private GitHub repository, grant faculty/TA access and add any identifying details required by the course. The existing .git directory was not evidence that a remote repository had been created during this work.

Scientific limitations remain:

- Different sensors, robot and labels from the supplied tactile experiment.
- One fixed recording split and one training seed.
- Very limited asphalt validation support.
- Correlated windows within each recording.
- Potential collection-condition, speed and sensor-offset effects.
- No grouped cross-validation, uncertainty intervals or significance tests.
- No calibrated probabilities, live control, cross-robot test or deployment latency measurement.
- No evaluation of the excluded mixed-terrain recording.
- No architecture or feature ablation beyond the raw-versus-hybrid comparison.
- No general-purpose resampling or arbitrary sensor-schema importer.

These are explicitly reported boundaries of the completed mini-project. They should guide the presentation and viva rather than be replaced with the original paper's claims.

### 16. Verification of this README expansion

This follow-up changed README.md only. Local Markdown file links were checked and all resolved. Model parameter counts were verified against instantiated networks after the import-order retry described above; prediction totals were cross-checked against the saved confusion matrices. Dataset sizes, run counts, epoch counts, hyperparameters and test status were checked against the manifest, histories, source files and existing test log. The recorded eight-pass result belongs to the completed build; the full training/test suite was not rerun for this documentation-only change.
