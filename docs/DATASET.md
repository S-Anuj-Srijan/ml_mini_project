# Data provenance and scope

The assignment folder initially contained only its instructions, the Choi/Thomasson paper and poster. The paper acknowledges receiving SAIL-R data from collaborators; it does not provide a download. Searches for the paper, authors and SAIL-R dataset did not locate the original archive. This does not establish that it is unavailable everywhere.

The selected alternative is **BorealTC**, real measurements collected with a Husky A200 wheeled robot. The user authorized selecting and downloading a suitable dataset.

- Source: https://github.com/norlab-ulaval/BorealTC
- Project: https://norlab-ulaval.github.io/BorealTC/
- Revision: 0146dcd9fa08c34a9075a80448ac04c0a947b568
- Download date: 8 October 2026.
- Data license: the dataset README declares CC0 1.0; the repository code license is separate. The dataset LICENSE file at this revision is empty, so data/UPSTREAM_DATASET_README.md preserves the explicit license statement.
- data/source_manifest.json records upstream paths, Git blob hashes, sizes and SHA-256 hashes. The downloader verifies each blob.
- Download contains all 71 single-terrain runs (142 CSVs) plus dataset README and LICENSE. MIXED is excluded because a single run-level terrain label is inappropriate.
- Labels preserve source folders: ASPHALT, FLOORING, ICE, SANDY_LOAM, SNOW. Authors describe the loam as "silty loam"; we do not silently rename source labels.
- Channels: wx, wy, wz (angular velocity); ax, ay, az (acceleration); curL, curR (motor current); velL, velR (wheel speed).
- Supplied IMU timestamps advance by 0.01 seconds. Slower wheel measurements are held forward causally onto IMU times, restricted to common coverage.
- Whole runs are split before generating non-overlapping 1.7-second windows. Run IDs include the terrain folder because numeric IDs repeat across folders.
- A fixed seeded per-class split reserves approximately 20% of run IDs each for validation and testing. Unequal durations mean window proportions differ.
- Trailing incomplete windows are discarded. No synthetic data or augmentation is used.
- Normalization and feature scaling see training data only.
- Raw/processed data remains local and is gitignored; the downloader recreates it. Fifty held-out real windows (ten per class) are included for offline demonstration.

## Differences from the supplied paper

| Aspect | Tactile paper | This implementation |
|---|---|---|
| Robot/sensor | SAIL-R leg-mounted tactile sensors | Husky IMU and wheel sensors |
| Input | 6 channels, 69 samples | 10 channels, 170 samples |
| Labels | 8 terrains / 4 high-level groups | 5 dataset terrain labels |
| Features | 33 force, gait and robot-state features | 60 channel statistics |
| Baseline | PUK SVM | Validation-tuned RBF SVM |
| CNN dropout | 0.5 | 0.3 |
| Scope | Original tactile experiment | Sensor-time-series terrain adaptation |

The tactile paper's textual pooling description and 1368-feature claim do not yield the same dimensions under straightforward repeated halving. Here dimensions are explicit: 170 -> 85 -> 42 -> 21 -> 10; 24 channels yield 240 learned features. There is no scientifically meaningful direct accuracy comparison between these datasets or with BorealTC's published evaluation protocol.

## Citation

Damien LaRocque, William Guimont-Martin, David-Alexandre Duclos, Philippe Giguère and François Pomerleau. *Proprioception Is All You Need: Terrain Classification for Boreal Forests*. IROS 2024, pp. 11686–11693. https://doi.org/10.1109/IROS58592.2024.10801407
