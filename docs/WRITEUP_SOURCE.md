# Terrain classification from robot sensor time series

Classify the terrain beneath a mobile robot from short sensor time series. Inspired by Choi and Thomasson's tactile-learning study [1], this project compares handcrafted statistics with learned temporal features to support terrain-aware robot navigation.

Real BorealTC measurements [2,3] provide 71 single-terrain recordings: asphalt, flooring, ice, loam and snow. Ten channels combine angular velocity (3), acceleration (3), motor current (2) and wheel speed (2). The supplied IMU grid is 100 Hz. We use 4,155 non-overlapping 170-sample windows (1.7 s), split into 2,654 training, 571 validation and 930 test windows.

The supplied PDFs contain no raw SAIL-R data, and a public download was not located. BorealTC was selected as a real, openly available alternative. It uses a wheeled Husky robot and proprioceptive sensors, not the paper's six tactile taxels; this is an adaptation, not an exact reproduction. The folder label SANDY_LOAM is preserved although the authors describe silty loam.

Recordings are assigned to train/validation/test within each terrain before windowing (seed 42). Wheel channels are aligned to IMU timestamps using the latest preceding observation within common time coverage. Normalization is fitted on training data only. Six statistics per channel (mean, standard deviation, minimum, maximum, RMS and first-difference RMS) yield 60 features.

An RBF SVM uses standardized features and balanced class weights. Validation macro-F1 selects C from {0.1, 1, 10, 100} and gamma from {scale, 0.01}. The CNN has four 24-channel, kernel-5 convolution blocks, each with ReLU, max-pooling and dropout 0.3. Its 240 learned features feed a 64-unit hidden layer and five output logits. A hybrid concatenates the 60 statistical features. CNNs use Adam (0.001), weighted cross-entropy, batch size 64 and early stopping (patience 12; at most 60 epochs).

The validation-selected model is cnn + features; it achieves 86.3% test accuracy and 0.845 test macro-F1 on unseen recordings. These are measurements from this project, not quoted paper results. The implementation demonstrates a reproducible real-data pipeline; conclusions are restricted to the selected BorealTC split.

One split and one training seed limit certainty. Recording durations and class counts are unequal; asphalt has only five validation windows. Windows from the same run remain correlated, so window count is not the number of independent trials. No confidence intervals, cross-robot transfer or live robot control are claimed. Future work: grouped cross-validation, multiple seeds and evaluation on genuine SAIL-R tactile data.

[1] H. Choi and R. Thomasson. Terrain Classification for Small Legged Robots Using Deep Learning on Tactile Data. Supplied Stanford report, 2020. cs229.stanford.edu/proj2020spr/report/Choi_Thomasson.pdf

[2] D. LaRocque et al. Proprioception Is All You Need: Terrain Classification for Boreal Forests. IROS, 2024, pp. 11686-11693. doi:10.1109/IROS58592.2024.10801407.

[3] BorealTC dataset: github.com/norlab-ulaval/BorealTC. Pinned revision 0146dcd9fa08c34a9075a80448ac04c0a947b568. Dataset README declares CC0 1.0. Retrieved 8 October 2026.