# Edge IoT Analytics: Smartwatch HAR Engine

This repository hosts a Human Activity Recognition (HAR) analytics system designed for wearable sensor tracking. The architecture pairs rigorous, leakage-free sequence modeling with an interactive dual-pipeline routing interface deployed live on the web.

## Live Portals & Deployments
* **Interactive Live Dashboard Workspace:** https://sensor-activity-classifier-nzxxpcnk24uq8gyzjd4zpw.streamlit.app/
* **Production Status:**  Active / Live Operational State

---

## Executive Summary & Core Architectural Ledger

### Project Objective
The objective of this analysis was to study the relationship between dynamic kinematic sensor characteristics and user kinetic states, developing a high-performance machine learning classification model capable of predicting human movement configurations (activity: Walking vs. Running) based on high-frequency streaming smartphone telemetry to determine physical behavior.

### Preprocessing & Data Engineering Pipeline
The data lifecycle implemented a rigorous data validation and transformation pipeline to insulate downstream models from real-world telemetry noise and target leakage:
1. **Chronological Alignment:** Re-mapped raw character tracking lines into a unified, sorted timestamp index layer.
2. **Telemetry Diagnostics:** Inferred a target hardware sampling metronome clock speed of 5.00 Hz (0.20s intervals) via continuous Probability Density (PDF) and Cumulative Distribution (CDF) math modeling.
3. **Anomaly Filtering:** Instituted a strict defensive boundary shield to isolate valid binary outputs, filter out corrupted text artifacts ("unknown", "null", "-1"), and lock active classification fields into pure int64 target matrices.
4. **Noise Mitigation:** Resampled chaotic microsecond records into uniform 10-second multi-metric summary windows, dropping zero-variance telemetry packet _count columns to prevent dimensional overfitting.
5. **Outlier & Disparity Capping:** Applied robust IQR Winsorization (.clip()) to handle physical shock spikes, paired with Z-score feature standardization (StandardScaler).

---

## Dual-Pipeline Benchmarking & Champion Results

The project established a comprehensive benchmarking tournament comparing traditional edge machine learning classifiers against high-capacity deep neural networks across two distinct data frameworks:

### 1. Pipeline A: Resampled Tabular Framework (`df_analysis`)
* Evaluated 7 optimized classical classifiers via cross-validated hyperparameter grids (GridSearchCV).
* **Tabular Champion:** Support Vector Machine (SVM) achieving **99.84% Test Accuracy**, successfully capturing macro-level geometric separations across the bimodal physical features.

### 2. Pipeline B: Raw Sequence 3D Window Framework (`df_ML`)
* Formatted raw continuous records into isolated 3D sequence tensors of 50 time-steps with a 50% chronological stride overlap to completely block temporal data leakage.
* **Production Champion:** Raw **1D Convolutional Neural Network (CNN)** hitting an absolute leakage-free **99.81% Global Test Accuracy** and a perfect **100.00% Test Recall/Sensitivity** across 1,063 continuous future evaluation windows.

---

##  Repository Layout & Blueprint
* `app.py`: Interactive Streamlit multi-tab architecture script mapping slider simulations to live model logic.
* `requirements.txt`: Package constraint registry ensuring environment replication (Streamlit, TensorFlow, Scikit-Learn).
* `sensor_model.pkl` & `nn_scaler.pkl`: Serialized mathematical production weights.

---
 *Note: The underlying deep learning training jupyter notebook ledger is currently undergoing local code optimization and comment restructuring. The underlying file will be pushed directly to this public repository upon version release v1.2.0.*
