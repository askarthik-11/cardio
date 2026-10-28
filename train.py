"""Train patient-grouped MIT-BIH beat-abnormality classifiers.

MIT-BIH beat annotations are targets for abnormal-beat detection, not diagnoses
of heart disease. All reported validation predictions are made for records
held out from the corresponding training fold.
"""

from collections import Counter
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wfdb
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from scipy.signal import butter, find_peaks, peak_widths, sosfiltfilt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" / "mitbih"
PATIENTS_DIR = PROJECT_DIR / "data" / "patients"
RESULTS_DIR = PROJECT_DIR / "data" / "results"
MODEL_PATH = PROJECT_DIR / "model.pkl"
WEARABLE_MODEL_PATH = PROJECT_DIR / "wearable_model.pkl"
WINDOW_SECONDS = 10
OUTER_FOLDS = 5
INNER_FOLDS = 3
RANDOM_STATE = 42
CALIBRATION_FOLDS = 5
LOW_NORMAL_RECALL_TARGET = 0.90
HIGH_PRECISION_TARGET = 0.80

# The four paced records are omitted for AAMI-style beat-class evaluation.
PACED_RECORDS = {"102", "104", "107", "217"}

# Standard AAMI beat groups: N, S, V, F, and Q.
NORMAL_BEAT_SYMBOLS = {"N", "L", "R", "e", "j"}
ABNORMAL_BEAT_SYMBOLS = {"A", "a", "J", "S", "V", "E", "F", "/", "f", "Q"}

# The requested production target is a window with one or more abnormal beats.
PRIMARY_LABEL_RULE = "at_least_1_abnormal_beat"

OFFLINE_FEATURE_NAMES = [
    "mean_heart_rate_bpm",
    "rr_interval_mean_s",
    "sdnn_s",
    "rmssd_s",
    "pnn50_percent",
    "rr_coefficient_of_variation",
    "mean_absolute_rr_change_s",
    "qrs_width_mean_s",
    "qrs_width_std_s",
    "qrs_amplitude_mean",
    "qrs_amplitude_std",
    "signal_quality_snr_db",
    "signal_quality_flatline_ratio",
]
WEARABLE_FEATURE_NAMES = [
    "hr_1hz_mean_bpm",
    "hr_1hz_std_bpm",
    "hr_1hz_min_bpm",
    "hr_1hz_max_bpm",
    "hr_1hz_range_bpm",
    "hr_1hz_trend_bpm_per_second",
]
WINDOW_FEATURE_NAMES = [
    *OFFLINE_FEATURE_NAMES,
    *WEARABLE_FEATURE_NAMES,
    "abnormal_beat_ratio",
]
LABEL_RULES = {PRIMARY_LABEL_RULE: 1}

# Each tree model has two small, predefined alternatives. Selection happens
# inside each outer training fold, never on its held-out patient records.
RANDOM_FOREST_CONFIGS = [
    {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 1},
    {"n_estimators": 300, "max_depth": 16, "min_samples_leaf": 2},
]
XGBOOST_CONFIGS = [
    {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05},
    {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05},
]


def download_database():
    """Download MIT-BIH records, reusing complete local files when available."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PATIENTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        records = wfdb.get_record_list("mitdb")
        if not records:
            raise RuntimeError("PhysioNet returned no MIT-BIH record names.")

        # Avoid re-fetching the online file index when a previous run completed.
        missing_files = [
            record
            for record in records
            if not all(
                (DATA_DIR / f"{record}.{extension}").is_file()
                and (DATA_DIR / f"{record}.{extension}").stat().st_size > 0
                for extension in ("hea", "dat", "atr")
            )
        ]
        if missing_files:   
            print(
                f"Downloading or resuming MIT-BIH records "
                f"({len(missing_files)} records need files) ..."
            )
            wfdb.dl_database(
                "mitdb",
                dl_dir=str(DATA_DIR),
                records=records,
                annotators=["atr"],
                keep_subdirs=False,
            )
        else:
            print(f"Using all {len(records)} locally cached MIT-BIH records.")
        return records
    except Exception as error:
        raise RuntimeError(
            "Could not download MIT-BIH from PhysioNet. Check the internet "
            "connection, then run train.py again."
        ) from error


def filter_ecg(signal, sampling_rate):
    """Remove baseline wander and high-frequency noise with a 0.5-40 Hz filter."""
    nyquist = sampling_rate / 2
    if nyquist <= 40:
        raise ValueError(
            f"Sampling rate {sampling_rate} Hz is too low for a 40 Hz high cut."
        )
    coefficients = butter(
        4,
        [0.5 / nyquist, 40 / nyquist],
        btype="bandpass",
        output="sos",
    )
    return sosfiltfilt(coefficients, signal)


def extract_window_features(raw_window, filtered_window, sampling_rate):
    """Calculate rhythm, QRS, and annotation-independent quality features."""
    # Absolute amplitude lets the detector find QRS complexes on inverted leads.
    magnitude = np.abs(filtered_window)
    peak_distance = max(1, int(0.25 * sampling_rate))
    peak_prominence = max(float(np.std(magnitude)) * 0.25, 1e-8)
    r_peaks, _ = find_peaks(
        magnitude,
        distance=peak_distance,
        prominence=peak_prominence,
    )

    rr_intervals = np.diff(r_peaks) / sampling_rate
    if len(rr_intervals):
        rr_mean = float(np.mean(rr_intervals))
        heart_rate = 60.0 / rr_mean if rr_mean > 0 else 0.0
        sdnn = (
            float(np.std(rr_intervals, ddof=1))
            if len(rr_intervals) > 1
            else 0.0
        )
        rr_std = float(np.std(rr_intervals))
        rmssd = (
            float(np.sqrt(np.mean(np.square(np.diff(rr_intervals)))))
            if len(rr_intervals) > 1
            else 0.0
        )
        pnn50 = (
            float(np.mean(np.abs(np.diff(rr_intervals)) > 0.05) * 100)
            if len(rr_intervals) > 1
            else 0.0
        )
        rr_cv = rr_std / rr_mean if rr_mean > 0 else 0.0
        mean_rr_change = (
            float(np.mean(np.abs(np.diff(rr_intervals))))
            if len(rr_intervals) > 1
            else 0.0
        )
    else:
        rr_mean = heart_rate = sdnn = rmssd = pnn50 = rr_cv = mean_rr_change = 0.0

    if len(r_peaks):
        widths = peak_widths(magnitude, r_peaks, rel_height=0.5)[0] / sampling_rate
        amplitudes = magnitude[r_peaks]
        qrs_width_mean = float(np.mean(widths))
        qrs_width_std = float(np.std(widths))
        qrs_amplitude_mean = float(np.mean(amplitudes))
        qrs_amplitude_std = float(np.std(amplitudes))
    else:
        qrs_width_mean = qrs_width_std = 0.0
        qrs_amplitude_mean = qrs_amplitude_std = 0.0

    # This SNR is a signal-quality proxy: band-limited power versus removed
    # baseline/high-frequency residual power, not a clinical quality grade.
    residual = raw_window - filtered_window
    signal_power = float(np.mean(np.square(filtered_window)))
    residual_power = float(np.mean(np.square(residual)))
    snr_db = 10 * np.log10(
        max(signal_power, 1e-12) / max(residual_power, 1e-12)
    )
    differences = np.diff(raw_window)
    flatline_ratio = (
        float(np.mean(np.abs(differences) <= 1e-8)) if len(differences) else 0.0
    )

    return [
        heart_rate,
        rr_mean,
        sdnn,
        rmssd,
        pnn50,
        rr_cv,
        mean_rr_change,
        qrs_width_mean,
        qrs_width_std,
        qrs_amplitude_mean,
        qrs_amplitude_std,
        float(snr_db),
        flatline_ratio,
    ]


def derive_one_hz_heart_rate(filtered_signal, sampling_rate):
    """Interpolate beat-to-beat rates onto one sample per second."""
    magnitude = np.abs(filtered_signal)
    peak_distance = max(1, int(0.25 * sampling_rate))
    peak_prominence = max(float(np.std(magnitude)) * 0.25, 1e-8)
    r_peaks, _ = find_peaks(
        magnitude,
        distance=peak_distance,
        prominence=peak_prominence,
    )
    seconds = int(len(filtered_signal) // sampling_rate)
    if len(r_peaks) < 2 or seconds == 0:
        return np.zeros(seconds, dtype=np.float64)

    rr_intervals = np.diff(r_peaks) / sampling_rate
    valid = rr_intervals > 0
    if not np.any(valid):
        return np.zeros(seconds, dtype=np.float64)

    # Assign each R-R-derived instantaneous rate to the interval midpoint,
    # then interpolate those observations to each second's midpoint.
    rate_times = (r_peaks[:-1][valid] + r_peaks[1:][valid]) / 2
    rates_bpm = 60.0 / rr_intervals[valid]
    second_centers = (np.arange(seconds, dtype=np.float64) + 0.5) * sampling_rate
    return np.interp(second_centers, rate_times, rates_bpm)


def summarize_one_hz_heart_rate(hr_series):
    """Summarize ten once-per-second heart-rate readings for the wearable model."""
    if len(hr_series) != WINDOW_SECONDS:
        raise ValueError(
            f"Expected {WINDOW_SECONDS} one-Hz heart-rate samples, got {len(hr_series)}."
        )
    trend = float(np.polyfit(np.arange(len(hr_series)), hr_series, 1)[0])
    minimum = float(np.min(hr_series))
    maximum = float(np.max(hr_series))
    return [
        float(np.mean(hr_series)),
        float(np.std(hr_series)),
        minimum,
        maximum,
        maximum - minimum,
        trend,
    ]


def make_patient_windows(record_name):
    """Extract windows, beat counts, and annotation-independent features."""
    record_path = DATA_DIR / record_name
    record = wfdb.rdrecord(str(record_path), channels=[0], physical=True)
    annotation = wfdb.rdann(str(record_path), "atr")
    sampling_rate = float(record.fs)
    window_samples = int(round(WINDOW_SECONDS * sampling_rate))
    raw_signal = np.asarray(record.p_signal[:, 0], dtype=np.float64)
    filtered_signal = filter_ecg(raw_signal, sampling_rate)
    one_hz_heart_rate = derive_one_hz_heart_rate(
        filtered_signal,
        sampling_rate,
    )

    # Count only beats assigned to the standard AAMI groups; rhythm markers,
    # unknown annotations, and other non-beat annotations do not become labels.
    beat_samples = []
    beat_is_abnormal = []
    for sample, symbol in zip(annotation.sample, annotation.symbol):
        if symbol in NORMAL_BEAT_SYMBOLS:
            beat_samples.append(int(sample))
            beat_is_abnormal.append(0)
        elif symbol in ABNORMAL_BEAT_SYMBOLS:
            beat_samples.append(int(sample))
            beat_is_abnormal.append(1)

    beat_samples = np.asarray(beat_samples, dtype=np.int64)
    beat_is_abnormal = np.asarray(beat_is_abnormal, dtype=np.int8)
    if len(beat_samples) == 0:
        raise ValueError("The record has no recognized AAMI beat annotations.")

    rows = []
    for start in range(0, len(filtered_signal) - window_samples + 1, window_samples):
        end = start + window_samples
        first_beat = np.searchsorted(beat_samples, start, side="left")
        after_last_beat = np.searchsorted(beat_samples, end, side="left")
        if first_beat == after_last_beat:
            continue

        window_abnormal = beat_is_abnormal[first_beat:after_last_beat]
        feature_values = extract_window_features(
            raw_signal[start:end],
            filtered_signal[start:end],
            sampling_rate,
        )
        second_start = int(round(start / sampling_rate))
        second_end = second_start + WINDOW_SECONDS
        wearable_features = summarize_one_hz_heart_rate(
            one_hz_heart_rate[second_start:second_end]
        )
        rows.append(
            {
                "record": record_name,
                "annotated_beats": int(len(window_abnormal)),
                "abnormal_beats": int(np.sum(window_abnormal)),
                "abnormal_beat_ratio": float(np.mean(window_abnormal)),
                **dict(zip(OFFLINE_FEATURE_NAMES, feature_values)),
                **dict(zip(WEARABLE_FEATURE_NAMES, wearable_features)),
            }
        )

    if not rows:
        raise ValueError("The record did not produce any labeled windows.")
    return rows, len(annotation.sample), len(beat_samples), int(np.sum(beat_is_abnormal))


def build_dataset(records):
    """Process eligible records and count real annotations and extracted windows."""
    window_rows = []
    used_records = []
    record_annotation_counts = {}
    record_aami_beat_counts = {}
    record_abnormal_beat_counts = {}
    excluded_records = []

    for record_name in records:
        if record_name in PACED_RECORDS:
            excluded_records.append(record_name)
            continue
        try:
            rows, annotation_count, beat_count, abnormal_count = make_patient_windows(
                record_name
            )
        except Exception as error:
            raise RuntimeError(
                f"Could not process MIT-BIH record {record_name}: {error}"
            ) from error

        used_records.append(record_name)
        window_rows.extend(rows)
        record_annotation_counts[record_name] = annotation_count
        record_aami_beat_counts[record_name] = beat_count
        record_abnormal_beat_counts[record_name] = abnormal_count
        print(
            f"Processed record {record_name}: {beat_count:,} AAMI beats, "
            f"{len(rows):,} windows"
        )

    if not window_rows:
        raise RuntimeError("No labeled windows could be extracted.")

    windows = pd.DataFrame(window_rows)
    return (
        windows,
        used_records,
        excluded_records,
        record_annotation_counts,
        record_aami_beat_counts,
        record_abnormal_beat_counts,
    )


def labels_for_rule(windows, minimum_abnormal_beats):
    """Label a window abnormal only when it meets the chosen beat-count rule."""
    return (windows["abnormal_beats"].to_numpy() >= minimum_abnormal_beats).astype(
        np.int8
    )


def write_dataset_statistics(
    records,
    used_records,
    excluded_records,
    annotation_counts,
    aami_beat_counts,
    abnormal_beat_counts,
    windows,
):
    """Save transparent record, annotation, window, and class-balance counts."""
    total_aami_beats = sum(aami_beat_counts.values())
    total_abnormal_beats = sum(abnormal_beat_counts.values())
    statistics = [
        {
            "label_rule": "all",
            "statistic": "MIT-BIH records available",
            "value": len(records),
        },
        {
            "label_rule": "all",
            "statistic": "paced records excluded",
            "value": len(excluded_records),
        },
        {
            "label_rule": "all",
            "statistic": "records included",
            "value": len(used_records),
        },
        {
            "label_rule": "all",
            "statistic": "excluded paced record IDs",
            "value": ",".join(excluded_records),
        },
        {
            "label_rule": "all",
            "statistic": "raw annotations in included records",
            "value": sum(annotation_counts.values()),
        },
        {
            "label_rule": "all",
            "statistic": "recognized AAMI annotated beats",
            "value": total_aami_beats,
        },
        {
            "label_rule": "all",
            "statistic": "normal AAMI beats",
            "value": total_aami_beats - total_abnormal_beats,
        },
        {
            "label_rule": "all",
            "statistic": "abnormal AAMI beats",
            "value": total_abnormal_beats,
        },
        {
            "label_rule": "all",
            "statistic": "windows with recognized beats",
            "value": len(windows),
        },
    ]

    for rule_name, threshold in LABEL_RULES.items():
        labels = labels_for_rule(windows, threshold)
        normal_count = int(np.sum(labels == 0))
        abnormal_count = int(np.sum(labels == 1))
        statistics.extend(
            [
                {
                    "label_rule": rule_name,
                    "statistic": "minimum abnormal beats per window",
                    "value": threshold,
                },
                {
                    "label_rule": rule_name,
                    "statistic": "windows",
                    "value": len(labels),
                },
                {
                    "label_rule": rule_name,
                    "statistic": "normal windows",
                    "value": normal_count,
                },
                {
                    "label_rule": rule_name,
                    "statistic": "abnormal windows",
                    "value": abnormal_count,
                },
                {
                    "label_rule": rule_name,
                    "statistic": "abnormal window percent",
                    "value": 100 * abnormal_count / len(labels),
                },
            ]
        )

    stats = pd.DataFrame(statistics)
    stats.to_csv(RESULTS_DIR / "dataset_stats.csv", index=False)
    print("\nDataset statistics:")
    print(
        f"Records: {len(used_records)} used, {len(excluded_records)} paced "
        f"records excluded ({', '.join(excluded_records)})"
    )
    print(
        f"Annotated beats: {total_aami_beats:,} AAMI-group beats "
        f"({sum(annotation_counts.values()):,} total annotations)"
    )
    print(f"Windows with annotated beats: {len(windows):,}")
    for rule_name, threshold in LABEL_RULES.items():
        labels = labels_for_rule(windows, threshold)
        abnormal_count = int(np.sum(labels))
        print(
            f"{rule_name}: {len(labels) - abnormal_count:,} normal, "
            f"{abnormal_count:,} abnormal "
            f"({100 * abnormal_count / len(labels):.2f}% abnormal)"
        )
    return stats


def make_estimator(model_name, y_fit, config=None):
    """Build one estimator with imbalance weights from its fit partition only."""
    if model_name == "Logistic Regression":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=RANDOM_STATE,
            ),
        )
    if model_name == "Random Forest":
        parameters = config or RANDOM_FOREST_CONFIGS[0]
        return RandomForestClassifier(
            **parameters,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=2,
        )
    if model_name == "XGBoost":
        if len(np.unique(y_fit)) < 2:
            raise RuntimeError("A training fold contains only one class.")
        counts = np.bincount(y_fit, minlength=2)
        parameters = config or XGBOOST_CONFIGS[0]
        return xgb.XGBClassifier(
            **parameters,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=counts[0] / counts[1],
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=2,
        )
    raise ValueError(f"Unknown model: {model_name}")


def choose_tree_config(model_name, x_train, y_train, groups_train):
    """Tune tree settings using grouped inner CV inside an outer train fold."""
    if model_name == "Random Forest":
        candidates = RANDOM_FOREST_CONFIGS
    elif model_name == "XGBoost":
        candidates = XGBOOST_CONFIGS
    else:
        return None

    splitter = GroupKFold(n_splits=INNER_FOLDS)
    best_config = None
    best_score = -1.0
    for config in candidates:
        fold_scores = []
        for fit_indices, validation_indices in splitter.split(
            x_train,
            y_train,
            groups_train,
        ):
            y_fit = y_train[fit_indices]
            if len(np.unique(y_fit)) < 2:
                raise RuntimeError(
                    "An inner patient-group training fold has only one class."
                )
            estimator = make_estimator(model_name, y_fit, config)
            estimator.fit(x_train.iloc[fit_indices], y_fit)
            predictions = estimator.predict(x_train.iloc[validation_indices])
            fold_scores.append(
                f1_score(
                    y_train[validation_indices],
                    predictions,
                    zero_division=0,
                )
            )
        mean_score = float(np.mean(fold_scores))
        if mean_score > best_score:
            best_score = mean_score
            best_config = config
    return best_config


def extract_importances(model, model_name):
    """Get model-specific feature importance values for the summary chart."""
    if model_name == "Logistic Regression":
        return np.abs(model.named_steps["logisticregression"].coef_[0])
    return np.asarray(model.feature_importances_, dtype=float)


def evaluate_label_rule(windows, rule_name, threshold, feature_names, modality):
    """Run tuned, patient-grouped outer CV and retain honest OOF predictions."""
    features = windows[feature_names].replace([np.inf, -np.inf], np.nan)
    if features.isna().any().any():
        raise RuntimeError("Feature extraction produced missing or infinite values.")
    labels = labels_for_rule(windows, threshold)
    groups = windows["record"].to_numpy()
    models = ["Logistic Regression", "Random Forest", "XGBoost"]
    outer_splitter = GroupKFold(n_splits=OUTER_FOLDS)
    fold_results = {name: [] for name in models}
    out_of_fold = {name: np.full(len(windows), -1, dtype=np.int8) for name in models}
    fold_importances = {name: [] for name in models}
    tuned_configs = {name: [] for name in models}

    for fold_number, (train_indices, test_indices) in enumerate(
        outer_splitter.split(features, labels, groups),
        start=1,
    ):
        train_groups = groups[train_indices]
        test_groups = groups[test_indices]
        if set(train_groups).intersection(test_groups):
            raise RuntimeError("Patient leakage detected between CV folds.")
        x_train = features.iloc[train_indices]
        x_test = features.iloc[test_indices]
        y_train = labels[train_indices]
        y_test = labels[test_indices]
        if len(np.unique(y_train)) < 2:
            raise RuntimeError(f"Outer training fold {fold_number} has one class.")

        print(
            f"\n{modality}, {rule_name}, fold {fold_number}/{OUTER_FOLDS}: "
            f"{len(set(train_groups))} train records, "
            f"{len(set(test_groups))} held-out records"
        )
        for model_name in models:
            config = choose_tree_config(
                model_name,
                x_train,
                y_train,
                train_groups,
            )
            estimator = make_estimator(model_name, y_train, config)
            estimator.fit(x_train, y_train)
            predictions = estimator.predict(x_test).astype(np.int8)
            out_of_fold[model_name][test_indices] = predictions
            fold_results[model_name].append(
                {
                    "accuracy": accuracy_score(y_test, predictions),
                    "precision": precision_score(
                        y_test,
                        predictions,
                        zero_division=0,
                    ),
                    "recall": recall_score(y_test, predictions, zero_division=0),
                    "f1": f1_score(y_test, predictions, zero_division=0),
                }
            )
            fold_importances[model_name].append(
                extract_importances(estimator, model_name)
            )
            tuned_configs[model_name].append(config)
            print(
                f"  {model_name}: F1="
                f"{fold_results[model_name][-1]['f1']:.4f}"
            )

    if any(np.any(predictions < 0) for predictions in out_of_fold.values()):
        raise RuntimeError("Some windows did not receive an out-of-fold prediction.")

    summary_rows = []
    for model_name in models:
        fold_frame = pd.DataFrame(fold_results[model_name])
        row = {"modality": modality, "label_rule": rule_name, "model": model_name}
        for metric in ("accuracy", "precision", "recall", "f1"):
            row[f"{metric}_mean"] = float(fold_frame[metric].mean())
            row[f"{metric}_std"] = float(fold_frame[metric].std(ddof=1))
        summary_rows.append(row)

    return {
        "summary_rows": summary_rows,
        "out_of_fold": out_of_fold,
        "fold_importances": fold_importances,
        "tuned_configs": tuned_configs,
        "labels": labels,
        "features": features,
        "feature_names": feature_names,
        "groups": groups,
    }


def most_common_config(configurations):
    """Pick the most frequently successful inner-CV tree configuration."""
    if not configurations or configurations[0] is None:
        return None
    keys = [tuple(sorted(config.items())) for config in configurations]
    most_common_key = Counter(keys).most_common(1)[0][0]
    return dict(most_common_key)


def save_model_artifacts(
    comparison,
    evaluation,
    model_path,
    feature_chart_path=None,
    confusion_matrix_path=None,
):
    """Select strictly by mean F1 and refit the winner on all eligible records."""
    if comparison.empty:
        raise RuntimeError("No cross-validation results are available.")
    # Mean F1 alone determines the winner; no accuracy-based tie-break is used.
    winner = comparison.loc[comparison["f1_mean"].idxmax()]
    best_name = winner["model"]
    y_true = evaluation["labels"]
    y_pred = evaluation["out_of_fold"][best_name]

    if confusion_matrix_path is not None:
        matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
        display = ConfusionMatrixDisplay(
            confusion_matrix=matrix,
            display_labels=["Normal", "Abnormal (>=1 beat)"],
        )
        display.plot(cmap="Blues", values_format="d")
        plt.title(f"5-Fold Patient-Level OOF Confusion Matrix - {best_name}")
        plt.tight_layout()
        plt.savefig(confusion_matrix_path, dpi=150)
        plt.close()

    if feature_chart_path is not None:
        importance_values = np.mean(
            evaluation["fold_importances"][best_name],
            axis=0,
        )
        importance = pd.Series(
            importance_values,
            index=evaluation["feature_names"],
        ).sort_values()
        plt.figure(figsize=(9, 7))
        importance.plot(kind="barh", color="teal")
        plt.xlabel("Mean feature importance across outer folds")
        plt.title(f"5-Fold Feature Importance - {best_name}")
        plt.tight_layout()
        plt.savefig(feature_chart_path, dpi=150)
        plt.close()

    # Refit on all eligible records using the modal inner-CV tree settings.
    model_features = evaluation["features"]
    labels = evaluation["labels"]
    config = most_common_config(evaluation["tuned_configs"][best_name])
    final_model = make_estimator(best_name, labels, config)
    final_model.fit(model_features, labels)
    joblib.dump(final_model, model_path)
    return winner, best_name


def choose_calibrated_risk_cutoffs(labels, probabilities):
    """Choose low and high alert boundaries from grouped validation probabilities."""
    labels = np.asarray(labels, dtype=np.int8)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    candidates = np.unique(np.clip(probabilities, 0.0, 1.0))
    candidates = np.unique(np.concatenate(([0.0], candidates, [1.0])))

    # LOW ends at the smallest boundary that returns at least 90% of normal
    # validation windows to the LOW band.
    normal_labels = (labels == 0).astype(np.int8)
    low_candidates = [
        cutoff
        for cutoff in candidates
        if cutoff > 0.0
        and recall_score(
            normal_labels,
            probabilities < cutoff,
            zero_division=0,
        )
        >= LOW_NORMAL_RECALL_TARGET
    ]
    low_max = min(low_candidates) if low_candidates else 0.0

    # HIGH starts at the first boundary above LOW with at least 80% validation
    # precision for abnormal windows. If none qualifies, reserve HIGH for p=1.
    high_candidates = [
        cutoff
        for cutoff in candidates
        if cutoff > low_max
        and precision_score(
            labels,
            probabilities >= cutoff,
            zero_division=0,
        )
        >= HIGH_PRECISION_TARGET
    ]
    high_min = min(high_candidates) if high_candidates else 1.0
    if high_min <= low_max:
        low_max = min(low_max, np.nextafter(high_min, 0.0))
    return {
        "low_max": float(low_max),
        "high_min": float(high_min),
        "high_enabled": bool(high_candidates),
        "low_normal_recall_target": LOW_NORMAL_RECALL_TARGET,
        "high_precision_target": HIGH_PRECISION_TARGET,
        "calibration": "sigmoid (CalibratedClassifierCV)",
    }


def report_risk_band_metrics(labels, probabilities, thresholds):
    """Print class precision and validation coverage for each risk band."""
    labels = np.asarray(labels, dtype=np.int8)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    levels = np.full(len(probabilities), "MEDIUM", dtype=object)
    levels[probabilities < thresholds["low_max"]] = "LOW"
    levels[probabilities >= thresholds["high_min"]] = "HIGH"

    summaries = []
    normal_labels = (labels == 0).astype(np.int8)
    for level in ("LOW", "MEDIUM", "HIGH"):
        selected = levels == level
        if level == "LOW":
            target = normal_labels
            description = "normal"
        else:
            target = labels
            description = "abnormal"
        precision = (
            float(np.mean(target[selected]))
            if np.any(selected)
            else float("nan")
        )
        recall = (
            float(np.sum(target[selected]) / np.sum(target))
            if np.sum(target) > 0
            else float("nan")
        )
        summaries.append(
            {
                "level": level,
                "precision": precision,
                "recall": recall,
                "count": int(np.sum(selected)),
                "target_class": description,
            }
        )
    if thresholds["high_enabled"]:
        high_text = f"HIGH for p >= {thresholds['high_min']:.4f}"
    else:
        high_text = (
            "HIGH unavailable from model probability (no validation cutoff "
            f"met {HIGH_PRECISION_TARGET:.0%} precision)"
        )
    print(
        f"\nCalibrated risk cutoffs: LOW for p < {thresholds['low_max']:.4f} "
        f"(normal-class recall target {LOW_NORMAL_RECALL_TARGET:.0%}); "
        f"MEDIUM otherwise below the HIGH cutoff; {high_text}"
    )
    print(
        "Validation risk-band metrics (LOW precision/recall refer to normal; "
        "MEDIUM/HIGH refer to abnormal):"
    )
    print("level   target     precision   recall    windows")
    for summary in summaries:
        precision = (
            f"{summary['precision']:.4f}"
            if np.isfinite(summary["precision"])
            else "N/A"
        )
        recall = (
            f"{summary['recall']:.4f}"
            if np.isfinite(summary["recall"])
            else "N/A"
        )
        print(
            f"{summary['level']:<7} {summary['target_class']:<10} "
            f"{precision:<11} {recall:<9} {summary['count']}"
        )
    return summaries


def calibrate_and_save_wearable_model(comparison, evaluation):
    """Calibrate the F1 winner and derive risk cutoffs from grouped OOF data."""
    winner = comparison.loc[comparison["f1_mean"].idxmax()]
    model_name = winner["model"]
    features = evaluation["features"]
    labels = evaluation["labels"]
    groups = evaluation["groups"]
    config = most_common_config(evaluation["tuned_configs"][model_name])
    validation_probabilities = np.full(len(labels), np.nan, dtype=np.float64)

    # Each outer held-out record group is scored by a sigmoid-calibrated model
    # fit only on other records; its calibrator also uses grouped inner folds.
    outer_cv = GroupKFold(n_splits=OUTER_FOLDS)
    for fold_number, (train_indices, test_indices) in enumerate(
        outer_cv.split(features, labels, groups),
        start=1,
    ):
        train_groups = groups[train_indices]
        test_groups = groups[test_indices]
        if set(train_groups).intersection(test_groups):
            raise RuntimeError("Calibration validation split leaks patient records.")
        base_estimator = make_estimator(model_name, labels[train_indices], config)
        inner_splits = list(
            GroupKFold(n_splits=INNER_FOLDS).split(
                features.iloc[train_indices],
                labels[train_indices],
                groups=train_groups,
            )
        )
        calibrated = CalibratedClassifierCV(
            estimator=base_estimator,
            method="sigmoid",
            cv=inner_splits,
            ensemble=True,
        )
        calibrated.fit(features.iloc[train_indices], labels[train_indices])
        positive_index = list(calibrated.classes_).index(1)
        validation_probabilities[test_indices] = calibrated.predict_proba(
            features.iloc[test_indices]
        )[:, positive_index]
        print(f"Calibrated held-out patient fold {fold_number}/{OUTER_FOLDS}.")

    if not np.isfinite(validation_probabilities).all():
        raise RuntimeError("Some windows did not receive calibrated OOF probabilities.")
    thresholds = choose_calibrated_risk_cutoffs(labels, validation_probabilities)
    thresholds["validation_precision_recall"] = report_risk_band_metrics(
        labels,
        validation_probabilities,
        thresholds,
    )

    # Refit grouped calibration on the complete cohort for production inference.
    final_estimator = make_estimator(model_name, labels, config)
    final_splits = list(
        GroupKFold(n_splits=CALIBRATION_FOLDS).split(
            features,
            labels,
            groups=groups,
        )
    )
    final_calibrated_model = CalibratedClassifierCV(
        estimator=final_estimator,
        method="sigmoid",
        cv=final_splits,
        ensemble=True,
    )
    final_calibrated_model.fit(features, labels)
    artifact = {
        "model": final_calibrated_model,
        "risk_thresholds": thresholds,
        "feature_names": WEARABLE_FEATURE_NAMES,
        "model_name": model_name,
        "calibration": "sigmoid (CalibratedClassifierCV)",
    }
    joblib.dump(artifact, WEARABLE_MODEL_PATH)
    return model_name, thresholds, validation_probabilities


def main():
    records = download_database()
    (
        windows,
        used_records,
        excluded_records,
        annotation_counts,
        aami_beat_counts,
        abnormal_beat_counts,
    ) = build_dataset(records)
    write_dataset_statistics(
        records,
        used_records,
        excluded_records,
        annotation_counts,
        aami_beat_counts,
        abnormal_beat_counts,
        windows,
    )

    rule_name = PRIMARY_LABEL_RULE
    threshold = LABEL_RULES[rule_name]
    print(
        f"\nBoth model types use the same target: a window is abnormal if it "
        f"contains at least {threshold} AAMI abnormal beat."
    )

    offline_evaluation = evaluate_label_rule(
        windows,
        rule_name,
        threshold,
        OFFLINE_FEATURE_NAMES,
        "Offline ECG model",
    )
    offline_comparison = pd.DataFrame(offline_evaluation["summary_rows"])
    offline_comparison = offline_comparison.sort_values(
        "f1_mean",
        ascending=False,
    )
    offline_comparison.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    offline_winner, offline_name = save_model_artifacts(
        offline_comparison,
        offline_evaluation,
        MODEL_PATH,
        feature_chart_path=RESULTS_DIR / "best_model_feature_importance.png",
        confusion_matrix_path=RESULTS_DIR / "best_model_confusion_matrix.png",
    )

    # The wearable estimator deliberately sees only six summaries of a
    # once-per-second heart-rate series, matching the live sensor's resolution.
    wearable_evaluation = evaluate_label_rule(
        windows,
        rule_name,
        threshold,
        WEARABLE_FEATURE_NAMES,
        "1 Hz wearable model",
    )
    wearable_comparison = pd.DataFrame(wearable_evaluation["summary_rows"])
    wearable_comparison["purpose"] = (
        "Predict annotated abnormal-beat windows from 1 Hz heart-rate data "
        "when no raw ECG waveform is available."
    )
    wearable_comparison = wearable_comparison.sort_values(
        "f1_mean",
        ascending=False,
    )
    wearable_comparison.to_csv(
        RESULTS_DIR / "wearable_model_comparison.csv",
        index=False,
    )
    wearable_name, risk_thresholds, calibrated_oof = calibrate_and_save_wearable_model(
        wearable_comparison,
        wearable_evaluation,
    )
    wearable_comparison["calibrated_low_max"] = risk_thresholds["low_max"]
    wearable_comparison["calibrated_high_min"] = risk_thresholds["high_min"]
    wearable_comparison.to_csv(
        RESULTS_DIR / "wearable_model_comparison.csv",
        index=False,
    )
    wearable_winner = wearable_comparison.loc[
        wearable_comparison["model"] == wearable_name
    ].iloc[0]

    display_columns = [
        "model",
        "accuracy_mean",
        "accuracy_std",
        "precision_mean",
        "precision_std",
        "recall_mean",
        "recall_std",
        "f1_mean",
        "f1_std",
    ]
    print("\nOffline ECG model — grouped patient-level 5-fold CV:")
    print(
        offline_comparison[display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )
    print("\nWearable model — grouped patient-level 5-fold CV:")
    print(
        wearable_comparison[display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )
    print(
        f"\nOffline winner by mean F1: {offline_name} "
        f"(mean F1={offline_winner['f1_mean']:.4f}, "
        f"std={offline_winner['f1_std']:.4f})"
    )
    print(
        f"Wearable winner by mean F1: {wearable_name} "
        f"(mean F1={wearable_winner['f1_mean']:.4f}, "
        f"std={wearable_winner['f1_std']:.4f})"
    )
    print(
        f"Saved validation-derived probability cutoffs: "
        f"LOW < {risk_thresholds['low_max']:.4f}; "
        f"MEDIUM < {risk_thresholds['high_min']:.4f}; HIGH >= "
        f"{risk_thresholds['high_min']:.4f}"
    )
    print("\nModel purpose: model.pkl is the offline ECG model for features "
          "derived from raw MIT-BIH waveforms. wearable_model.pkl is a separate "
          "model for six features derived from a once-per-second heart-rate "
          "series, for deployments without raw ECG. Both predict the same "
          "annotation-based abnormal-beat target; neither is a clinical "
          "heart-disease diagnosis.")
    print(f"\nSaved models and results: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
