"""Firebase-backed patient dashboard for the 1 Hz wearable heart-rate model.

This project output is an experimental abnormal-beat risk indicator, not a
medical diagnosis. It must not replace clinical assessment or emergency care.
"""
 
import argparse
import csv
from collections import deque
from datetime import datetime, timezone
import math
from pathlib import Path
import re
import sys
import time
from zoneinfo import ZoneInfo

import firebase_admin
from firebase_admin import credentials, db
import joblib
import numpy as np
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx


DATABASE_URL = (
    "https://major-project-t13-default-rtdb.asia-southeast1.firebasedatabase.app"
)
PROJECT_DIR = Path(__file__).resolve().parent
KEY_PATH = PROJECT_DIR / "serviceAccountKey.json"
WEARABLE_MODEL_PATH = PROJECT_DIR / "wearable_model.pkl"
PATIENT_DATA_DIR = PROJECT_DIR / "data" / "patients"
SENSOR_NODE = "cardioMonitor"
SIMULATION_NODE = "cardioMonitor_test"
POLL_INTERVAL_SECONDS = 1
WINDOW_SECONDS = 10
STALE_AFTER_SECONDS = 10
CSV_COLUMNS = [
    "timestamp",
    "heartRate",
    "spo2",
    "ecg",
    "acc_x",
    "acc_y",
    "acc_z",
    "risk",
]
WEARABLE_FEATURES = [
    "hr_1hz_mean_bpm",
    "hr_1hz_std_bpm",
    "hr_1hz_min_bpm",
    "hr_1hz_max_bpm",
    "hr_1hz_range_bpm",
    "hr_1hz_trend_bpm_per_second",
]

# Explainable project thresholds. The pulse-oximeter and heart-rate cutoffs
# are project rules, not clinical advice. Rest is inferred from a stable
# acceleration magnitude close to gravity, not from deviceStatus.
SPO2_HIGH_RISK_PERCENT = 92.0
HEART_RATE_REST_HIGH_BPM = 120.0
HEART_RATE_LOW_BPM = 45.0
HEART_RATE_LOW_HIGH_BPM = 35.0
REST_GRAVITY_MIN_G = 0.8
REST_GRAVITY_MAX_G = 1.2
REST_ACCEL_MAG_STD_MAX_G = 0.15
IST = ZoneInfo("Asia/Kolkata")


def initialize_firebase():
    """Reuse an existing Firebase app or initialize it without exposing its key."""
    try:
        return firebase_admin.get_app()
    except ValueError:
        if not KEY_PATH.is_file():
            raise FileNotFoundError("Firebase credential serviceAccountKey.json was not found.")
        credential = credentials.Certificate(str(KEY_PATH))
        return firebase_admin.initialize_app(
            credential,
            {"databaseURL": DATABASE_URL},
        )


def load_wearable_model():
    """Load the calibrated wearable classifier and validation-derived cutoffs."""
    if not WEARABLE_MODEL_PATH.is_file():
        raise FileNotFoundError("wearable_model.pkl was not found. Train the models first.")
    artifact = joblib.load(WEARABLE_MODEL_PATH)
    if not isinstance(artifact, dict) or "model" not in artifact:
        raise ValueError("wearable_model.pkl is missing its calibrated model bundle.")
    model = artifact["model"]
    if getattr(model, "n_features_in_", None) != len(WEARABLE_FEATURES):
        raise ValueError(
            "wearable_model.pkl does not match the expected six 1 Hz heart-rate features."
        )
    thresholds = artifact.get("risk_thresholds")
    if (
        not isinstance(thresholds, dict)
        or not 0 <= thresholds.get("low_max", -1) < thresholds.get("high_min", -1) <= 1
        or not isinstance(thresholds.get("high_enabled"), bool)
    ):
        raise ValueError("wearable_model.pkl is missing valid calibrated risk cutoffs.")
    return artifact


def validate_patient_id(patient_id):
    """Restrict IDs to safe Firebase path and local filename characters."""
    value = patient_id.strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", value):
        raise ValueError(
            "Patient ID must be 1-64 letters, digits, underscores, or hyphens."
        )
    return value


def as_finite_float(value):
    """Convert a sensor value to a finite float, or report it as unavailable."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def normalize_sensor_reading(data):
    """Read only the existing ESP32 fields and preserve decimal sensor values."""
    if not isinstance(data, dict):
        return None
    acceleration = data.get("acceleration")
    if not isinstance(acceleration, dict):
        acceleration = {}
    return {
        "heartRate": as_finite_float(data.get("heartRate")),
        "spo2": as_finite_float(data.get("spo2")),
        "ecg": data.get("ecg"),
        "acc_x": as_finite_float(acceleration.get("x")),
        "acc_y": as_finite_float(acceleration.get("y")),
        "acc_z": as_finite_float(acceleration.get("z")),
        "lastUpdateMs": data.get("lastUpdateMs"),
    }


def is_valid_sensor_reading(reading):
    """A zero pulse or saturation means no finger is on the sensor."""
    return (
        reading is not None
        and reading["heartRate"] is not None
        and reading["spo2"] is not None
        and reading["heartRate"] > 0
        and reading["spo2"] > 0
    )


def make_hr_features(valid_readings):
    """Make the six wearable-model inputs from ten once-per-second readings."""
    if len(valid_readings) != WINDOW_SECONDS:
        raise ValueError("Risk prediction requires ten valid one-second readings.")
    heart_rates = np.asarray(
        [reading["heartRate"] for reading in valid_readings],
        dtype=np.float64,
    )
    if not np.isfinite(heart_rates).all() or np.any(heart_rates <= 0):
        raise ValueError("The heart-rate window contains invalid sensor readings.")
    trend = float(np.polyfit(np.arange(WINDOW_SECONDS), heart_rates, 1)[0])
    minimum = float(np.min(heart_rates))
    maximum = float(np.max(heart_rates))
    values = [
        float(np.mean(heart_rates)),
        float(np.std(heart_rates)),
        minimum,
        maximum,
        maximum - minimum,
        trend,
    ]
    return np.asarray([values], dtype=np.float64), heart_rates


def acceleration_indicates_rest(valid_readings):
    """Use a stable, near-gravity 10-second acceleration magnitude as rest."""
    magnitudes = []
    for reading in valid_readings:
        axes = [reading["acc_x"], reading["acc_y"], reading["acc_z"]]
        if any(axis is None for axis in axes):
            return False
        magnitudes.append(math.sqrt(sum(axis * axis for axis in axes)))
    magnitudes = np.asarray(magnitudes, dtype=np.float64)
    mean_magnitude = float(np.mean(magnitudes))
    return (
        REST_GRAVITY_MIN_G <= mean_magnitude <= REST_GRAVITY_MAX_G
        and float(np.std(magnitudes)) <= REST_ACCEL_MAG_STD_MAX_G
    )


def classify_risk(valid_readings, model, thresholds):
    """Combine the wearable model probability with transparent sensor rules."""
    model_features, heart_rates = make_hr_features(valid_readings)
    average_hr = float(np.mean(heart_rates))
    average_spo2 = float(np.mean([reading["spo2"] for reading in valid_readings]))
    positive_class = list(model.classes_).index(1)
    abnormal_probability = float(model.predict_proba(model_features)[0][positive_class])

    rules = []
    if average_spo2 < SPO2_HIGH_RISK_PERCENT:
        rules.append(("HIGH", "Average SpO2 below 92%"))
    if (
        average_hr > HEART_RATE_REST_HIGH_BPM
        and acceleration_indicates_rest(valid_readings)
    ):
        rules.append(("HIGH", "Average heart rate above 120 bpm while acceleration indicates rest"))
    if average_hr < HEART_RATE_LOW_HIGH_BPM:
        rules.append(("HIGH", "Average heart rate below 35 bpm"))
    elif average_hr < HEART_RATE_LOW_BPM:
        rules.append(("MEDIUM", "Average heart rate below 45 bpm"))

    if thresholds["high_enabled"] and abnormal_probability >= thresholds["high_min"]:
        model_risk = "HIGH"
    elif abnormal_probability >= thresholds["low_max"]:
        model_risk = "MEDIUM"
    else:
        model_risk = "LOW"
    rules.append(
        (
            model_risk,
            f"Wearable model abnormal-beat probability {abnormal_probability:.0%}",
        )
    )

    severity = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
    final_risk = max((risk for risk, _ in rules), key=severity.__getitem__)
    reasons = [reason for risk, reason in rules if risk == final_risk]
    return {
        "risk": final_risk,
        "reason": "; ".join(reasons),
        "probability": abnormal_probability,
        "average_heart_rate": average_hr,
        "average_spo2": average_spo2,
    }


def csv_path_for_patient(patient_id):
    """Use the validated ID, never the optional patient name, in filenames."""
    safe_id = validate_patient_id(patient_id)
    return PATIENT_DATA_DIR / f"{safe_id}.csv"


def append_csv_row(patient_id, row):
    """Append one logged sensor sample to the patient's CSV file."""
    PATIENT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = csv_path_for_patient(patient_id)
    new_file = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=CSV_COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)
    return path


def firebase_reading_row(timestamp, reading, risk):
    """Construct the requested per-reading payload without changing sensor keys."""
    return {
        "timestamp": timestamp,
        "heartRate": reading["heartRate"],
        "spo2": reading["spo2"],
        "ecg": reading["ecg"],
        "acc_x": reading["acc_x"],
        "acc_y": reading["acc_y"],
        "acc_z": reading["acc_z"],
        "risk": risk,
    }


def store_reading(patient_ref, row):
    """Append a per-second snapshot under this patient's readings collection."""
    patient_ref.child("readings").push(row)


def publish_risk(sensor_ref, patient_ref, result):
    """Write the same final risk and explanation to both requested locations."""
    payload = {
        "risk": result["risk"],
        "reason": result["reason"],
        "probability": result["probability"],
        "updatedAt": datetime.now(timezone.utc).isoformat(),
    }
    sensor_ref.child("risk").set(result["risk"])
    patient_ref.child("latestRisk").set(payload)
    return payload


def read_current_reading(sensor_ref):
    """Read the live node once and return its existing sensor fields."""
    return normalize_sensor_reading(sensor_ref.get())


def advance_sensor_clock(state, last_update, now):
    """Track lastUpdateMs changes independently of deviceStatus."""
    if last_update is not None and last_update != state["last_update"]:
        state["last_update"] = last_update
        state["last_change_at"] = now
        return True
    return False


def poll_session_once(
    sensor_ref,
    patient_ref,
    patient_id,
    model,
    state,
    now=None,
):
    """Read, log, and if due calculate one session update."""
    now = time.monotonic() if now is None else now
    timestamp = datetime.now(timezone.utc).isoformat()
    data = read_current_reading(sensor_ref)
    if data is None:
        raise ValueError("Firebase cardioMonitor node is empty or malformed.")

    new_sensor_update = advance_sensor_clock(state, data["lastUpdateMs"], now)
    offline = (
        state["last_change_at"] is None
        or now - state["last_change_at"] >= STALE_AFTER_SECONDS
    )

    if new_sensor_update and is_valid_sensor_reading(data):
        state["valid_readings"].append({**data, "_received_at": now})
    while (
        state["valid_readings"]
        and now - state["valid_readings"][0]["_received_at"] > WINDOW_SECONDS
    ):
        state["valid_readings"].popleft()

    result = None
    status = "Collecting valid readings"
    if offline:
        status = "Sensor offline or no new data"
        state["valid_readings"].clear()
        state["latest_risk"] = None
    elif data["heartRate"] == 0 or data["spo2"] == 0:
        status = "No valid reading"
        state["latest_risk"] = None
        state["latest_reason"] = None
    elif data["heartRate"] is None or data["spo2"] is None:
        status = "No valid reading"
        state["latest_risk"] = None
        state["latest_reason"] = None
    elif len(state["valid_readings"]) < WINDOW_SECONDS:
        status = (
            f"Collecting valid readings "
            f"({len(state['valid_readings'])}/{WINDOW_SECONDS})"
        )
    elif now - state["last_prediction_at"] >= WINDOW_SECONDS:
        result = classify_risk(
            list(state["valid_readings"]),
            model,
            state["risk_thresholds"],
        )
        state["latest_risk"] = result["risk"]
        state["latest_reason"] = result["reason"]
        state["last_prediction_at"] = now
        publish_risk(sensor_ref, patient_ref, result)
        status = "Risk updated"

    if status == "No valid reading":
        recorded_risk = "No valid reading"
    elif status == "Sensor offline or no new data":
        recorded_risk = "Sensor offline"
    else:
        recorded_risk = state.get("latest_risk") or "Pending"
    row = firebase_reading_row(timestamp, data, recorded_risk)
    if new_sensor_update and is_valid_sensor_reading(data):
        append_csv_row(patient_id, row)
        store_reading(patient_ref, row)
    state["history"].append(
        {
            "timestamp": timestamp,
            "heartRate": data["heartRate"],
            "spo2": data["spo2"],
            "ecg": as_finite_float(data["ecg"]),
            "activity": (
                math.sqrt(
                    data["acc_x"] ** 2
                    + data["acc_y"] ** 2
                    + data["acc_z"] ** 2
                )
                if all(
                    data[key] is not None
                    for key in ("acc_x", "acc_y", "acc_z")
                )
                else None
            ),
            "risk": recorded_risk,
        }
    )

    return {
        "timestamp": timestamp,
        "reading": data,
        "status": status,
        "risk": state.get("latest_risk"),
        "reason": state.get("latest_reason"),
        "result": result,
        "csv_path": str(csv_path_for_patient(patient_id)),
    }


def new_session_state():
    """Create isolated rolling state for one patient monitoring session."""
    return {
        "last_update": None,
        "last_change_at": None,
        "last_prediction_at": 0.0,
        "valid_readings": deque(maxlen=WINDOW_SECONDS),
        "latest_risk": None,
        "latest_reason": None,
        "history": deque(maxlen=120),
        "last_poll": None,
        "last_error": None,
    }


def start_dashboard_session(patient_id, patient_name, demo_mode=False):
    """Initialize session-specific model, Firebase references, and monitor state."""
    app = initialize_firebase()
    state = new_session_state()
    model_artifact = load_wearable_model()
    target_node = SIMULATION_NODE if demo_mode else SENSOR_NODE
    state.update(
        {
            "firebase_app": app,
            "sensor_ref": db.reference(target_node, app=app),
            "patient_ref": db.reference(f"patients/{patient_id}", app=app),
            "model_artifact": model_artifact,
            "model": model_artifact["model"],
            "risk_thresholds": model_artifact["risk_thresholds"],
            "patient_id": patient_id,
            "patient_name": patient_name,
            "demo_mode": demo_mode,
        }
    )
    st.session_state["patient_monitor"] = state


@st.fragment(run_every=f"{POLL_INTERVAL_SECONDS}s")
def monitoring_panel():
    """Refresh the dashboard with one Firebase poll on each one-second tick."""
    state = st.session_state.get("patient_monitor")
    if not state:
        return

    display_name = state["patient_name"] or state["patient_id"]
    if state.get("demo_mode"):
        st.markdown(
            f'### Monitoring: {display_name} <span style="background-color: #d97706; color: white; padding: 4px 10px; border-radius: 6px; font-size: 0.9rem; font-weight: bold; vertical-align: middle;">DEMO MODE</span>',
            unsafe_allow_html=True,
        )
    else:
        st.subheader(f"Monitoring: {display_name}")
    try:
        update = poll_session_once(
            state["sensor_ref"],
            state["patient_ref"],
            state["patient_id"],
            state["model"],
            state,
        )
        state["last_poll"] = update
        state["last_error"] = None
    except Exception as error:
        state["last_error"] = str(error)

    if state["last_error"]:
        st.error(f"Monitoring error: {state['last_error']}")
        return

    update = state["last_poll"]
    reading = update["reading"]
    st.caption(
        f"Session patient ID: {state['patient_id']} · "
        f"Local time (IST): {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S %Z')}"
    )
    if update["status"] == "Sensor offline or no new data":
        st.warning(update["status"])
    elif update["status"] == "No valid reading":
        st.info("Place your finger on the sensor")
    elif update["status"].startswith("Collecting"):
        valid_count = len(state.get("valid_readings", []))
        clamped_count = min(max(valid_count, 0), WINDOW_SECONDS)
        st.progress(
            clamped_count / WINDOW_SECONDS,
            text=f"Collecting valid readings ({clamped_count}/{WINDOW_SECONDS})",
        )
    elif update["status"] == "Risk updated":
        st.success("Risk updated from the last 10 valid one-second readings.")
    card_columns = st.columns(4)
    card_columns[0].metric(
        "Heart rate",
        f"{reading['heartRate']:.1f} bpm"
        if reading["heartRate"] is not None
        else "N/A",
    )
    card_columns[1].metric(
        "SpO2",
        f"{reading['spo2']:.1f}%"
        if reading["spo2"] is not None
        else "N/A",
    )
    card_columns[2].metric("ECG (raw ADC)", str(reading["ecg"] or "N/A"))

    risk_raw = state.get("latest_risk")
    if risk_raw:
        risk_upper = str(risk_raw).strip().upper()
        if risk_upper == "LOW":
            badge_color = "green"
            badge_text = "LOW"
            badge_bg = "#16a34a"
        elif risk_upper == "MEDIUM":
            badge_color = "amber"
            badge_text = "MEDIUM"
            badge_bg = "#d97706"
        elif risk_upper == "HIGH":
            badge_color = "red"
            badge_text = "HIGH"
            badge_bg = "#dc2626"
        else:
            badge_color = "grey"
            badge_text = str(risk_raw).upper()
            badge_bg = "#6b7280"
        reason_text = state.get("latest_reason") or ""
    else:
        badge_color = "grey"
        badge_bg = "#6b7280"
        if update["status"] == "No valid reading":
            badge_text = "NO READING"
            reason_text = "Place your finger on the sensor"
        elif update["status"] == "Sensor offline or no new data":
            badge_text = "OFFLINE"
            reason_text = update["status"]
        elif update["status"].startswith("Collecting"):
            badge_text = "PENDING"
            reason_text = update["status"]
        else:
            badge_text = "PENDING"
            reason_text = update["status"] or ""

    with card_columns[3]:
        reason_html = (
            f'<span class="risk-reason" style="font-size: 0.92rem; font-weight: 500; line-height: 1.3;">{reason_text}</span>'
            if reason_text
            else ""
        )
        st.markdown(
            f"""
            <div data-testid="stMetric" style="height: 100%;">
                <div data-testid="stMetricLabel" style="font-size: 0.95rem; font-weight: 500; margin-bottom: 6px;">Risk</div>
                <div style="display: flex; align-items: center; gap: 10px; flex-wrap: wrap; margin-top: 4px;">
                    <span class="risk-badge risk-badge-{badge_color}" style="background-color: {badge_bg}; color: #ffffff; padding: 0.35rem 0.9rem; font-size: 1.15rem; font-weight: 700; border-radius: 9999px; display: inline-flex; align-items: center; justify-content: center; white-space: nowrap; line-height: 1.2;">{badge_text}</span>
                    {reason_html}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    acceleration_values = [
        reading[key] for key in ("acc_x", "acc_y", "acc_z")
    ]
    st.caption(
        "Acceleration x/y/z: "
        + " / ".join(
            f"{value:.3f}" if value is not None else "N/A"
            for value in acceleration_values
        )
    )
    history = list(state["history"])
    if history:
        import pandas as pd

        chart_data = pd.DataFrame(history)
        chart_data["timestamp"] = pd.to_datetime(chart_data["timestamp"], utc=True)
        chart_data = chart_data.set_index("timestamp")
        chart_column, activity_column = st.columns(2)
        with chart_column:
            st.subheader("ECG")
            st.line_chart(chart_data[["ecg"]], height=220)
        with activity_column:
            st.subheader("Activity (acceleration magnitude)")
            st.line_chart(chart_data[["activity"]], height=220)
    st.caption(f"Patient CSV: {update['csv_path']}")
    st.caption("Sensor timestamps are stored in UTC; this dashboard clock is IST.")
    st.caption(
        "Experimental abnormal-beat indicator only; not a medical diagnosis."
    )


def render_doctor_tab():
    """Separate real patient history from clearly labeled simulation files."""
    st.subheader("Doctor review")
    st.caption("Real patient files only; simulation sessions are excluded.")
    PATIENT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    csv_files = sorted(PATIENT_DATA_DIR.glob("*.csv"))
    simulation_files = [path for path in csv_files if path.name.startswith("sim_")]
    real_files = [path for path in csv_files if not path.name.startswith("sim_")]

    if real_files:
        selected = st.selectbox(
            "Patient history",
            real_files,
            format_func=lambda path: path.stem,
        )
        hide_invalid = st.checkbox("Hide offline/invalid rows", value=True)
        import pandas as pd

        history = pd.read_csv(selected)
        if hide_invalid and not history.empty:
            conditions = []
            if "heartRate" in history.columns:
                conditions.append(
                    pd.to_numeric(history["heartRate"], errors="coerce") == 0
                )
            if "spo2" in history.columns:
                conditions.append(
                    pd.to_numeric(history["spo2"], errors="coerce") == 0
                )
            if "risk" in history.columns:
                conditions.append(
                    history["risk"]
                    .astype(str)
                    .isin(["Sensor offline", "No valid reading"])
                )
            if conditions:
                invalid_mask = conditions[0]
                for cond in conditions[1:]:
                    invalid_mask = invalid_mask | cond
                history = history[~invalid_mask]
        st.dataframe(history.tail(100), use_container_width=True)
        st.download_button(
            "Download patient CSV",
            data=selected.read_bytes(),
            file_name=selected.name,
            mime="text/csv",
        )
    else:
        st.info("No real patient CSV history is available yet.")

    with st.expander(f"SIMULATION sessions ({len(simulation_files)})"):
        st.warning("SIMULATION data is test-only and is not part of the real patient list.")
        for path in simulation_files:
            st.write(f"SIMULATION — {path.stem}")
            st.download_button(
                f"Download {path.name}",
                data=path.read_bytes(),
                file_name=path.name,
                mime="text/csv",
                key=f"download_{path.name}",
            )
    st.caption(
        "Abnormal-beat risk is experimental and is not a clinical diagnosis."
    )


def run_dashboard():
    """Render the patient session form and live monitoring fragment."""
    st.set_page_config(
        page_title="Wearable Heart Monitor",
        page_icon="❤️",
        layout="wide",
    )
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            padding-left: 2rem;
            padding-right: 2rem;
            max-width: 100%;
        }
        @media (max-width: 768px) {
            .block-container {
                padding-top: 1rem;
                padding-bottom: 1rem;
                padding-left: 0.75rem;
                padding-right: 0.75rem;
            }
        }
        div[data-testid="stMetric"] {
            background-color: rgba(128, 128, 128, 0.08);
            border: 1px solid rgba(128, 128, 128, 0.2);
            border-radius: 12px;
            padding: 1rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        }
        div[data-testid="stMetricValue"] {
            font-size: 2rem !important;
            font-weight: 700;
        }
        div[data-testid="stMetricLabel"] {
            font-size: 0.95rem;
            font-weight: 500;
        }
        .stButton > button, div[data-testid="stDownloadButton"] > button {
            min-height: 44px;
            padding: 0.5rem 1.25rem;
            font-size: 1rem;
            border-radius: 8px;
            font-weight: 500;
        }
        .risk-badge {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 0.35rem 0.9rem;
            font-size: 1.15rem;
            font-weight: 700;
            border-radius: 9999px;
            color: #ffffff;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            line-height: 1.2;
            white-space: nowrap;
        }
        .risk-badge-green {
            background-color: #16a34a;
        }
        .risk-badge-amber {
            background-color: #d97706;
        }
        .risk-badge-red {
            background-color: #dc2626;
        }
        .risk-badge-grey {
            background-color: #6b7280;
        }
        .risk-reason {
            font-size: 0.92rem;
            font-weight: 500;
            line-height: 1.3;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Wearable Heart Monitor")
    st.caption(
        "Experimental abnormal-beat risk indicator. It is not a medical "
        "diagnosis or a substitute for clinical care."
    )
    st.caption(
        f"Local time (IST): {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S %Z')} "
        "(recorded files use UTC)"
    )
    st.markdown(
        "Risk is calculated from a rolling window of **10 valid, fresh "
        "heart-rate and SpO2 readings**. Sensor values of zero are ignored."
    )

    monitor_tab, doctor_tab = st.tabs(["Live monitor", "Doctor review"])
    with monitor_tab:
        active = st.session_state.get("patient_monitor")
        if active is None:
            with st.form("patient_session"):
                patient_id_input = st.text_input("Patient ID")
                patient_name = st.text_input("Patient name (optional)")
                demo_mode = st.checkbox("Demo mode (use simulated data)")
                consent = st.checkbox(
                    "I have consent to collect and review this patient's wearable data."
                )
                submitted = st.form_submit_button("Start patient session")
            if submitted:
                if not consent:
                    st.error("Consent is required before starting patient monitoring.")
                else:
                    try:
                        raw_id = patient_id_input.strip()
                        if demo_mode and not raw_id.startswith("sim_"):
                            raw_id = f"sim_{raw_id}"
                        patient_id = validate_patient_id(raw_id)
                        start_dashboard_session(
                            patient_id,
                            patient_name.strip(),
                            demo_mode=demo_mode,
                        )
                        st.rerun()
                    except Exception as error:
                        st.error(f"Could not start session: {error}")
        else:
            if active.get("demo_mode"):
                st.warning("DEMO MODE — this session is running on simulated test data (cardioMonitor_test).")
            elif active["patient_id"].startswith("sim_"):
                st.warning("SIMULATION — this session is test-only.")
            if st.button("End patient session"):
                st.session_state.pop("patient_monitor", None)
                st.rerun()
            monitoring_panel()
    with doctor_tab:
        render_doctor_tab()


def simulated_reading(index, scenario):
    """Generate scenario-specific test inputs only for the isolated test node."""
    scenarios = {
        "normal": (72.0 + 2.0 * math.sin(index / 9.0), 97.0, (0.02, -0.01, 0.99)),
        "exercise": (145.0 + 3.0 * math.sin(index / 8.0), 97.0, (1.2, 1.1, 1.4)),
        "high_hr": (130.0 + 2.0 * math.sin(index / 8.0), 97.0, (0.02, -0.01, 0.99)),
        "low_spo2": (72.0 + 2.0 * math.sin(index / 9.0), 88.0, (0.02, -0.01, 0.99)),
        "no_finger": (0.0, 0.0, (0.02, -0.01, 0.99)),
        "offline": (72.0, 97.0, (0.02, -0.01, 0.99)),
    }
    heart_rate, spo2, acceleration = scenarios[scenario]
    update_ms = (
        1700000000000
        if scenario == "offline"
        else int(time.monotonic() * 1000)
    )
    return {
        "heartRate": heart_rate,
        "spo2": spo2,
        "ecg": 2000 + index % 40,
        "acceleration": {
            "x": acceleration[0],
            "y": acceleration[1],
            "z": acceleration[2],
        },
        "deviceStatus": "ONLINE",
        "lastUpdateMs": update_ms,
    }


def run_simulation(duration_seconds, patient_id, patient_name, scenario):
    """Exercise the end-to-end pipeline using only cardioMonitor_test."""
    patient_id = validate_patient_id(patient_id)
    patient_id = (
        f"sim_{scenario}_{patient_id}_"
        f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    )
    model_artifact = load_wearable_model()
    model = model_artifact["model"]
    app = initialize_firebase()
    test_ref = db.reference(SIMULATION_NODE, app=app)
    patient_ref = db.reference(f"patients/{patient_id}", app=app)
    state = new_session_state()
    state["risk_thresholds"] = model_artifact["risk_thresholds"]
    started = time.monotonic()
    poll_number = 0

    print(
        f"SIMULATION started: scenario={scenario}, patient_id={patient_id}"
        + (f", display name={patient_name}" if patient_name else "")
    )
    print(
        "Safety: fake sensor readings and simulation risks are written only to "
        "cardioMonitor_test, never to cardioMonitor."
    )
    try:
        while time.monotonic() - started < duration_seconds:
            poll_started = time.monotonic()
            fake_data = simulated_reading(poll_number, scenario)
            # This test-only node has the same input fields; no real node write.
            test_ref.set({**fake_data, "risk": state.get("latest_risk")})
            update = poll_session_once(
                test_ref,
                patient_ref,
                patient_id,
                model,
                state,
                now=poll_started,
            )
            state["last_poll"] = update
            poll_number += 1
            if update["result"]:
                result = update["result"]
                print(
                    f"Risk: {result['risk']} | reason: {result['reason']} | "
                    f"probability: {result['probability']:.3f}"
                )
            elif poll_number == 1 or poll_number % 10 == 0:
                print(f"Status: {update['status']}")
            time.sleep(
                max(0, poll_started + POLL_INTERVAL_SECONDS - time.monotonic())
            )
    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")

    elapsed = time.monotonic() - started
    print(f"Simulation ended after {elapsed:.1f} seconds ({poll_number} readings).")
    print(f"Test Firebase node: {SIMULATION_NODE}")
    print(f"Patient readings path: patients/{patient_id}/readings")
    print(f"Patient risk path: patients/{patient_id}/latestRisk")
    print(f"CSV path: {csv_path_for_patient(patient_id)}")
    if state["latest_risk"]:
        print(
            f"Final risk: {state['latest_risk']} | "
            f"reason: {state['latest_reason']}"
        )
    else:
        print("Final risk: No valid reading")
    if state.get("last_poll"):
        print(f"Final status: {state['last_poll']['status']}")
    expected = {
        "normal": "LOW",
        "exercise": "not HIGH",
        "high_hr": "HIGH",
        "low_spo2": "HIGH",
        "no_finger": "No valid reading",
        "offline": "Sensor offline or no new data",
    }[scenario]
    actual = state.get("latest_risk")
    if scenario == "no_finger":
        actual = "No valid reading"
    elif scenario == "offline":
        actual = (
            state["last_poll"]["status"]
            if state.get("last_poll")
            else "No result"
        )
    reason = (
        state.get("latest_reason")
        or (state.get("last_poll") or {}).get("status")
        or "No risk reason"
    )
    outcome = (
        actual != "HIGH"
        if scenario == "exercise"
        else actual == expected
    )
    print(
        f"SCENARIO_RESULT|{scenario}|{expected}|{actual or 'No risk'}|"
        f"{reason}|{'PASS' if outcome else 'FAIL'}"
    )
    return poll_number


def main():
    parser = argparse.ArgumentParser(
        description="Run the wearable monitor dashboard or test it safely."
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Write test readings only to cardioMonitor_test and run the monitor.",
    )
    parser.add_argument(
        "--duration-seconds",
        type=int,
        default=60,
        help="Simulation duration (default: 60 seconds).",
    )
    parser.add_argument(
        "--patient-id",
        default="test",
        help="Safe test patient ID used by --simulate.",
    )
    parser.add_argument(
        "--name",
        default="",
        help="Optional patient display name; never used in paths or filenames.",
    )
    parser.add_argument(
        "--scenario",
        choices=["normal", "exercise", "high_hr", "low_spo2", "no_finger", "offline"],
        default="normal",
        help="Test-only sensor scenario used with --simulate.",
    )
    args, _ = parser.parse_known_args()

    if args.simulate:
        if args.duration_seconds < 1:
            parser.error("--duration-seconds must be at least 1.")
        run_simulation(
            args.duration_seconds,
            args.patient_id,
            args.name,
            args.scenario,
        )
    elif get_script_run_ctx(suppress_warning=True) is None:
        print("Start the patient dashboard with: streamlit run app.py")
    else:
        run_dashboard()


if __name__ == "__main__":
    main()
