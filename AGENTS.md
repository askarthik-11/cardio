# Quick start (Windows PowerShell)

Activate the existing virtual environment with `.\.venv\Scripts\Activate.ps1`.
Train/evaluate with `python .\train.py`. Run the dashboard with
`streamlit run .\app.py`. Run an isolated test scenario with
`python .\app.py --simulate --scenario normal --duration-seconds 30`.
Other scenarios are `exercise`, `high_hr`, `low_spo2`, `no_finger`, and
`offline`; simulation writes only to `cardioMonitor_test`.

# Project rules and architecture

## Project rules

- Keep this project beginner-friendly and explain each step briefly with the exact Windows command to run.
- Hardware is complete. The ESP32 sends data to Firebase Realtime Database at `https://major-project-t13-default-rtdb.asia-southeast1.firebasedatabase.app/`, under the `cardioMonitor` node.
- The node contains `heartRate`, `spo2`, raw ADC `ecg` (0-4095), `acceleration` (`x`, `y`, `z`), `deviceStatus`, `lastUpdateMs`, and `risk`.
- Treat the Firebase service-account key as a secret. Never print or display it, upload it, or commit it. Keep it local and ignored by Git.
- Do not create files or folders beyond this allowlist without asking first: `serviceAccountKey.json`, `AGENTS.md`, `requirements.txt`, `.gitignore`, `train.py`, `app.py`, `model.pkl`, and `data/` with `mitbih/`, `patients/`, and `results/` subfolders.
- Keep the project folder lean.

## Architecture

- The ESP32 publishes wearable readings and its current `risk` value to Firebase Realtime Database at `cardioMonitor`.
- `train.py` will prepare permitted training data and train/evaluate the heart-disease risk model, saving the trained model as `model.pkl`.
- `app.py` will provide the Streamlit application, read the live `cardioMonitor` readings from Firebase, use the trained model for risk prediction, and present the results.
- Python dependencies are listed in `requirements.txt`. Use a local Windows virtual environment (`.venv`) for development.
