# NFL Third-Down Predictor — Streamlit Dashboard

Interactive classroom demo comparing the 2023–2025 third-down logistic regression and a 16-neuron neural network.

## Run locally

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy

1. Create a GitHub repository named `nfl-third-down-predictor`.
2. Upload every file from this folder to the repository root (including both `.joblib` model files).
3. Open https://share.streamlit.io and connect GitHub.
4. Choose **Create app** → **Yup, I have an app** and select the repository, branch (`main`), and `app.py` entrypoint.
5. Deploy. Open the resulting `*.streamlit.app` URL, test it, and make a QR code for your presentation.

## Model notes

- Both `.joblib` bundles contain their training preprocessing pipelines and list the required feature columns.
- Scikit-learn version is pinned to the version used for serialization (`1.8.0`). Do not load arbitrary downloaded joblib files; deserialization can execute code.
- Field `posteam_type`, `drive`, and other requested variables are not present in the current saved model. UI only requests features that were actually trained.
- `offense_formation` supports `SHOTGUN`, `UNDER CENTER`, `PISTOL`; Empty is not an independently labeled input.
- The dataset's `half_seconds_remaining` is an explicit clock predictor. The dashboard derives it from the selected half and seconds remaining.

## Testing checklist

- Compare a specific scenario's values from this app with `bundle['pipeline'].predict_proba(pd.DataFrame([inputs])[bundle['features']])[:,1]` in Python.
- Check indoor/outdoor combinations, zero seconds remaining, goal-to-go, unusual personnel, and multiple simultaneous visitors.
- Test a phone and laptop before presenting.
