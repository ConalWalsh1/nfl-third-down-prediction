"""NFL third-down conversion demo. Model files contain trusted sklearn pipelines."""
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import joblib

st.set_page_config(page_title="NFL Third-Down Predictor", page_icon="🏈", layout="wide")
HERE = Path(__file__).resolve().parent

@st.cache_resource
def load_models():
    return {
        "Logistic regression": joblib.load(HERE / "logistic_model.joblib"),
        "Neural network": joblib.load(HERE / "neural_model.joblib"),
    }

models = load_models()
st.title("🏈 NFL Third-Down Conversion Predictor")
st.caption("Interactive research demonstration • Trained on 2023–2025 regular-season third-down plays • Predictions are estimates, not guarantees")
st.markdown("Set the pre-snap situation below. The two trained models estimate the probability that the offense converts the third down.")

with st.sidebar:
    st.header("Game situation")
    week = st.slider("Week", 1, 18, 14)
    yards = st.slider("Yards to go", 1, 30, 5)
    yardline = st.slider("Yards from opponent's end zone", 1, 99, 40, help="Opponent's 40-yard line = 40; own 40-yard line = 60")
    goal = st.checkbox("Goal-to-go", value=False)
    score = st.slider("Offense score differential", -35, 35, -3, help="Negative = offense is trailing")
    drive = st.caption("The current saved models do not use drive, home/away, or team identity; those source fields were absent from the training export.")
    half = st.radio("Half", ["First", "Second", "Overtime"], index=1, horizontal=True)
    if half == "Overtime":
        seconds_half = None
        seconds_game = st.number_input("Overtime seconds remaining (placeholder)", 0, 900, 600)
        st.caption("Overtime half clock was missing during training; it is imputed by the model.")
    else:
        seconds_half = st.slider("Seconds remaining in current half", 0, 1800, 480, step=15)
        seconds_game = seconds_half + (1800 if half == "First" else 0)
    st.header("Offense")
    formation = st.selectbox("Formation", ["SHOTGUN", "UNDER CENTER", "PISTOL"])
    personnel = st.selectbox("Offensive personnel", ["11 personnel", "12 personnel", "10 personnel", "21 personnel", "13 personnel", "22 personnel", "00 personnel", "Other / atypical"])
    shotgun = st.checkbox("Shotgun indicator", value=formation == "SHOTGUN", help="This indicator is separate from the recorded formation field.")
    no_huddle = st.checkbox("No huddle")
    st.header("Defense")
    defensive = st.selectbox("Defensive personnel", ["Nickel (5 DB)", "Dime (6 DB)", "Base (4 DB)", "Heavy Formation (5+ DL)", "Quarter (7 DB)", "Other DB count"])
    box = st.slider("Defenders in box", 2, 11, 5)
    st.header("Environment")
    roof = st.selectbox("Stadium roof", ["outdoors", "dome", "closed", "open"])
    indoor = roof in ("dome", "closed")
    if indoor:
        st.info("Indoor game: temperature and wind treated as missing / not applicable, as in the training data.")
        temperature = np.nan
        wind = np.nan
        precip = "Not Applicable (Indoor)"
    else:
        missing_weather = st.checkbox("Temperature / wind unavailable")
        if missing_weather:
            temperature = np.nan
            wind = np.nan
        else:
            temperature = st.number_input("Temperature (°F)", min_value=-20, max_value=120, value=10, step=1)
            wind = st.number_input("Wind (mph)", min_value=0, max_value=60, value=5, step=1)
        precip = st.selectbox("Precipitation", ["Unspecified", "Rain", "Snow", "Unknown", "Not Applicable"])

row = {
    "ydstogo": yards,
    "yardline_100": yardline,
    "goal_to_go": int(goal),
    "score_differential": score,
    "game_seconds_remaining": float(seconds_game),
    "half_seconds_remaining": float(seconds_half) if seconds_half is not None else np.nan,
    "half_clock_missing": int(seconds_half is None),
    "week": week,
    "shotgun": int(shotgun),
    "no_huddle": int(no_huddle),
    "offense_formation": formation,
    "off_personnel_group": personnel,
    "def_formation_group": defensive,
    "defenders_in_box": box,
    "temperature_clean": float(temperature),
    "wind_speed_clean": float(wind),
    "temp_missing": int(pd.isna(temperature)),
    "wind_missing": int(pd.isna(wind)),
    "precipitation_model": precip,
    "roof": roof,
    "indoor_game": int(indoor),
}

# Compare pipelines using precisely the feature order saved with each trained model.
probs = {}
for name, bundle in models.items():
    frame = pd.DataFrame([row])[bundle["features"]]
    probs[name] = float(bundle["pipeline"].predict_proba(frame)[0, 1])

st.subheader(f"Scenario: 3rd & {yards} · {formation.title()} · {personnel} · {defensive}")
a, b = st.columns(2)
with a:
    st.metric("Logistic regression", f"{100*probs['Logistic regression']:.1f}%", help="Primary BEMO model")
    st.progress(probs["Logistic regression"])
with b:
    st.metric("Neural network", f"{100*probs['Neural network']:.1f}%", help="Alternative model")
    st.progress(probs["Neural network"])
st.caption(f"Difference: {abs(100*(probs['Logistic regression']-probs['Neural network'])):.1f} percentage points. These are conversion probabilities, not predicted final game scores.")

st.subheader("Side-by-side comparison")
st.bar_chart(pd.DataFrame({"Model": list(probs), "Conversion probability (%)": [100*v for v in probs.values()]}).set_index("Model"))
with st.expander("Model inputs sent to the predictors"):
    st.dataframe(pd.DataFrame([row]).T.rename(columns={0:"Input value"}), use_container_width=True)
with st.expander("Methodology and important limitations"):
    st.markdown("""
- Models are **pre-snap classification models** returning an estimated probability of converting a third down.
- The primary model is a regularized logistic regression; the alternative is a feed-forward neural network with one 16-unit hidden layer.
- Source dataset included **20,407** filtered plays; partitioning occurred at the game level across 2023–2025, to avoid placing plays from the same game in both training and evaluation.
- The saved models **do not use** franchise identity, drive, home/away, surface, or betting lines; these fields were missing from the derived model-ready table.
- The personnel categories are based on player counts, **not verified coverage schemes or an Empty formation**. An Empty set is not separately coded in the model.
- Weather values are sometimes missing, especially indoors. Missing flags are passed and training-data imputers handle numerical gaps.
- The model was evaluated on historically randomized games, not an independently held-out future season.
- The dashboard is for classroom research, not real-time wagering or live operational play-calling.
""")
