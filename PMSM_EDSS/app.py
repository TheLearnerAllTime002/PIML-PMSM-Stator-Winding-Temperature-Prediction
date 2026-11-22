"""Local realtime dashboard backed by the saved PMSM thermal model."""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import streamlit as st
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Explainability"))
from shap_rul_analysis import FEATURES, engineer, load_model, load_scaler  # noqa: E402

st.set_page_config(page_title="PMSM Thermal Digital Twin", layout="wide")
MODEL_PATH = ROOT / "models" / "rq4_v2_best.pth"
SCALER_PATH = ROOT / "models" / "feature_scaler.pkl"

@st.cache_resource
def get_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_model(MODEL_PATH, device)
    return model, load_scaler(SCALER_PATH), checkpoint, device

def make_window(values, scaler, sequence_length):
    row = engineer(pd.DataFrame([values]))
    scaled = scaler.transform(row[FEATURES]).astype(np.float32)[0]
    return np.repeat(scaled[None, :], sequence_length, axis=0)

def make_history_window(history, scaler, sequence_length):
    history = engineer(history.copy())
    if len(history) < sequence_length:
        raise ValueError(f"CSV needs at least {sequence_length} rows; received {len(history)}")
    return scaler.transform(history[FEATURES].tail(sequence_length)).astype(np.float32)

@st.cache_data(show_spinner=False)
def global_shap_table():
    path = ROOT / "Explainability" / "results" / "shap_global_importance.csv"
    return pd.read_csv(path) if path.exists() else None

st.title("PMSM Thermal Digital Twin")
st.caption("Realtime inference from the saved RQ4 V2 trained model — no retraining")
try:
    model, scaler, checkpoint, device = get_model()
except Exception as exc:
    st.error(f"Could not load the trained model or scaler: {exc}")
    st.stop()

horizon = int(checkpoint.get("horizon", 20))
sequence_length = int(checkpoint.get("sequence_length", 128))

with st.sidebar:
    st.header("Live operating point")
    uploaded_file = st.file_uploader(
        "Upload latest sensor history (CSV)",
        type=["csv"],
        help="Preferred mode: upload at least 128 consecutive rows with the raw sensor columns. The last 128 rows are sent to the trained model."
    )
    st.caption("Preferred: upload 128 real consecutive sensor rows. Sliders below are a demo mode.")
    u_q = st.slider("q-axis voltage u_q (V)", -150.0, 150.0, -0.35, 0.01,
                    help="q-axis electrical voltage; affects voltage magnitude and electrical power.")
    u_d = st.slider("d-axis voltage u_d (V)", -150.0, 150.0, -0.33, 0.01,
                    help="d-axis electrical voltage; affects voltage magnitude and electrical power.")
    i_d = st.slider("d-axis current i_d (A)", -400.0, 400.0, -0.05, 0.01,
                    help="d-axis current component; combines with i_q to determine current magnitude.")
    i_q = st.slider("q-axis current i_q (A)", -400.0, 400.0, 0.03, 0.01,
                    help="q-axis current component; strongly influences current magnitude and copper heating.")
    motor_speed = st.slider("Motor speed (RPM)", -1000.0, 7000.0, 2000.0, 10.0,
                            help="rotational speed; contributes to mechanical power and heat generation.")
    torque = st.slider("Torque (Nm)", -300.0, 300.0, 30.0, 1.0,
                       help="mechanical load; together with speed determines mechanical power.")
    coolant = st.slider("Coolant temperature (°C)", 0.0, 100.0, 25.0, 0.5,
                        help="coolant temperature; higher coolant temperature reduces cooling capacity.")
    ambient = st.slider("Ambient temperature (°C)", 0.0, 60.0, 25.0, 0.5,
                        help="surrounding air temperature used during the forecast rollout.")
    current_temperature = st.slider("Current PM temperature (°C)", 0.0, 180.0, 65.0, 0.5,
                                    help="latest measured permanent-magnet temperature, used as the forecast initial state.")
    limit = st.slider("Thermal limit for RUL (°C)", 40.0, 200.0, 120.0, 1.0,
                      help="temperature at which the thermal-limit RUL countdown is triggered.")

with st.expander("Which feature contributes what?", expanded=False):
    st.markdown("""
    **Direct sensor inputs**

    - **u_d, u_q:** d/q-axis voltages. Together they form voltage magnitude and electrical power.
    - **i_d, i_q:** d/q-axis currents. Their combined magnitude drives copper-loss heating.
    - **Motor speed:** rotational speed. It affects mechanical power and the thermal operating regime.
    - **Torque:** mechanical load. Speed × torque gives mechanical power.
    - **Coolant:** coolant temperature. A hotter coolant removes less heat.
    - **Ambient:** surrounding temperature. A hotter environment reduces the temperature gradient for cooling.

    **Engineered features used by the trained model**

    - **I_mag:** √(i_d² + i_q²), the current magnitude.
    - **V_mag:** √(u_d² + u_q²), the voltage magnitude.
    - **P_elec:** u_d·i_d + u_q·i_q, electrical power proxy.
    - **P_mech:** torque × 2π × speed / 60, mechanical power proxy.
    - **P_cu:** I_mag² × 0.05, copper-loss heating proxy.

    The SHAP chart ranks the model’s learned influence on the forecast. A high
    SHAP value means the feature changes the prediction strongly; it does not
    by itself prove causality.
    """)

values = {"u_q": u_q, "coolant": coolant, "u_d": u_d, "motor_speed": motor_speed,
          "i_d": i_d, "i_q": i_q, "torque": torque, "ambient": ambient}
history_df = None
history_error = None
if uploaded_file is not None:
    try:
        history_df = pd.read_csv(uploaded_file)
        required = ["u_q", "coolant", "u_d", "motor_speed", "i_d", "i_q", "torque", "ambient"]
        missing = [column for column in required if column not in history_df.columns]
        if missing:
            raise ValueError("Missing columns: " + ", ".join(missing))
        window = make_history_window(history_df, scaler, sequence_length)
        latest = history_df.iloc[-1]
        values = {column: float(latest[column]) for column in required}
        ambient = float(latest["ambient"])
        if "pm" in history_df.columns:
            current_temperature = float(history_df["pm"].iloc[-1])
        st.sidebar.success(f"Using real history: {len(history_df)} rows; last {sequence_length} rows selected.")
    except Exception as exc:
        history_error = str(exc)
        window = make_window(values, scaler, sequence_length)
else:
    window = make_window(values, scaler, sequence_length)
if history_error:
    st.sidebar.error(f"History upload ignored: {history_error}")
ambient_future = np.full((1, horizon), ambient, dtype=np.float32)
x_tensor = torch.tensor(window[None, ...], dtype=torch.float32, device=device)
t_tensor = torch.tensor([current_temperature], dtype=torch.float32, device=device)
a_tensor = torch.tensor(ambient_future, dtype=torch.float32, device=device)
pcu_future = x_tensor[:, -1, 12].unsqueeze(1).expand(-1, horizon)
with torch.no_grad():
    forecast = model(x_tensor, t_tensor, a_tensor, pcu_future).cpu().numpy()[0]

hit = np.flatnonzero(forecast >= limit)
rul = int(hit[0] + 1) if len(hit) else horizon
peak = float(forecast.max())
risk = "LOW" if peak < 60 else "MODERATE" if peak < 90 else "HIGH" if peak < limit else "CRITICAL"
c1, c2, c3, c4 = st.columns(4)
c1.metric("Next-step PM temperature", f"{forecast[0]:.2f} °C")
c2.metric("Forecast peak", f"{peak:.2f} °C")
c3.metric("Thermal-limit RUL", f"{rul} steps" + ("+" if rul == horizon else ""))
c4.metric("Risk", risk)

st.subheader("Realtime temperature forecast")
forecast_df = pd.DataFrame({"Step": np.arange(1, horizon + 1), "Predicted PM temperature (°C)": forecast})
st.line_chart(forecast_df.set_index("Step"))
if history_df is not None:
    st.success(f"Model input: last {sequence_length} real rows from the uploaded CSV.")
else:
    st.warning(f"Model input: one slider snapshot repeated {sequence_length} times. Upload real history for reliable thermal-memory prediction.")
st.caption("Scenario: latest copper-loss/operating point and ambient temperature are held constant over the forecast horizon.")

left, right = st.columns(2)
with left:
    st.subheader("Current engineered state")
    engineered = engineer(pd.DataFrame([values]))
    st.dataframe(engineered[FEATURES].T.rename(columns={0: "value"}), width="stretch")
with right:
    st.subheader("Saved SHAP global importance")
    shap_table = global_shap_table()
    if shap_table is None:
        st.info("Run the SHAP analysis first to populate this panel.")
    else:
        shap_table = shap_table.copy()
        shap_table["factor"] = shap_table["feature"].str.replace(" (sequence)", "", regex=False)
        total_shap = shap_table["mean_abs_shap"].sum()
        shap_table["influence_percent"] = 100 * shap_table["mean_abs_shap"] / total_shap
        shap_table["rank"] = np.arange(1, len(shap_table) + 1)
        shap_table["interpretation"] = shap_table["factor"].map({
            "P_cu": "Copper-loss heating from current magnitude",
            "motor_speed": "Mechanical operating regime and speed-related losses",
            "V_mag": "Voltage magnitude and electrical loading",
            "u_q": "q-axis voltage contribution",
            "I_mag": "Current magnitude and copper heating",
            "torque": "Mechanical load and power generation",
            "i_d": "d-axis current contribution",
            "P_mech": "Mechanical power proxy",
            "P_elec": "Electrical power proxy",
            "ambient": "Environmental cooling boundary",
            "u_d": "d-axis voltage contribution",
            "i_q": "q-axis current contribution",
            "coolant": "Coolant temperature and cooling effectiveness",
        })
        st.bar_chart(shap_table.head(10).set_index("factor")["mean_abs_shap"])
        st.dataframe(
            shap_table[["rank", "factor", "mean_abs_shap", "influence_percent", "interpretation"]]
            .style.format({"mean_abs_shap": "{:.8f}", "influence_percent": "{:.2f}%"}),
            width="stretch",
            hide_index=True,
        )
        st.caption("Mean absolute SHAP amount across analysed windows. Percentages show relative model influence; SHAP importance is not a causal percentage of heat.")

st.divider()
st.subheader("Model provenance")
st.write(f"Checkpoint: `{MODEL_PATH.relative_to(ROOT)}`  \nDevice: `{device}`  \nInput history: `{sequence_length}` steps  \nForecast horizon: `{horizon}` steps")
st.warning("RUL is a thermal-limit proxy. The dataset contains no run-to-failure or degradation labels, so this is not validated component-life RUL.")
