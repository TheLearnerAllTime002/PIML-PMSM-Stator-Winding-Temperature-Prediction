import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

from engineering import (
    engineering_diagnosis,
    recommend_actions,
    risk_level,
    explain_engineering
)

xgb = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# load target scaler only if your model target was scaled
try:
    target_scaler = joblib.load("target_scaler.pkl")
except FileNotFoundError:
    target_scaler = None

explainer = joblib.load("explainer.pkl")
feature_names = joblib.load("feature_names.pkl")

st.set_page_config(
    page_title="PMSM Thermal Decision Support System",
    layout="wide"
)

st.title("PMSM Thermal Decision Support System")

st.write(
    "Physics-Guided Explainable Machine Learning Framework "
    "for Stator Winding Temperature Prediction"
)

st.sidebar.header("Operating Conditions")

current = st.sidebar.slider(
    "Current Magnitude (A)",
    min_value=0.0,
    max_value=380.0,
    value=107.2,
    step=0.1
)

voltage = st.sidebar.slider(
    "Voltage Magnitude (V)",
    min_value=0.0,
    max_value=140.0,
    value=97.8,
    step=0.1
)

omega = st.sidebar.slider(
    "Rotor Speed (rad/s)",
    min_value=-30.0,
    max_value=630.0,
    value=196.0,
    step=0.5
)

torque = st.sidebar.slider(
    "Torque (Nm)",
    min_value=-250.0,
    max_value=270.0,
    value=10.5,
    step=0.1
)

coolant = st.sidebar.slider(
    "Coolant Temperature (°C)",
    min_value=10.0,
    max_value=95.0,
    value=27.7,
    step=0.1
)

ambient = st.sidebar.slider(
    "Ambient Temperature (°C)",
    min_value=8.5,
    max_value=30.5,
    value=24.8,
    step=0.1
)

t_prev = st.sidebar.slider(
    "Previous Winding Temperature (°C)",
    min_value=18.5,
    max_value=142.0,
    value=64.8,
    step=0.1
)

st.sidebar.markdown(
    "### Risk thresholds\n"
    "- LOW: < 60 °C\n"
    "- MODERATE: 60–90 °C\n"
    "- HIGH: 90–120 °C\n"
    "- CRITICAL: ≥ 120 °C"
)

sample = pd.DataFrame(
    [[
        current,
        voltage,
        omega,
        torque,
        coolant,
        ambient,
        t_prev
    ]],
    columns=feature_names
)

scaled = scaler.transform(sample)

raw_prediction = xgb.predict(scaled)[0]

if target_scaler is not None:
    prediction = float(target_scaler.inverse_transform([[raw_prediction]]))
else:
    prediction = float(raw_prediction)

shap_values = explainer.shap_values(scaled)

feature_df, group_df = engineering_diagnosis(
    feature_names,
    shap_values[0]
)

diagnosis = explain_engineering(group_df)

actions = recommend_actions(group_df)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Predicted Temperature",
        f"{prediction:.2f} °C"
    )

with col2:
    risk = risk_level(prediction)

    if risk == "LOW":
        st.success(risk)
    elif risk == "MODERATE":
        st.warning(risk)
    elif risk == "HIGH":
        st.error(risk)
    else:
        st.error(risk)

with col3:
    st.metric(
        "Dominant Engineering Factor",
        group_df.iloc[0]["Engineering Group"]
    )

st.divider()

st.subheader("Prediction Summary")

st.write(f"Predicted Winding Temperature: **{prediction:.2f} °C**")

st.write(
    f"Dominant Engineering Factor: **{group_df.iloc[0]['Engineering Group']}**"
)

st.write(f"Risk Level: **{risk}**")

st.divider()

st.subheader("Feature Contributions")

fig = plt.figure(figsize=(9, 5))

shap.plots.bar(
    shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=sample.iloc[0].values,
        feature_names=feature_names
    ),
    show=False
)

st.pyplot(fig)

st.divider()

st.subheader("Engineering Contribution")

group_plot = (
    group_df
    .sort_values("SHAP", ascending=False)
    .set_index("Engineering Group")
)

st.bar_chart(group_plot["SHAP"])

st.divider()

st.subheader("Engineering Diagnosis")

if diagnosis:
    for line in diagnosis:
        st.write(line)
else:
    st.write("No significant positive engineering contributors were identified.")

st.divider()

st.subheader("Recommended Actions")

if actions:
    for action in actions:
        st.info(action)
else:
    st.write("No corrective actions are recommended for the current operating condition.")

st.divider()

st.subheader("Feature Contribution Details")

st.dataframe(
    feature_df.style.format({"SHAP": "{:.4f}"})
)

st.subheader("Engineering Group Summary")

st.dataframe(
    group_df.style.format({"SHAP": "{:.4f}"})
)