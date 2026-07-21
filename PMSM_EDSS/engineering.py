import pandas as pd
import numpy as np

ENGINEERING_GROUPS = {
    "Thermal Memory": ["T_prev"],
    "Electrical Heating": ["current_magnitude", "voltage_magnitude"],
    "Mechanical Loading": ["omega", "torque"],
    "Cooling": ["coolant"],
    "Environment": ["ambient"]
}

feature_to_group = {}
for group, feats in ENGINEERING_GROUPS.items():
    for f in feats:
        feature_to_group[f] = group


def engineering_diagnosis(feature_names, shap_values):
    feature_df = pd.DataFrame({
        "Feature": feature_names,
        "SHAP": shap_values
    })

    feature_df["Engineering Group"] = feature_df["Feature"].map(feature_to_group)

    group_df = (
        feature_df.groupby("Engineering Group")["SHAP"]
        .sum()
        .reset_index()
        .sort_values("SHAP", ascending=False)
    )

    return feature_df, group_df


def explain_engineering(group_df):
    diagnosis = []

    for _, row in group_df.iterrows():
        if row["SHAP"] <= 0:
            continue

        if row["Engineering Group"] == "Thermal Memory":
            diagnosis.append("Residual thermal energy dominates the current temperature.")
        elif row["Engineering Group"] == "Electrical Heating":
            diagnosis.append("Electrical loading contributes significantly to stator heating.")
        elif row["Engineering Group"] == "Mechanical Loading":
            diagnosis.append("Mechanical loading contributes to heat generation.")
        elif row["Engineering Group"] == "Cooling":
            diagnosis.append("Cooling effectiveness influences the winding temperature.")
        elif row["Engineering Group"] == "Environment":
            diagnosis.append("Ambient conditions have a minor thermal influence.")

    return diagnosis


def recommend_actions(group_df):
    actions = []

    for _, row in group_df.head(3).iterrows():
        group = row["Engineering Group"]

        if group == "Electrical Heating":
            actions.append("Reduce stator current if operationally feasible.")
        elif group == "Cooling":
            actions.append("Improve coolant flow or reduce coolant temperature.")
        elif group == "Mechanical Loading":
            actions.append("Reduce torque demand or operating speed.")
        elif group == "Thermal Memory":
            actions.append("Allow sufficient cooling time before increasing load.")
        elif group == "Environment":
            actions.append("Reduce ambient temperature where possible.")

    return actions


def risk_level(temp):
    if temp < 60:
        return "LOW"
    elif temp < 90:
        return "MODERATE"
    elif temp < 120:
        return "HIGH"
    else:
        return "CRITICAL"