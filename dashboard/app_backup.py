
import sys
import json
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

# Make src modules accessible.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fuzzy_logic import assess_machine


DATA_PATH = ROOT / "dataset" / "raw" / "predictive_maintenance_v3.csv"
MODEL_PATH = ROOT / "models" / "ann_pipeline.joblib"
GA_PATH = ROOT / "results" / "ga_maintenance_schedule.csv"
ANN_METRICS_PATH = ROOT / "results" / "ann_metrics.json"
GA_METRICS_PATH = ROOT / "results" / "ga_metrics.json"

NUMERIC_FEATURES = [
    "vibration_rms",
    "temperature_motor",
    "current_phase_avg",
    "pressure_level",
    "rpm",
    "hours_since_maintenance",
    "ambient_temp",
]

CATEGORICAL_FEATURES = ["machine_type", "operating_mode"]


st.set_page_config(
    page_title="Predictive Maintenance System",
    page_icon="⚙️",
    layout="wide"
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["timestamp"] = pd.to_datetime(
        df["timestamp"], errors="coerce"
    )
    return df.dropna(subset=["timestamp", "machine_id"])


@st.cache_data
def load_machine_assessments():
    df = load_data()
    model = load_model()

    latest = (
        df.sort_values("timestamp")
        .groupby("machine_id")
        .tail(1)
        .copy()
    )

    for col in NUMERIC_FEATURES:
        latest[col] = pd.to_numeric(
            latest[col], errors="coerce"
        )
        latest[col] = latest[col].fillna(
            pd.to_numeric(df[col], errors="coerce").median()
        )

    for col in CATEGORICAL_FEATURES:
        latest[col] = latest[col].fillna(
            df[col].mode().iloc[0]
        )

    X = latest[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    latest["failure_probability"] = model.predict_proba(X)[:, 1]

    assessments = []

    for _, row in latest.iterrows():
        fuzzy = assess_machine(
            temperature=float(row["temperature_motor"]),
            vibration=float(row["vibration_rms"]),
            failure_probability=float(row["failure_probability"]),
            hours_since_maintenance=float(row["hours_since_maintenance"])
        )

        assessments.append({
            "machine_id": str(row["machine_id"]),
            "machine_type": str(row["machine_type"]),
            "timestamp": row["timestamp"],
            "temperature": float(row["temperature_motor"]),
            "vibration": float(row["vibration_rms"]),
            "failure_probability": float(row["failure_probability"]),
            "urgency_score": float(fuzzy["urgency_score"]),
            "health_score": float(fuzzy["health_score"]),
            "severity": fuzzy["severity"],
            "recommendation": fuzzy["recommendation"],
        })

    return pd.DataFrame(assessments)


# ---------------- PAGE HEADER ----------------

st.title("⚙️ AI-Based Predictive Maintenance System")
st.caption(
    "Hybrid Soft Computing: Artificial Neural Network + "
    "Mamdani Fuzzy Logic + Genetic Algorithm"
)

st.divider()


# ---------------- LOAD DATA ----------------

try:
    machines = load_machine_assessments()
    raw_data = load_data()
except Exception as error:
    st.error(f"Unable to load project data: {error}")
    st.stop()


# ---------------- SIDEBAR ----------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Dashboard View",
    [
        "Overview",
        "Machine Monitoring",
        "Maintenance Schedule",
        "Model Performance"
    ]
)


# ---------------- OVERVIEW ----------------

if page == "Overview":

    st.subheader("System Overview")

    total_machines = len(machines)
    avg_health = machines["health_score"].mean()
    high_risk = machines["severity"].isin(
        ["HIGH", "CRITICAL"]
    ).sum()
    avg_failure = machines["failure_probability"].mean() * 100

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Machines", total_machines)
    c2.metric("Average Health", f"{avg_health:.1f}/100")
    c3.metric("High / Critical", int(high_risk))
    c4.metric("Average Failure Risk", f"{avg_failure:.2f}%")

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Machine Health Scores")

        fig = px.bar(
            machines.sort_values("health_score"),
            x="machine_id",
            y="health_score",
            color="severity",
            title="Health Score by Machine",
            labels={
                "machine_id": "Machine ID",
                "health_score": "Health Score"
            }
        )

        fig.update_layout(yaxis_range=[0, 100])
        st.plotly_chart(fig, width="stretch")

    with right:
        st.subheader("Failure Probability")

        fig = px.bar(
            machines.sort_values("failure_probability", ascending=False),
            x="machine_id",
            y="failure_probability",
            color="failure_probability",
            title="ANN-Predicted Failure Probability",
            labels={
                "machine_id": "Machine ID",
                "failure_probability": "Failure Probability"
            }
        )

        fig.update_layout(yaxis_tickformat=".1%")
        st.plotly_chart(fig, width="stretch")

    st.subheader("Current Machine Assessments")

    st.dataframe(
        machines[
            [
                "machine_id",
                "machine_type",
                "failure_probability",
                "health_score",
                "urgency_score",
                "severity",
                "recommendation"
            ]
        ].sort_values("urgency_score", ascending=False),
        width="stretch",
        hide_index=True
    )


# ---------------- MACHINE MONITORING ----------------

elif page == "Machine Monitoring":

    st.subheader("Individual Machine Monitoring")

    selected_id = st.selectbox(
        "Select Machine",
        machines["machine_id"].tolist()
    )

    machine = machines[
        machines["machine_id"] == selected_id
    ].iloc[0]

    history = raw_data[
        raw_data["machine_id"].astype(str) == selected_id
    ].sort_values("timestamp")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Failure Probability",
        f"{machine['failure_probability']:.2%}"
    )
    c2.metric(
        "Health Score",
        f"{machine['health_score']:.1f}/100"
    )
    c3.metric(
        "Urgency",
        f"{machine['urgency_score']:.1f}/100"
    )
    c4.metric("Severity", machine["severity"])

    st.info(machine["recommendation"])

    st.subheader("Sensor History")

    sensor = st.selectbox(
        "Choose Sensor",
        [
            "temperature_motor",
            "vibration_rms",
            "current_phase_avg",
            "pressure_level",
            "rpm"
        ]
    )

    fig = px.line(
        history,
        x="timestamp",
        y=sensor,
        markers=True,
        title=f"{sensor} over time"
    )

    st.plotly_chart(fig, width="stretch")


# ---------------- GA MAINTENANCE SCHEDULE ----------------

elif page == "Maintenance Schedule":

    st.subheader("Genetic Algorithm — Optimized Schedule")

    if not GA_PATH.exists():
        st.warning("Run the Genetic Algorithm module first.")
    else:
        schedule = pd.read_csv(GA_PATH)

        if GA_METRICS_PATH.exists():
            with open(GA_METRICS_PATH, encoding="utf-8") as file:
                ga_metrics = json.load(file)

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Original-order Cost",
                f"{ga_metrics['fifo_cost']:.4f}"
            )
            c2.metric(
                "Priority-order Cost",
                f"{ga_metrics['priority_order_cost']:.4f}"
            )
            c3.metric(
                "GA Optimized Cost",
                f"{ga_metrics['ga_cost']:.4f}"
            )

        st.caption(
            "This prototype schedules machine inspections/maintenance "
            "by priority. A scheduled machine is not necessarily a "
            "machine that has already failed."
        )

        st.dataframe(
            schedule,
            width="stretch",
            hide_index=True
        )

        st.subheader("Machines Scheduled Per Day")

        daily_counts = (
            schedule.groupby("scheduled_day")
            .size()
            .reset_index(name="machine_count")
        )

        fig = px.bar(
            daily_counts,
            x="scheduled_day",
            y="machine_count",
            text="machine_count",
            title="Maintenance Workload by Day"
        )

        st.plotly_chart(fig, width="stretch")

        st.download_button(
            "Download Maintenance Schedule",
            schedule.to_csv(index=False),
            file_name="maintenance_schedule.csv",
            mime="text/csv"
        )


# ---------------- MODEL PERFORMANCE ----------------

elif page == "Model Performance":

    st.subheader("ANN Model Evaluation")

    if ANN_METRICS_PATH.exists():
        with open(ANN_METRICS_PATH, encoding="utf-8") as file:
            metrics = json.load(file)

        c1, c2, c3, c4, c5 = st.columns(5)

        c1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
        c2.metric("Precision", f"{metrics['precision']:.2%}")
        c3.metric("Recall", f"{metrics['recall']:.2%}")
        c4.metric("F1 Score", f"{metrics['f1_score']:.2%}")
        c5.metric("ROC-AUC", f"{metrics['roc_auc']:.2%}")

        st.subheader("Confusion Matrix")

        cm = metrics["confusion_matrix"]

        fig = px.imshow(
            cm,
            text_auto=True,
            x=["Predicted Normal", "Predicted Failure"],
            y=["Actual Normal", "Actual Failure"],
            color_continuous_scale="Blues",
            title="ANN Confusion Matrix"
        )

        st.plotly_chart(fig, width="stretch")

    else:
        st.warning("ANN metrics file not found.")

    st.caption(
        "These are preliminary results from the chronological "
        "training/testing split. Further baseline and generalization "
        "experiments are still required."
    )


# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "Academic prototype | ANN + Fuzzy Logic + Genetic Algorithm"
)
