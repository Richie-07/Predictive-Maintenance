import sys
import json
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib

# ---------------- PATHS & CONFIGURATION ----------------

ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "src"
sys.path.insert(0, str(SRC_PATH))

from fuzzy_logic import assess_machine

DATA_PATH = ROOT / "dataset" / "raw" / "predictive_maintenance_v3.csv"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
GA_PATH = RESULTS_DIR / "ga_maintenance_schedule.csv"

MODEL_FILES = {
    "Original ANN (9 features)": "ann_metrics.json",
    "Four-feature ANN": "ann_fair_metrics.json",
    "Original ANFIS": "anfis_metrics.json",
    "Balanced ANFIS": "anfis_balanced_metrics.json",
}

PREDICTION_FILES = {
    "Original ANN (9 features)": "ann_predictions.csv",
    "Four-feature ANN": "ann_fair_predictions.csv",
    "Original ANFIS": "anfis_predictions.csv",
    "Balanced ANFIS": "anfis_balanced_predictions.csv",
}

st.set_page_config(
    page_title="AI Predictive Maintenance | Evaluation Dashboard",
    page_icon="⚙️",
    layout="wide",
)

# ---------------- DATA LOADING ----------------

@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


@st.cache_data
def load_json(filename):
    path = RESULTS_DIR / filename
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_csv(filename):
    path = RESULTS_DIR / filename
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_resource
def load_ann_model():
    path = MODEL_DIR / "ann_pipeline.joblib"
    if not path.exists():
        raise FileNotFoundError(f"ANN model not found: {path}")
    return joblib.load(path)


@st.cache_data
def load_machine_assessments():
    df = load_data()
    required = [
        "machine_id", "timestamp", "vibration_rms", "temperature_motor",
        "current_phase_avg", "pressure_level", "rpm",
        "hours_since_maintenance", "ambient_temp", "machine_type",
        "operating_mode",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required dashboard columns: {missing}")

    numeric_features = [
        "vibration_rms", "temperature_motor", "current_phase_avg",
        "pressure_level", "rpm", "hours_since_maintenance", "ambient_temp",
    ]
    categorical_features = ["machine_type", "operating_mode"]

    latest = (
        df.dropna(subset=["timestamp", "machine_id"])
        .sort_values("timestamp")
        .groupby("machine_id")
        .tail(1)
        .copy()
    )

    for col in numeric_features:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        latest[col] = pd.to_numeric(latest[col], errors="coerce")
        median_value = df[col].median()
        latest[col] = latest[col].fillna(0 if pd.isna(median_value) else median_value)

    for col in categorical_features:
        mode_values = df[col].dropna().mode()
        fallback = mode_values.iloc[0] if not mode_values.empty else "Unknown"
        latest[col] = latest[col].fillna(fallback)

    model = load_ann_model()
    X = latest[numeric_features + categorical_features]
    latest["failure_probability"] = model.predict_proba(X)[:, 1]

    records = []
    for _, row in latest.iterrows():
        fuzzy = assess_machine(
            temperature=float(row["temperature_motor"]),
            vibration=float(row["vibration_rms"]),
            failure_probability=float(row["failure_probability"]),
            hours_since_maintenance=float(row["hours_since_maintenance"]),
        )
        records.append({
            "machine_id": str(row["machine_id"]),
            "machine_type": str(row["machine_type"]),
            "timestamp": row["timestamp"],
            "temperature": float(row["temperature_motor"]),
            "vibration": float(row["vibration_rms"]),
            "current": float(row["current_phase_avg"]),
            "pressure": float(row["pressure_level"]),
            "rpm": float(row["rpm"]),
            "failure_probability": float(row["failure_probability"]),
            "urgency_score": float(fuzzy["urgency_score"]),
            "health_score": float(fuzzy["health_score"]),
            "severity": str(fuzzy["severity"]),
            "recommendation": str(fuzzy["recommendation"]),
        })
    return pd.DataFrame(records)


def metric_value(metrics, key):
    value = metrics.get(key)
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def find_column(df, candidates):
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def render_confusion_matrix(metrics, model_name):
    cm = metrics.get("confusion_matrix")
    if not isinstance(cm, list) or len(cm) != 2:
        st.info("A 2×2 confusion matrix is not available in this metrics file.")
        return
    try:
        fig = px.imshow(
            cm,
            text_auto=True,
            x=["Predicted normal", "Predicted failure"],
            y=["Actual normal", "Actual failure"],
            labels={"x": "Prediction", "y": "Actual", "color": "Count"},
            color_continuous_scale="Blues",
            title=f"{model_name} — Confusion Matrix",
        )
        fig.update_layout(height=380)
        st.plotly_chart(fig, width="stretch")
    except Exception as exc:
        st.warning(f"Could not render confusion matrix: {exc}")


def render_roc_comparison():
    """Draw ROC curves only when saved prediction files contain actual labels and probabilities."""
    from sklearn.metrics import roc_curve, auc

    plotted = 0
    fig = go.Figure()
    aliases_y = ["y_true", "actual", "actual_label", "true", "target", "failure_within_24h"]
    aliases_p = ["y_prob", "probability", "failure_probability", "predicted_probability",
                 "prob_failure", "positive_probability", "prediction_probability"]

    for model_name, filename in PREDICTION_FILES.items():
        pred = load_csv(filename)
        if pred is None:
            continue
        y_col = find_column(pred, aliases_y)
        p_col = find_column(pred, aliases_p)
        if y_col is None or p_col is None:
            continue
        y = pd.to_numeric(pred[y_col], errors="coerce")
        p = pd.to_numeric(pred[p_col], errors="coerce")
        valid = y.notna() & p.notna()
        y, p = y[valid], p[valid]
        if y.nunique() < 2:
            continue
        fpr, tpr, _ = roc_curve(y, p)
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            name=f"{model_name} (AUC {auc(fpr, tpr):.3f})",
        ))
        plotted += 1

    if plotted:
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line={"dash": "dash", "color": "gray"},
            name="Random classifier",
        ))
        fig.update_layout(
            title="ROC Curve Comparison",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            xaxis_range=[0, 1],
            yaxis_range=[0, 1],
            height=480,
        )
        st.plotly_chart(fig, width="stretch")
    else:
        st.info(
            "ROC curves are not displayed because the saved prediction CSVs "
            "do not contain a recognized actual-label and probability column pair. "
            "The ROC-AUC scores from the saved metrics are displayed above."
        )


# ---------------- LOAD CORE DATA ----------------

st.title("⚙️ AI-Based Predictive Maintenance System")
st.caption(
    "Hybrid Soft Computing Evaluation Dashboard | "
    "ANN + Mamdani Fuzzy Logic + ANFIS + Genetic Algorithm"
)

try:
    raw_data = load_data()
    machines = load_machine_assessments()
except Exception as error:
    st.error(f"Unable to load project data: {error}")
    st.stop()

# ---------------- NAVIGATION ----------------

pages = [
    "Project Overview",
    "Dataset Analysis",
    "Model Performance",
    "Fuzzy Logic",
    "Maintenance Optimization",
    "Machine Monitoring",
    "Experimental Results",
]
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Dashboard View", pages)
st.sidebar.divider()
st.sidebar.caption("Academic prototype | Saved experimental results")

# ---------------- PROJECT OVERVIEW ----------------

if page == "Project Overview":
    st.header("Project Overview")
    st.write(
        "A decision-support prototype that combines neural-network failure prediction, "
        "fuzzy assessment of machine condition, and genetic-algorithm maintenance scheduling."
    )

    total_rows = len(raw_data)
    total_machines = raw_data["machine_id"].nunique() if "machine_id" in raw_data else len(machines)
    target_col = "failure_within_24h"
    failure_rate = (
        pd.to_numeric(raw_data[target_col], errors="coerce").mean() * 100
        if target_col in raw_data.columns else None
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dataset Records", f"{total_rows:,}")
    c2.metric("Machines", f"{total_machines}")
    c3.metric("Average Machine Health", f"{machines['health_score'].mean():.1f}/100")
    c4.metric("Dataset Failure Rate", f"{failure_rate:.2f}%" if failure_rate is not None else "N/A")

    st.divider()
    st.subheader("System Architecture")
    arch = st.columns(4)
    arch[0].markdown("### 1. Sensor Data\nOperational readings and machine history")
    arch[1].markdown("### 2. ANN / ANFIS\nPredict failure probability")
    arch[2].markdown("### 3. Mamdani FIS\nEstimate health and maintenance urgency")
    arch[3].markdown("### 4. Genetic Algorithm\nOptimize maintenance order")

    st.info(
        "ANN and ANFIS are evaluated as alternative failure-prediction models. "
        "The Mamdani Fuzzy Inference System assesses machine condition, while "
        "the Genetic Algorithm optimizes the maintenance schedule."
    )

    st.subheader("Current Machine Assessments")
    st.dataframe(
        machines[[
            "machine_id", "machine_type", "failure_probability",
            "health_score", "urgency_score", "severity", "recommendation",
        ]].sort_values("urgency_score", ascending=False),
        width="stretch",
        hide_index=True,
    )

    left, right = st.columns(2)
    with left:
        fig = px.bar(
            machines.sort_values("health_score"),
            x="machine_id", y="health_score", color="severity",
            title="Health Score by Machine",
            labels={"machine_id": "Machine ID", "health_score": "Health Score"},
        )
        fig.update_layout(yaxis_range=[0, 100])
        st.plotly_chart(fig, width="stretch")
    with right:
        fig = px.bar(
            machines.sort_values("failure_probability", ascending=False),
            x="machine_id", y="failure_probability",
            color="failure_probability",
            title="ANN-Predicted Failure Probability",
            labels={"machine_id": "Machine ID", "failure_probability": "Failure Probability"},
        )
        fig.update_layout(yaxis_tickformat=".1%")
        st.plotly_chart(fig, width="stretch")


# ---------------- DATASET ANALYSIS ----------------

elif page == "Dataset Analysis":
    st.header("Dataset Analysis")
    st.caption("Dataset statistics are calculated from the raw CSV currently saved in the project.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(raw_data):,}")
    c2.metric("Columns", f"{len(raw_data.columns)}")
    c3.metric("Unique Machines", str(raw_data["machine_id"].nunique()) if "machine_id" in raw_data else "N/A")

    st.subheader("Dataset Preview")
    st.dataframe(raw_data.head(15), width="stretch", hide_index=True)

    st.subheader("Missing Values")
    missing = raw_data.isna().sum().reset_index()
    missing.columns = ["Feature", "Missing Values"]
    missing = missing[missing["Missing Values"] > 0].sort_values("Missing Values", ascending=False)
    if missing.empty:
        st.success("No missing values found in the raw dataset.")
    else:
        fig = px.bar(missing, x="Feature", y="Missing Values", title="Missing Values by Feature")
        st.plotly_chart(fig, width="stretch")
        st.dataframe(missing, width="stretch", hide_index=True)

    st.subheader("Failure Class Distribution")
    if "failure_within_24h" in raw_data.columns:
        counts = raw_data["failure_within_24h"].value_counts(dropna=False).rename_axis("Class").reset_index(name="Records")
        counts["Class Label"] = counts["Class"].map({0: "No failure", 1: "Failure within 24h"}).fillna(counts["Class"].astype(str))
        left, right = st.columns(2)
        with left:
            fig = px.pie(counts, names="Class Label", values="Records", hole=0.45, title="Target Distribution")
            st.plotly_chart(fig, width="stretch")
        with right:
            st.dataframe(counts[["Class Label", "Records"]], width="stretch", hide_index=True)

    if "machine_type" in raw_data.columns:
        st.subheader("Records by Machine Type")
        type_counts = raw_data["machine_type"].value_counts().rename_axis("Machine Type").reset_index(name="Records")
        fig = px.bar(type_counts, x="Machine Type", y="Records", title="Machine Type Distribution")
        st.plotly_chart(fig, width="stretch")

    st.subheader("Sensor Trends")
    if "timestamp" in raw_data.columns:
        sensor_options = [c for c in [
            "temperature_motor", "vibration_rms", "current_phase_avg",
            "pressure_level", "rpm", "ambient_temp",
        ] if c in raw_data.columns]
        if sensor_options:
            sensor = st.selectbox("Select sensor for dataset trend", sensor_options)
            trend = raw_data.dropna(subset=["timestamp"]).copy()
            trend[sensor] = pd.to_numeric(trend[sensor], errors="coerce")
            trend = trend.dropna(subset=[sensor]).sort_values("timestamp")
            fig = px.line(
                trend, x="timestamp", y=sensor, color="machine_type" if "machine_type" in trend else None,
                title=f"{sensor} over time",
            )
            st.plotly_chart(fig, width="stretch")


# ---------------- MODEL PERFORMANCE ----------------

elif page == "Model Performance":
    st.header("Model Performance Comparison")
    st.caption(
        "All metrics below are loaded from the saved evaluation JSON files. "
        "The test set is unchanged and contains 4,809 records."
    )

    loaded_metrics = {}
    for model_name, filename in MODEL_FILES.items():
        metrics = load_json(filename)
        if metrics is not None:
            loaded_metrics[model_name] = metrics

    if not loaded_metrics:
        st.error("No model metrics JSON files were found in the results folder.")
        st.stop()

    metric_keys = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    metric_labels = {
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1_score": "F1-score",
        "roc_auc": "ROC-AUC",
    }

    rows = []
    for name, metrics in loaded_metrics.items():
        row = {"Model": name}
        for key in metric_keys:
            value = metric_value(metrics, key)
            row[metric_labels[key]] = value * 100 if value is not None else None
        rows.append(row)
    comparison = pd.DataFrame(rows)

    st.subheader("Side-by-Side Metrics")
    st.dataframe(
        comparison.style.format({label: "{:.2f}%" for label in metric_labels.values()}, na_rep="—"),
        width="stretch",
        hide_index=True,
    )

    chart_metric = st.selectbox("Choose metric to compare", list(metric_labels.values()), index=3)
    fig = px.bar(
        comparison, x="Model", y=chart_metric, color="Model",
        text_auto=".2f", title=f"{chart_metric} Comparison",
    )
    fig.update_layout(yaxis_title=f"{chart_metric} (%)", yaxis_range=[0, 105], showlegend=False)
    st.plotly_chart(fig, width="stretch")

    st.subheader("Detailed Model Results")
    selected_model = st.selectbox("Select model", list(loaded_metrics.keys()))
    selected_metrics = loaded_metrics[selected_model]
    metric_cols = st.columns(5)
    for col, key in zip(metric_cols, metric_keys):
        value = metric_value(selected_metrics, key)
        col.metric(metric_labels[key], f"{value:.2%}" if value is not None else "N/A")

    render_confusion_matrix(selected_metrics, selected_model)

    with st.expander("Show saved metrics JSON"):
        st.json(selected_metrics)

    st.subheader("ROC Curve Comparison")
    render_roc_comparison()

    st.warning(
        "Interpretation: the original ANN uses a larger feature set, while the four-feature ANN "
        "and ANFIS experiments use four inputs. Compare the four-feature ANN and ANFIS as the "
        "closer controlled comparison. Results are specific to this dataset and split; they do "
        "not establish that one model is universally better."
    )


# ---------------- FUZZY LOGIC ----------------

elif page == "Fuzzy Logic":
    st.header("Mamdani Fuzzy Inference System")
    st.write(
        "The fuzzy system combines temperature, vibration, ANN failure probability, "
        "and hours since maintenance to estimate maintenance urgency and machine health."
    )

    st.subheader("Try an Interactive Machine Assessment")
    a, b = st.columns(2)
    with a:
        temperature = st.slider("Motor temperature (°C)", 0.0, 120.0, 75.0, 1.0)
        vibration = st.slider("Vibration RMS", 0.0, 10.0, 3.2, 0.1)
    with b:
        failure_probability = st.slider("Failure probability", 0.0, 1.0, 0.85, 0.01)
        hours_since_maintenance = st.slider("Hours since maintenance", 0.0, 1000.0, 300.0, 5.0)

    try:
        assessment = assess_machine(
            temperature=float(temperature),
            vibration=float(vibration),
            failure_probability=float(failure_probability),
            hours_since_maintenance=float(hours_since_maintenance),
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Urgency Score", f"{assessment['urgency_score']:.2f}/100")
        c2.metric("Health Score", f"{assessment['health_score']:.2f}/100")
        c3.metric("Severity", str(assessment["severity"]))
        st.info(str(assessment["recommendation"]))
    except Exception as exc:
        st.error(f"Fuzzy assessment could not be calculated: {exc}")

    st.subheader("Fuzzy Logic in This Project")
    st.markdown(
        """
        - **Fuzzification:** converts numerical sensor readings and risk into fuzzy values.
        - **Rule evaluation:** applies the Mamdani IF–THEN rule base.
        - **Aggregation:** combines the activated rule outputs.
        - **Defuzzification:** converts the aggregated fuzzy output into a numerical urgency score.
        - **Health score:** calculated by the implemented system as 100 minus urgency.
        """
    )

    fuzzy_example = load_json("fuzzy_example.json")
    if fuzzy_example:
        with st.expander("Previously saved fuzzy example"):
            st.json(fuzzy_example)

    rule_file = SRC_PATH / "fuzzy_logic.py"
    if rule_file.exists():
        with st.expander("View implemented fuzzy rule and membership-function source"):
            st.code(rule_file.read_text(encoding="utf-8"), language="python")


# ---------------- MAINTENANCE OPTIMIZATION ----------------

elif page == "Maintenance Optimization":
    st.header("Genetic Algorithm — Maintenance Scheduling")
    ga_metrics = load_json("ga_metrics.json")
    schedule = load_csv("ga_maintenance_schedule.csv")

    if ga_metrics is None:
        st.error("GA metrics file not found.")
    else:
        fifo = float(ga_metrics["fifo_cost"])
        priority = float(ga_metrics["priority_order_cost"])
        ga_cost = float(ga_metrics["ga_cost"])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Machines Scheduled", ga_metrics.get("machines_scheduled", "N/A"))
        c2.metric("Daily Capacity", ga_metrics.get("daily_capacity", "N/A"))
        c3.metric("GA Cost", f"{ga_cost:.4f}")
        c4.metric("Improvement vs FIFO", f"{ga_metrics.get('ga_improvement_vs_fifo_percent', 0):.2f}%")

        cost_df = pd.DataFrame({
            "Strategy": ["FIFO", "Priority-based", "Genetic Algorithm"],
            "Objective Cost": [fifo, priority, ga_cost],
        })
        st.subheader("Scheduling Strategy Comparison")
        fig = px.bar(
            cost_df, x="Strategy", y="Objective Cost", color="Strategy",
            text_auto=".4f", title="Objective Cost by Scheduling Strategy",
        )
        st.plotly_chart(fig, width="stretch")

        st.write(
            f"GA improvement over priority-based scheduling: "
            f"**{ga_metrics.get('ga_improvement_vs_priority_percent', 0):.2f}%**."
        )
        st.caption(
            "The objective cost is the prototype's risk-weighted scheduling objective, "
            "not a monetary amount. Capacity and changeover penalty are implementation assumptions."
        )

        st.subheader("Genetic Algorithm Configuration")
        c1, c2, c3 = st.columns(3)
        c1.metric("Population Size", ga_metrics.get("population_size", "N/A"))
        c2.metric("Generations", ga_metrics.get("generations", "N/A"))
        c3.metric("Daily Capacity", ga_metrics.get("daily_capacity", "N/A"))

        history = ga_metrics.get("best_cost_history")
        if isinstance(history, list) and history:
            st.subheader("GA Optimization Progress")
            history_df = pd.DataFrame({
                "Generation": range(1, len(history) + 1),
                "Best Objective Cost": history,
            })
            fig = px.line(history_df, x="Generation", y="Best Objective Cost", markers=False)
            st.plotly_chart(fig, width="stretch")

    if schedule is not None:
        st.subheader("Optimized Maintenance Schedule")
        st.dataframe(schedule, width="stretch", hide_index=True)

        if "scheduled_day" in schedule.columns:
            daily_counts = schedule.groupby("scheduled_day").size().reset_index(name="Machine Count")
            fig = px.bar(
                daily_counts, x="scheduled_day", y="Machine Count",
                text="Machine Count", title="Machines Scheduled Per Day",
            )
            st.plotly_chart(fig, width="stretch")

        st.download_button(
            "Download Maintenance Schedule",
            schedule.to_csv(index=False),
            file_name="maintenance_schedule.csv",
            mime="text/csv",
        )
    else:
        st.warning("GA schedule CSV not found.")


# ---------------- MACHINE MONITORING ----------------

elif page == "Machine Monitoring":
    st.header("Individual Machine Monitoring")
    selected_id = st.selectbox("Select Machine", machines["machine_id"].tolist())
    machine = machines[machines["machine_id"] == selected_id].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Failure Probability", f"{machine['failure_probability']:.2%}")
    c2.metric("Health Score", f"{machine['health_score']:.1f}/100")
    c3.metric("Urgency", f"{machine['urgency_score']:.1f}/100")
    c4.metric("Severity", machine["severity"])
    st.info(machine["recommendation"])

    history = raw_data[raw_data["machine_id"].astype(str) == selected_id].sort_values("timestamp")
    st.subheader("Sensor History")
    sensor_options = [
        c for c in ["temperature_motor", "vibration_rms", "current_phase_avg", "pressure_level", "rpm"]
        if c in history.columns
    ]
    if sensor_options:
        sensor = st.selectbox("Choose Sensor", sensor_options)
        history[sensor] = pd.to_numeric(history[sensor], errors="coerce")
        fig = px.line(history, x="timestamp", y=sensor, markers=True, title=f"{sensor} over time")
        st.plotly_chart(fig, width="stretch")

    st.subheader("Machine Details")
    st.dataframe(machine.to_frame("Value"), width="stretch")


# ---------------- EXPERIMENTAL RESULTS ----------------

elif page == "Experimental Results":
    st.header("Experimental Results & Discussion")

    st.subheader("Main Observations")
    st.markdown(
        """
        1. The original ANN produced the strongest overall metrics in the current experiments.
        2. In the four-input comparison, balanced ANFIS achieved higher recall, F1-score and ROC-AUC
           than the four-feature ANN in the saved test results.
        3. Balanced ANFIS achieved 90% recall, detecting 270 of the 300 positive test records.
        4. The GA schedule reduced the objective cost relative to FIFO and priority-based ordering.
        """
    )

    balanced = load_json("anfis_balanced_metrics.json")
    if balanced:
        tp = balanced.get("confusion_matrix", [[0, 0], [0, 0]])[1][1]
        fn = balanced.get("confusion_matrix", [[0, 0], [0, 0]])[1][0]
        fp = balanced.get("confusion_matrix", [[0, 0], [0, 0]])[0][1]
        st.write(
            f"Balanced ANFIS confusion matrix interpretation: **{tp}** failures detected, "
            f"**{fn}** failures missed, and **{fp}** false alarms on the test set."
        )

    st.subheader("Limitations to Acknowledge")
    st.markdown(
        """
        - The dataset covers a limited time period and a limited number of machines.
        - The evaluation uses a chronological train/test split; machines can occur in both partitions.
        - The original ANN and four-feature experiments use different input feature sets.
        - Oversampling is applied to training data only; the test data remains imbalanced and untouched.
        - GA capacity and changeover penalty are prototype assumptions, not validated factory operating costs.
        - High test metrics should be treated as preliminary until further validation on unseen machines or longer periods.
        """
    )

    st.subheader("Saved Experiment Files")
    file_rows = []
    for name, filename in MODEL_FILES.items():
        path = RESULTS_DIR / filename
        file_rows.append({"Experiment": name, "Metrics File": filename, "Available": path.exists()})
    for name, filename in PREDICTION_FILES.items():
        path = RESULTS_DIR / filename
        file_rows.append({"Experiment": f"{name} predictions", "Metrics File": filename, "Available": path.exists()})
    for filename in ["ga_metrics.json", "ga_maintenance_schedule.csv", "fuzzy_example.json", "preprocessing_summary.json"]:
        file_rows.append({"Experiment": filename, "Metrics File": filename, "Available": (RESULTS_DIR / filename).exists()})
    st.dataframe(pd.DataFrame(file_rows), width="stretch", hide_index=True)

# ---------------- FOOTER ----------------

st.divider()
st.caption(
    "Academic prototype | ANN / ANFIS + Mamdani Fuzzy Logic + Genetic Algorithm | "
    "Results shown are from the locally saved project files."
)
