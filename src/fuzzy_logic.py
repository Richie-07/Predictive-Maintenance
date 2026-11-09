
import json
import numpy as np
from pathlib import Path


# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ==========================================
# 2. MEMBERSHIP FUNCTIONS
# ==========================================

def triangular(x, a, b, c):
    """Triangular membership function."""

    if x <= a or x >= c:
        return 0.0

    if x == b:
        return 1.0

    if x < b:
        return (x - a) / (b - a)

    return (c - x) / (c - b)


def trapezoidal(x, a, b, c, d):
    """Trapezoidal membership function."""

    if x <= a or x >= d:
        return 0.0

    if b <= x <= c:
        return 1.0

    if a < x < b:
        return (x - a) / (b - a)

    return (d - x) / (d - c)


def left_shoulder(x, a, b):
    """Left-shoulder (Z-shaped) membership function.
    Returns 1 for x <= a, 0 for x >= b, linear in between.
    """
    if x <= a:
        return 1.0
    if x >= b:
        return 0.0
    return (b - x) / (b - a)


def right_shoulder(x, a, b):
    """Right-shoulder (S-shaped) membership function.
    Returns 0 for x <= a, 1 for x >= b, linear in between.
    """
    if x <= a:
        return 0.0
    if x >= b:
        return 1.0
    return (x - a) / (b - a)


# ==========================================
# 3. INPUT FUZZIFICATION
# ==========================================

def fuzzify_temperature(value):
    """
    Fuzzify motor temperature (°C).

    FIX (audit C5/H5): Closed the dead zone that existed between 55–65 °C
    in the original version.  The low / medium / high sets now provide
    full universe coverage:
      Low    : left-shoulder  0 → 50 °C  (full membership ≤ 40, zero ≥ 55)
      Medium : triangular     40–55–75 °C (peak at 60 °C)
      High   : right-shoulder 65 → 80 °C (full membership ≥ 80)
    """
    return {
        "low":    left_shoulder(value, 40, 55),
        "medium": triangular(value, 40, 60, 80),
        "high":   right_shoulder(value, 65, 80),
    }


def fuzzify_vibration(value):
    """
    Fuzzify vibration RMS.

    FIX (audit H6): The original trapezoidal "high" set was capped at 5.0,
    so values > 5.0 received zero membership in every set.  Using a
    right-shoulder MF removes the artificial upper cap.
    """
    return {
        "low":    left_shoulder(value, 0.8, 1.5),
        "medium": triangular(value, 1.0, 1.9, 3.0),
        "high":   right_shoulder(value, 2.3, 3.5),
    }


def fuzzify_risk(value):
    """Fuzzify ANN failure probability (0–1)."""
    return {
        "low":    trapezoidal(value, 0, 0, 0.2, 0.4),
        "medium": triangular(value, 0.25, 0.5, 0.75),
        "high":   trapezoidal(value, 0.6, 0.8, 1, 1),
    }


def fuzzify_maintenance(value):
    """
    Fuzzify hours since last maintenance.

    FIX (audit M3): The original "high" set was capped at 600 h via a
    trapezoidal function, so machines overdue beyond 600 h received zero
    membership.  Using a right-shoulder MF ensures any value ≥ 380 h
    receives full "high" membership.
    """
    return {
        "low":    left_shoulder(value, 100, 160),
        "medium": triangular(value, 100, 220, 340),
        "high":   right_shoulder(value, 280, 380),
    }


# ==========================================
# 4. OUTPUT MEMBERSHIP FUNCTIONS
# ==========================================

def output_memberships(universe):

    low = np.array([
        trapezoidal(x, 0, 0, 15, 35)
        for x in universe
    ])

    medium = np.array([
        triangular(x, 25, 45, 65)
        for x in universe
    ])

    high = np.array([
        triangular(x, 55, 72, 88)
        for x in universe
    ])

    critical = np.array([
        trapezoidal(x, 80, 92, 100, 100)
        for x in universe
    ])

    return {
        "low": low,
        "medium": medium,
        "high": high,
        "critical": critical
    }


# ==========================================
# 5. MAMDANI FUZZY RULE BASE
# ==========================================

RULES = [

    # High predicted failure risk alone → high urgency
    (("risk", "high"), "high"),

    # High sensor readings and high risk → critical
    (
        ("temperature", "high"),
        ("vibration", "high"),
        ("risk", "high"),
        "critical"
    ),

    # High temperature and vibration (physical danger — ANN-independent)
    (
        ("temperature", "high"),
        ("vibration", "high"),
        "high"
    ),

    # FIX (audit M5): Physical danger override — high temperature even when
    # ANN probability is low.  Ensures the fuzzy layer can catch cases where
    # the ANN misses a thermally dangerous state.
    (
        ("temperature", "high"),
        ("risk", "low"),
        "high"
    ),

    # High temperature and medium risk
    (
        ("temperature", "high"),
        ("risk", "medium"),
        "high"
    ),

    # High vibration and medium risk
    (
        ("vibration", "high"),
        ("risk", "medium"),
        "high"
    ),

    # Overdue maintenance with medium risk
    (
        ("maintenance", "high"),
        ("risk", "medium"),
        "high"
    ),

    # Overdue maintenance and high risk → critical
    (
        ("maintenance", "high"),
        ("risk", "high"),
        "critical"
    ),

    # Medium risk and medium maintenance age
    (
        ("risk", "medium"),
        ("maintenance", "medium"),
        "medium"
    ),

    # Healthy readings and low risk → low urgency
    (
        ("temperature", "low"),
        ("vibration", "low"),
        ("risk", "low"),
        "low"
    ),

    # Low risk with moderate sensor readings → medium (watchful)
    (
        ("risk", "low"),
        ("temperature", "medium"),
        "medium"
    ),

    (
        ("risk", "low"),
        ("vibration", "medium"),
        "medium"
    ),
]


# ==========================================
# 6. FUZZY INFERENCE SYSTEM
# ==========================================

def assess_machine(
    temperature,
    vibration,
    failure_probability,
    hours_since_maintenance
):
    """
    Mamdani fuzzy assessment of machine maintenance urgency.

    Parameters
    ----------
    temperature           : float  Motor temperature in °C (≥ 0)
    vibration             : float  Vibration RMS (≥ 0)
    failure_probability   : float  ANN predicted failure probability [0, 1]
    hours_since_maintenance : float Hours elapsed since last maintenance (≥ 0)

    Returns
    -------
    dict with keys:
        urgency_score   (float, 0–100)
        health_score    (float, 0–100)
        severity        (str)
        recommendation  (str)
    """

    values = {
        "temperature": temperature,
        "vibration": vibration,
        "risk": failure_probability,
        "maintenance": hours_since_maintenance,
    }

    if not all(np.isfinite(float(v)) for v in values.values()):
        raise ValueError("All input values must be finite numbers.")

    if not 0 <= failure_probability <= 1:
        raise ValueError("Failure probability must be between 0 and 1.")

    if temperature < 0 or vibration < 0 or hours_since_maintenance < 0:
        raise ValueError("Sensor values and maintenance age cannot be negative.")

    # Fuzzification
    fuzzy_inputs = {
        "temperature": fuzzify_temperature(temperature),
        "vibration":   fuzzify_vibration(vibration),
        "risk":        fuzzify_risk(failure_probability),
        "maintenance": fuzzify_maintenance(hours_since_maintenance),
    }

    # Output universe: 1001 evenly spaced points from 0 to 100
    universe = np.linspace(0, 100, 1001)
    output_sets = output_memberships(universe)

    # Initialize aggregated output
    aggregated = np.zeros(len(universe))

    # Apply Mamdani rules
    for rule in RULES:

        *antecedents, consequent = rule

        # AND operator = minimum of all antecedent memberships
        strength = min(
            fuzzy_inputs[input_name][membership]
            for input_name, membership in antecedents
        )

        # Implication = minimum (Mamdani clipping)
        clipped_output = np.minimum(
            strength,
            output_sets[consequent]
        )

        # Aggregation = maximum (union)
        aggregated = np.maximum(
            aggregated,
            clipped_output
        )

    # Centroid defuzzification
    area = np.sum(aggregated)

    if area == 0:
        urgency = 0.0
    else:
        urgency = float(
            np.sum(universe * aggregated) / area
        )

    health_score = 100.0 - urgency

    # Severity classification
    if urgency < 25:
        severity = "NORMAL"
        recommendation = "Continue normal machine operation."

    elif urgency < 45:
        severity = "LOW"
        recommendation = "Monitor machine condition."

    elif urgency < 65:
        severity = "MEDIUM"
        recommendation = "Schedule preventive maintenance."

    elif urgency < 85:
        severity = "HIGH"
        recommendation = "Arrange maintenance soon."

    else:
        severity = "CRITICAL"
        recommendation = "Prioritize immediate maintenance inspection."

    return {
        "urgency_score": round(urgency, 2),
        "health_score": round(health_score, 2),
        "severity": severity,
        "recommendation": recommendation,
    }


# ==========================================
# 7. TEST THE FUZZY ENGINE
# ==========================================

def main():

    print("\n========== FUZZY LOGIC TEST ==========\n")

    test_cases = [
        {
            "label": "Normal operation",
            "temperature": 45, "vibration": 0.5,
            "failure_probability": 0.05, "hours_since_maintenance": 50,
        },
        {
            "label": "Elevated temperature",
            "temperature": 75, "vibration": 1.0,
            "failure_probability": 0.3, "hours_since_maintenance": 150,
        },
        {
            "label": "High vibration",
            "temperature": 50, "vibration": 4.0,
            "failure_probability": 0.4, "hours_since_maintenance": 200,
        },
        {
            "label": "High ANN failure probability",
            "temperature": 75, "vibration": 3.2,
            "failure_probability": 0.85, "hours_since_maintenance": 300,
        },
        {
            "label": "Long since maintenance",
            "temperature": 50, "vibration": 1.5,
            "failure_probability": 0.3, "hours_since_maintenance": 500,
        },
        {
            "label": "Multiple abnormal readings",
            "temperature": 90, "vibration": 5.0,
            "failure_probability": 0.9, "hours_since_maintenance": 600,
        },
        {
            "label": "High temperature but low ANN probability (ANN blind spot test)",
            "temperature": 85, "vibration": 1.0,
            "failure_probability": 0.05, "hours_since_maintenance": 50,
        },
        {
            "label": "Vibration above old 5.0 cap (sensor > 5)",
            "temperature": 50, "vibration": 7.5,
            "failure_probability": 0.6, "hours_since_maintenance": 200,
        },
        {
            "label": "Maintenance > 600 h (old cap test)",
            "temperature": 50, "vibration": 1.0,
            "failure_probability": 0.3, "hours_since_maintenance": 800,
        },
    ]

    all_results = []

    for case in test_cases:
        label = case.pop("label")
        result = assess_machine(**case)
        print(f"[{label}]")
        print(f"  Inputs : {case}")
        print(f"  Urgency: {result['urgency_score']:.2f}  "
              f"Health: {result['health_score']:.2f}  "
              f"Severity: {result['severity']}")
        print(f"  → {result['recommendation']}\n")
        all_results.append({"label": label, "inputs": case, "result": result})

    # Save one representative example (high ANN probability case)
    example = all_results[3]
    output = {
        "inputs": example["inputs"],
        "fuzzy_assessment": example["result"],
    }

    with open(
        RESULTS_DIR / "fuzzy_example.json",
        "w"
    ) as file:
        json.dump(output, file, indent=4)

    print("Saved: results/fuzzy_example.json")


if __name__ == "__main__":
    main()
