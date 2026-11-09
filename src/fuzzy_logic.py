
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
    if x <= a:
        return 1.0
    if x >= b:
        return 0.0
    return (b - x) / (b - a)


def right_shoulder(x, a, b):
    if x <= a:
        return 0.0
    if x >= b:
        return 1.0
    return (x - a) / (b - a)


# ==========================================
# 3. INPUT FUZZIFICATION
# ==========================================

def fuzzify_temperature(value):

    return {
        "low": trapezoidal(value, 0, 0, 40, 55),
        "medium": triangular(value, 45, 60, 75),
        "high": trapezoidal(value, 65, 80, 120, 120)
    }


def fuzzify_vibration(value):

    return {
        "low": trapezoidal(value, 0, 0, 0.8, 1.5),
        "medium": triangular(value, 1.0, 1.9, 2.8),
        "high": trapezoidal(value, 2.3, 3.5, 5, 5)
    }


def fuzzify_risk(value):

    return {
        "low": trapezoidal(value, 0, 0, 0.2, 0.4),
        "medium": triangular(value, 0.25, 0.5, 0.75),
        "high": trapezoidal(value, 0.6, 0.8, 1, 1)
    }


def fuzzify_maintenance(value):

    return {
        "low": trapezoidal(value, 0, 0, 100, 160),
        "medium": triangular(value, 100, 220, 340),
        "high": trapezoidal(value, 280, 380, 600, 600)
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

    # High predicted failure risk
    (("risk", "high"), "high"),

    # High sensor readings and high risk
    (
        ("temperature", "high"),
        ("vibration", "high"),
        ("risk", "high"),
        "critical"
    ),

    # High temperature and vibration
    (
        ("temperature", "high"),
        ("vibration", "high"),
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

    # Old maintenance with medium risk
    (
        ("maintenance", "high"),
        ("risk", "medium"),
        "high"
    ),

    # Old maintenance and high risk
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

    # Low risk and healthy sensor readings
    (
        ("temperature", "low"),
        ("vibration", "low"),
        ("risk", "low"),
        "low"
    ),

    # Low risk with moderate sensor readings
    (
        ("risk", "low"),
        ("temperature", "medium"),
        "medium"
    ),

    (
        ("risk", "low"),
        ("vibration", "medium"),
        "medium"
    )
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

    values = {
        "temperature": temperature,
        "vibration": vibration,
        "risk": failure_probability,
        "maintenance": hours_since_maintenance
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
        "vibration": fuzzify_vibration(vibration),
        "risk": fuzzify_risk(failure_probability),
        "maintenance": fuzzify_maintenance(
            hours_since_maintenance
        )
    }

    # Output universe
    universe = np.linspace(0, 100, 1001)

    output_sets = output_memberships(universe)

    # Initialize aggregated output
    aggregated = np.zeros(len(universe))

    # Apply Mamdani rules
    for rule in RULES:

        *antecedents, consequent = rule

        # AND operator = minimum
        strength = min(
            fuzzy_inputs[input_name][membership]
            for input_name, membership in antecedents
        )

        # Implication = minimum
        clipped_output = np.minimum(
            strength,
            output_sets[consequent]
        )

        # Aggregation = maximum
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

    health_score = 100 - urgency

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
        "recommendation": recommendation
    }


# ==========================================
# 7. TEST THE FUZZY ENGINE
# ==========================================

def main():

    print("\n========== FUZZY LOGIC TEST ==========\n")

    test_machine = {
        "temperature": 75,
        "vibration": 3.2,
        "failure_probability": 0.85,
        "hours_since_maintenance": 300
    }

    result = assess_machine(
        temperature=test_machine["temperature"],
        vibration=test_machine["vibration"],
        failure_probability=test_machine["failure_probability"],
        hours_since_maintenance=test_machine[
            "hours_since_maintenance"
        ]
    )

    print("Test Machine Inputs:")
    print(test_machine)

    print("\nFuzzy Logic Results:")

    for key, value in result.items():
        print(f"{key}: {value}")

    output = {
        "inputs": test_machine,
        "fuzzy_assessment": result
    }

    with open(
        RESULTS_DIR / "fuzzy_example.json",
        "w"
    ) as file:
        json.dump(output, file, indent=4)

    print("\nSaved: results/fuzzy_example.json")


if __name__ == "__main__":
    main()
