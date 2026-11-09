
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from fuzzy_logic import assess_machine


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "dataset" / "raw" / "predictive_maintenance_v3.csv"
MODEL_PATH = ROOT / "models" / "ann_pipeline.joblib"
RESULTS_DIR = ROOT / "results"

# Prototype scheduling assumptions
CAPACITY_PER_DAY = 4
POPULATION_SIZE = 80
GENERATIONS = 150
MUTATION_RATE = 0.15
ELITE_COUNT = 4
RANDOM_SEED = 42

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


def load_machine_jobs():
    """Generate one current maintenance job per machine."""

    df = pd.read_csv(DATA_PATH)
    model = joblib.load(MODEL_PATH)

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "machine_id"])

    # Keep the latest available reading for each machine.
    latest = (
        df.sort_values("timestamp")
        .groupby("machine_id")
        .tail(1)
        .copy()
    )

    # Fill missing sensor readings using dataset medians.
    for col in NUMERIC_FEATURES:
        latest[col] = pd.to_numeric(latest[col], errors="coerce")
        median_value = pd.to_numeric(
            df[col], errors="coerce"
        ).median()
        latest[col] = latest[col].fillna(median_value)

    for col in CATEGORICAL_FEATURES:
        latest[col] = latest[col].fillna(
            df[col].mode().iloc[0]
        )

    # ANN failure probabilities.
    X = latest[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    probabilities = model.predict_proba(X)[:, 1]

    jobs = []

    for (_, row), probability in zip(latest.iterrows(), probabilities):
        fuzzy = assess_machine(
            temperature=float(row["temperature_motor"]),
            vibration=float(row["vibration_rms"]),
            failure_probability=float(probability),
            hours_since_maintenance=float(row["hours_since_maintenance"]),
        )

        priority = (
            0.60 * float(probability)
            + 0.40 * (float(fuzzy["urgency_score"]) / 100)
        )

        jobs.append({
            "machine_id": str(row["machine_id"]),
            "machine_type": str(row["machine_type"]),
            "failure_probability": float(probability),
            "urgency_score": float(fuzzy["urgency_score"]),
            "health_score": float(fuzzy["health_score"]),
            "severity": str(fuzzy["severity"]),
            "recommendation": str(fuzzy["recommendation"]),
            "priority": priority,
        })

    return jobs


def schedule_cost(order, jobs):
    """
    Lower cost is better.

    Cost includes:
    1. Risk-weighted waiting time.
    2. Machine-type changeover penalty within each day.
    """

    total_cost = 0.0

    for position, job_index in enumerate(order):
        day = position // CAPACITY_PER_DAY
        slot = position % CAPACITY_PER_DAY

        waiting_time = day + (slot / CAPACITY_PER_DAY)

        priority = jobs[job_index]["priority"]

        total_cost += priority * waiting_time

        # Assumed changeover penalty between different machine types.
        if slot > 0:
            previous_index = order[position - 1]

            current_type = jobs[job_index]["machine_type"]
            previous_type = jobs[previous_index]["machine_type"]

            if current_type != previous_type:
                total_cost += 0.03

    return total_cost


def tournament_selection(population, jobs, tournament_size=3):
    candidates = random.sample(
        population,
        min(tournament_size, len(population))
    )

    return min(
        candidates,
        key=lambda chromosome: schedule_cost(chromosome, jobs)
    )


def order_crossover(parent1, parent2):
    """Order crossover for permutation chromosomes."""

    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))

    child = [None] * size
    child[start:end + 1] = parent1[start:end + 1]

    remaining = [
        gene for gene in parent2
        if gene not in child
    ]

    empty_positions = [
        i for i, gene in enumerate(child)
        if gene is None
    ]

    for position, gene in zip(empty_positions, remaining):
        child[position] = gene

    return child


def swap_mutation(chromosome):
    """Swap two positions in a chromosome."""

    mutated = chromosome.copy()

    if len(mutated) >= 2:
        i, j = random.sample(range(len(mutated)), 2)
        mutated[i], mutated[j] = mutated[j], mutated[i]

    return mutated


def run_genetic_algorithm(jobs):
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    number_of_jobs = len(jobs)

    if number_of_jobs < 2:
        raise ValueError("At least two machines are required for GA.")

    # Initial population of random maintenance orders.
    population = []

    for _ in range(POPULATION_SIZE):
        chromosome = list(range(number_of_jobs))
        random.shuffle(chromosome)
        population.append(chromosome)

    best_cost_history = []

    for generation in range(GENERATIONS):

        population.sort(
            key=lambda chromosome: schedule_cost(chromosome, jobs)
        )

        best_cost_history.append(
            schedule_cost(population[0], jobs)
        )

        # Elitism: retain the best chromosomes.
        new_population = [
            chromosome.copy()
            for chromosome in population[:ELITE_COUNT]
        ]

        while len(new_population) < POPULATION_SIZE:

            parent1 = tournament_selection(population, jobs)
            parent2 = tournament_selection(population, jobs)

            child = order_crossover(parent1, parent2)

            if random.random() < MUTATION_RATE:
                child = swap_mutation(child)

            new_population.append(child)

        population = new_population

    population.sort(
        key=lambda chromosome: schedule_cost(chromosome, jobs)
    )

    best_order = population[0]

    return best_order, best_cost_history


def create_schedule(order, jobs):
    schedule = []

    for position, job_index in enumerate(order):
        job = jobs[job_index].copy()

        day_number = position // CAPACITY_PER_DAY + 1
        slot_number = position % CAPACITY_PER_DAY + 1

        job["scheduled_day"] = day_number
        job["daily_slot"] = slot_number

        schedule.append(job)

    return pd.DataFrame(schedule)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n========== GENETIC ALGORITHM ==========")

    jobs = load_machine_jobs()

    print(f"\nMachines considered: {len(jobs)}")
    print(f"Daily maintenance capacity: {CAPACITY_PER_DAY}")

    # Baseline 1: original machine order.
    fifo_order = list(range(len(jobs)))

    # Baseline 2: descending combined ANN/fuzzy priority.
    priority_order = sorted(
        range(len(jobs)),
        key=lambda i: jobs[i]["priority"],
        reverse=True
    )

    fifo_cost = schedule_cost(fifo_order, jobs)
    priority_cost = schedule_cost(priority_order, jobs)

    print("\nRunning Genetic Algorithm...")
    best_order, history = run_genetic_algorithm(jobs)

    ga_cost = schedule_cost(best_order, jobs)

    schedule = create_schedule(best_order, jobs)

    output_path = RESULTS_DIR / "ga_maintenance_schedule.csv"
    schedule.to_csv(output_path, index=False)

    metrics = {
        "machines_scheduled": len(jobs),
        "daily_capacity": CAPACITY_PER_DAY,
        "generations": GENERATIONS,
        "population_size": POPULATION_SIZE,
        "fifo_cost": float(fifo_cost),
        "priority_order_cost": float(priority_cost),
        "ga_cost": float(ga_cost),
        "ga_improvement_vs_fifo_percent": float(
            ((fifo_cost - ga_cost) / fifo_cost) * 100
            if fifo_cost else 0
        ),
        "ga_improvement_vs_priority_percent": float(
            ((priority_cost - ga_cost) / priority_cost) * 100
            if priority_cost else 0
        ),
        "best_cost_history": history,
    }

    metrics_path = RESULTS_DIR / "ga_metrics.json"

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print("\n========== GA RESULTS ==========")
    print(f"FIFO baseline cost:     {fifo_cost:.4f}")
    print(f"Priority baseline cost: {priority_cost:.4f}")
    print(f"GA optimized cost:      {ga_cost:.4f}")

    print("\nOptimized Maintenance Schedule:")
    print(
        schedule[
            [
                "machine_id",
                "machine_type",
                "failure_probability",
                "urgency_score",
                "severity",
                "scheduled_day",
                "daily_slot",
            ]
        ].to_string(index=False)
    )

    print("\n========== FILES SAVED ==========")
    print(f"Schedule: {output_path}")
    print(f"Metrics:  {metrics_path}")


if __name__ == "__main__":
    main()
