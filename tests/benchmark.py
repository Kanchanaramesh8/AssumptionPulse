import random
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib.pyplot as plt

from core.graph_engine import ADIEGraphEngine
from core.models import Assumption, Decision, DependencyEdge
from core.validator import HybridAssumptionValidator


def build_synthetic_dag(num_assumptions: int, num_decisions: int, seed: int = 42) -> ADIEGraphEngine:
    """Generates a realistic multi-hop layered DAG for scalability benchmarking."""
    rng = random.Random(seed)
    engine = ADIEGraphEngine(project_name=f"Synthetic-{num_assumptions + num_decisions}")

    for i in range(num_assumptions):
        engine.add_assumption(
            Assumption(id=f"A{i}", statement=f"Synthetic assumption {i} holds under load {i * 10}")
        )

    for j in range(num_decisions):
        engine.add_decision(
            Decision(
                id=f"D{j}",
                title=f"Decision {j}",
                description=f"Architectural decision {j}",
                criticality=round(rng.uniform(0.5, 1.0), 2),
            )
        )
        # Connect to 1-2 parent assumptions
        for a_idx in rng.sample(range(num_assumptions), k=min(2, num_assumptions)):
            engine.add_dependency(
                DependencyEdge(
                    source_id=f"A{a_idx}",
                    target_id=f"D{j}",
                    weight=round(rng.uniform(0.4, 0.95), 2),
                )
            )
        # Connect to earlier decisions to create multi-hop chains (DAG guaranteed since prev < j)
        if j > 0:
            prev_d = rng.randint(0, j - 1)
            engine.add_dependency(
                DependencyEdge(
                    source_id=f"D{prev_d}",
                    target_id=f"D{j}",
                    weight=round(rng.uniform(0.5, 0.9), 2),
                )
            )
    return engine


def run_scalability_benchmark():
    print("\n--- 1. ADIE Multi-Hop Propagation Scalability Benchmark ---")
    configs = [(5, 15), (15, 35), (25, 75), (50, 200), (100, 400)]
    node_counts = []
    latencies_ms = []

    for n_a, n_d in configs:
        engine = build_synthetic_dag(n_a, n_d)
        total_nodes = n_a + n_d
        total_edges = engine.graph.number_of_edges()

        # Average over 20 propagation runs
        start = time.perf_counter()
        for _ in range(20):
            engine.invalidate_assumption("A0", new_confidence=0.0)
            engine.invalidate_assumption("A0", new_confidence=1.0)
        elapsed_ms = ((time.perf_counter() - start) / 40.0) * 1000.0

        node_counts.append(total_nodes)
        latencies_ms.append(round(elapsed_ms, 4))
        print(f"Nodes: {total_nodes:3d} | Edges: {total_edges:4d} | Avg Propagation Latency: {elapsed_ms:.4f} ms")

    # Save chart for project report
    chart_path = ROOT_DIR / "data" / "benchmark_results.png"
    plt.figure(figsize=(7, 4))
    plt.plot(node_counts, latencies_ms, marker="o", linewidth=2, color="#2563eb")
    plt.title("ADIE Multi-Hop Impact Propagation Latency vs. Graph Size")
    plt.xlabel("Total Graph Nodes (Assumptions + Decisions)")
    plt.ylabel("Mean Propagation Time (ms)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(chart_path, dpi=150)
    print(f"Saved scalability chart to: {chart_path}")


def run_validator_accuracy_benchmark():
    print("\n--- 2. Hybrid Validator Contradiction & Drift Detection Benchmark ---")
    validator = HybridAssumptionValidator()

    test_cases = [
        # (Stored Assumption, Incoming Fact, Expected_Flagged: bool)
        ("SMS OTP costs 0.25 INR per message", "Vendor raised SMS OTP pricing to 0.85 INR", True),
        ("API rate limit is 1000 requests per minute", "Provider reduced API rate limit to 100 requests per minute", True),
        ("Redis cache stays under 512 MB RAM", "Redis memory usage exceeded 512 MB limit", True),
        ("User authentication uses SHA-256 hashing", "Security policy deprecated SHA-256 hashing", True),
        ("Database backup runs every 24 hours", "Automated database backup stopped working", True),
        ("SMS OTP costs 0.25 INR per message", "Frontend button color updated to dark blue", False),
        ("Redis cache stays under 512 MB RAM", "Added new user profile avatar upload feature", False),
        ("Transactional Email API is free up to 10000 emails", "Updated terms of service footer link", False),
    ]

    tp = fp = tn = fn = 0
    for idx, (stmt, fact, expected) in enumerate(test_cases):
        g = ADIEGraphEngine("Eval")
        g.add_assumption(Assumption(id=f"A{idx}", statement=stmt))
        alerts = validator.evaluate_incoming_fact(g, fact, auto_propagate=False)
        predicted = len(alerts) > 0

        if predicted and expected:
            tp += 1
        elif predicted and not expected:
            fp += 1
        elif not predicted and not expected:
            tn += 1
        else:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    print(f"TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    print(f"Precision: {precision:.2%} | Recall: {recall:.2%} | F1-Score: {f1:.2%}")


if __name__ == "__main__":
    run_scalability_benchmark()
    run_validator_accuracy_benchmark()