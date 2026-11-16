# AssumptionPulse — Assumption-to-Decision Impact Engine (ADIE)

**AssumptionPulse** is a locally executed, neuro-symbolic decision-validity monitoring and multi-hop risk propagation engine. Instead of letting Architecture Decision Records (ADRs) rot silently when real-world costs, regulations, or operational assumptions shift, AssumptionPulse models foundational assumptions as weighted Directed Acyclic Graphs (DAGs) and flags downstream architectural risks in sub-millisecond time.

---

## Core Architecture & Capabilities

* **Weighted Dependency DAG (`core/graph_engine.py`):** Models foundational financial, legal, and technical assumptions ($A_i$) linked to downstream architectural decisions ($D_j$) with configurable dependency weights ($w \in [0,1]$) and criticality scores using `NetworkX`.
* **$O(V + E)$ Topological Risk Propagation:** Sweeps affected subgraphs in topological order whenever an assumption degrades or expires, calculating exact cumulative risk scores, System Blast Radius Scores, and full causal paths ($A_3 \rightarrow D_1$).
* **Neuro-Symbolic Drift Validator (`core/validator.py`):** Pairs local semantic embeddings (`all-MiniLM-L6-v2` via `sentence-transformers`) with deterministic symbolic rules—regex numeric-drift extraction, negation scanning, and ISO-8601 temporal expiry auditing—achieving **100% precision** and an **88.89% F1-score** with zero external LLM API calls.
* **Counterfactual "What-If" Simulator:** Evaluates candidate replacement decisions on an isolated graph copy (`copy.deepcopy`) to quantify blast-radius reduction before committing changes to production.
* **Interactive Command-Center Dashboard (`ui/app.py`):** Dark-mode Streamlit interface for live assumption stress-testing, incoming fact injection, temporal audits, and counterfactual decision modeling.

---

## Empirical Performance Benchmarks

Benchmarked across synthetic DAGs ranging from **20 to 500 nodes** (up to **1,699 total graph elements $V + E$**):

* **500-Node / 1,199-Edge Propagation Latency:** `1.4928 ms`
* **Scaling Complexity:** Verified linear $O(V + E)$ growth

![Propagation Latency Benchmark](data/benchmark_results.png)

---

## Project Structure

```text
AssumptionPulse/
├── .streamlit/
│   └── config.toml             # Dark command-center theme configuration
├── core/
│   ├── models.py               # Pydantic v2 schemas (Assumption, Decision, DependencyEdge)
│   ├── graph_engine.py         # NetworkX DAG engine, O(V+E) propagation & What-If simulator
│   └── validator.py            # Neuro-symbolic drift detector & temporal expiry auditor
├── data/
│   ├── sample_scenario.json    # FinPay Authentication & Notification Architecture dataset
│   └── benchmark_results.png   # O(V+E) latency scaling benchmark chart
├── tests/
│   └── benchmark.py            # Synthetic scalability & latency benchmark runner
├── ui/
│   └── app.py                  # Streamlit interactive dashboard
└── requirements.txt