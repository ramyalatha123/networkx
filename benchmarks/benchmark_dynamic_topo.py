"""
Benchmark: DynamicTopologicalSorter vs recomputing topological_sort() from scratch.

Scenario: Build a DAG by inserting edges one at a time.
- Baseline: after each insertion, call nx.topological_sort() on the whole graph.
- Dynamic: use DynamicTopologicalSorter.add_edge() which updates incrementally.

Run with:
    python benchmarks/benchmark_dynamic_topo.py
"""

import random
import time

import networkx as nx
from networkx.algorithms.dag import DynamicTopologicalSorter


def build_random_dag(n_nodes, n_edges, seed=42):
    """Return a list of (u, v) edges forming a DAG on nodes 0..n_nodes-1."""
    rng = random.Random(seed)
    nodes = list(range(n_nodes))
    edges = set()
    attempts = 0
    while len(edges) < n_edges and attempts < n_edges * 10:
        attempts += 1
        u, v = rng.sample(nodes, 2)
        if u < v:  # enforce u < v to guarantee no cycles
            edges.add((u, v))
    edges = sorted(edges)
    rng.shuffle(edges)
    return edges


def benchmark_baseline(edges, n_nodes):
    """Recompute topological_sort from scratch after every edge insertion."""
    G = nx.DiGraph()
    G.add_nodes_from(range(n_nodes))
    order = None
    for u, v in edges:
        G.add_edge(u, v)
        order = list(nx.topological_sort(G))
    return order


def benchmark_dynamic(edges, n_nodes):
    """Use DynamicTopologicalSorter — incremental updates only."""
    sorter = DynamicTopologicalSorter()
    for i in range(n_nodes):
        sorter.add_node(i)
    for u, v in edges:
        sorter.add_edge(u, v)
    return sorter.topological_order()


def run_benchmark(label, n_nodes, n_edges, seed=42):
    edges = build_random_dag(n_nodes, n_edges, seed)

    # Baseline
    start = time.perf_counter()
    baseline_order = benchmark_baseline(edges, n_nodes)
    baseline_time = time.perf_counter() - start

    # Dynamic
    start = time.perf_counter()
    dynamic_order = benchmark_dynamic(edges, n_nodes)
    dynamic_time = time.perf_counter() - start

    speedup = baseline_time / dynamic_time if dynamic_time > 0 else float("inf")

    print(
        f"{label:<35} | baseline: {baseline_time*1000:7.2f} ms"
        f" | dynamic: {dynamic_time*1000:7.2f} ms"
        f" | speedup: {speedup:.2f}x"
    )


def main():
    print()
    print("=" * 85)
    print("  DynamicTopologicalSorter vs topological_sort()-from-scratch benchmark")
    print("  Scenario: insert edges one at a time into a growing DAG")
    print("=" * 85)
    print(
        f"{'Configuration':<35} | {'Baseline (ms)':>15}"
        f" | {'Dynamic (ms)':>14} | Speedup"
    )
    print("-" * 85)

    configs = [
        ("Small  (100 nodes,   200 edges)", 100, 200),
        ("Small  (100 nodes,   500 edges)", 100, 500),
        ("Medium (500 nodes,  1000 edges)", 500, 1000),
        ("Medium (500 nodes,  3000 edges)", 500, 3000),
        ("Large  (1000 nodes, 2000 edges)", 1000, 2000),
        ("Large  (1000 nodes, 5000 edges)", 1000, 5000),
        ("XLarge (2000 nodes, 5000 edges)", 2000, 5000),
        ("XLarge (2000 nodes,10000 edges)", 2000, 10000),
    ]

    for label, n, e in configs:
        run_benchmark(label, n, e)

    print("=" * 85)
    print()
    print("Notes:")
    print("  - Baseline calls nx.topological_sort(G) after EVERY edge insertion.")
    print("  - Dynamic updates only the affected region of the position map.")
    print("  - Speedup grows with graph size because baseline cost is O(V+E) per")
    print("    insertion, while dynamic cost is O(k log k) where k << V+E in")
    print("    sparse graphs with locally ordered insertions.")
    print()


if __name__ == "__main__":
    main()
