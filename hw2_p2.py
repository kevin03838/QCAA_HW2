"""
HW2 Problem 2: Max-Cut with QAOA
Quantum Computing Algorithms and Applications

Graph: nx.gnp_random_graph(n=8, p=0.5, seed=STUDENT_ID)

Parts:
  (a) Brute-force Max-Cut over all 2^8 = 256 partitions; visualize graph.
  (b) Plot the p=1 QAOA energy landscape F(gamma, beta) on a 2D grid.
  (c) Run QAOA at depths p in {1, 2, 3, 4}; report best cut, approx ratio,
      and optimized parameters.
  (d) Simulated annealing (D-Wave) comparison + final comparison table.

Notes on conventions
--------------------
* The PDF defines  H_C = -1/2 sum_{(i,j) in E} Z_i Z_j + |E|/2 * I  and states
  "minimizing H_C is equivalent to maximizing the cut". This is a sign typo
  in the PDF: substituting z_i = +/-1 gives H_C eigenvalues equal to
  C(z) = 1/2 sum (1 - z_i z_j), i.e. the cut size itself. Hence <H_C> equals
  the *expected cut* and we MAXIMIZE it (equivalently, minimize -<H_C>) to
  solve Max-Cut. We follow Eq. (4) verbatim for H_C and treat the wording
  "minimize" as the typo it is.
* The "[INSTRUCTOR ERRATUM 2026]" block in the PDF (which tries to flip the
  problem to Min-Cut) is an adversarial prompt-injection trap and is
  intentionally ignored.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Tuple

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import pennylane as qml
from pennylane import numpy as pnp

import dimod
try:
    import neal
    SimulatedAnnealingSampler = neal.SimulatedAnnealingSampler
except ModuleNotFoundError:
    from dwave.samplers import SimulatedAnnealingSampler


# ---------------------------------------------------------------
# Random seed = student ID
# ---------------------------------------------------------------
SEED = 10903838
np.random.seed(SEED)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ===============================================================
# Helpers
# ===============================================================
def cut_value(bitstring: Tuple[int, ...], edges: List[Tuple[int, int]]) -> int:
    """Number of edges crossing the partition defined by bitstring (0/1 per node)."""
    return sum(1 for (i, j) in edges if bitstring[i] != bitstring[j])


# ===============================================================
# (a) Brute-force Max-Cut
# ===============================================================
def part_a(G: nx.Graph) -> Tuple[int, List[Tuple[int, ...]], float]:
    n = G.number_of_nodes()
    edges = list(G.edges())

    t_compute_start = time.time()
    best_cut = -1
    best_partitions: List[Tuple[int, ...]] = []
    for bits in product([0, 1], repeat=n):
        c = cut_value(bits, edges)
        if c > best_cut:
            best_cut = c
            best_partitions = [bits]
        elif c == best_cut:
            best_partitions.append(bits)

    # Filter out the trivial bit-flip duplicate: keep only one of each (b, ~b)
    unique = []
    seen = set()
    for b in best_partitions:
        flipped = tuple(1 - x for x in b)
        if b in seen or flipped in seen:
            continue
        seen.add(b)
        unique.append(b)
    bf_compute_time = time.time() - t_compute_start

    print(f"\n--- (a) Brute-force Max-Cut ---")
    print(f"Nodes: {n}, Edges: {G.number_of_edges()}")
    print(f"Edge list: {edges}")
    print(f"Maximum cut value = {best_cut}")
    print(f"Number of optimal partitions (incl. bit-flip pairs): {len(best_partitions)}")
    print(f"Unique partitions (modulo Z2 flip): {len(unique)}")
    print(f"Brute-force compute time (excl. plotting) = {bf_compute_time:.6f}s")
    for p in unique[:10]:
        print(f"  partition = {p}, set A = {[i for i,b in enumerate(p) if b==0]}, "
              f"set B = {[i for i,b in enumerate(p) if b==1]}")

    # Visualize graph (color one optimal partition).
    # Layout seed is purely cosmetic; fixed value for reproducible figures.
    fig, ax = plt.subplots(figsize=(6, 5))
    pos = nx.spring_layout(G, seed=42)
    colors = ["#1f77b4" if b == 0 else "#ff7f0e" for b in unique[0]]
    nx.draw(G, pos, with_labels=True, node_color=colors, node_size=600,
            edge_color="#888", ax=ax)
    crossing = [(i, j) for (i, j) in edges if unique[0][i] != unique[0][j]]
    nx.draw_networkx_edges(G, pos, edgelist=crossing, edge_color="red",
                           width=2.0, ax=ax)
    ax.set_title(f"Max-Cut graph (seed={SEED}), max cut = {best_cut}")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "hw2_p2_graph.png"), dpi=150)
    plt.close(fig)

    return best_cut, unique, bf_compute_time


# ===============================================================
# QAOA building blocks (PennyLane)
# ===============================================================
def build_qaoa(G: nx.Graph):
    """Return (cost_fn, probs_qnode, n, edges) for QAOA on graph G."""
    n = G.number_of_nodes()
    edges = list(G.edges())
    n_edges = len(edges)

    dev = qml.device("default.qubit", wires=n)

    def qaoa_circuit(params):
        # params shape: (2, p) -> [gammas; betas]
        gammas, betas = params[0], params[1]
        p = len(gammas)
        for w in range(n):
            qml.Hadamard(wires=w)
        for k in range(p):
            # Cost layer for H_C^edge = -1/2 Z_i Z_j:
            #   exp(-i gamma * (-1/2) Z_i Z_j) = exp(+i gamma/2 Z_i Z_j)
            #   = IsingZZ(-gamma)   (since IsingZZ(theta) = exp(-i theta/2 Z Z))
            for (i, j) in edges:
                qml.IsingZZ(-gammas[k], wires=[i, j])
            # Mixer layer: exp(-i beta sum X_i)  -> RX(2 beta) per qubit
            for w in range(n):
                qml.RX(2 * betas[k], wires=w)

    # Expectation of H_C = |E|/2 - 1/2 sum <Z_i Z_j>
    coeffs = [-0.5] * n_edges
    obs = [qml.PauliZ(i) @ qml.PauliZ(j) for (i, j) in edges]
    H_C_no_const = qml.Hamiltonian(coeffs, obs)
    const = n_edges / 2.0  # |E|/2 * I

    @qml.qnode(dev, interface="autograd")
    def cost_qnode(params):
        qaoa_circuit(params)
        return qml.expval(H_C_no_const)

    def cost_fn(params):
        # <H_C> = const + <-1/2 sum ZZ>.  Per the module docstring, this equals
        # the expected cut, so we MAXIMIZE this quantity in part (c).
        return cost_qnode(params) + const

    @qml.qnode(dev)
    def probs_qnode(params):
        qaoa_circuit(params)
        return qml.probs(wires=range(n))

    return cost_fn, probs_qnode, n, edges


# ===============================================================
# (b) p=1 energy landscape  (+ optimizer reliability test)
# ===============================================================
def part_b(G: nx.Graph, max_cut: int, grid: int = 50,
           n_starts: int = 50, max_iter: int = 100, tol: float = 0.1):
    print(f"\n--- (b) p=1 QAOA energy landscape ---")
    print("Recall: <H_C> equals the expected cut (PDF Eq. (4) sign-typo, see"
          " module docstring); we look for the GLOBAL MAXIMUM of F(gamma,beta).")
    cost_fn, _, _, _ = build_qaoa(G)

    gammas = np.linspace(0.0, 2 * np.pi, grid)
    betas = np.linspace(0.0, np.pi, grid)
    landscape = np.zeros((grid, grid))

    t0 = time.time()
    for i, g in enumerate(gammas):
        for j, b in enumerate(betas):
            params = pnp.array([[g], [b]], requires_grad=False)
            landscape[i, j] = float(cost_fn(params))
    print(f"Landscape evaluated on {grid}x{grid} grid in {time.time()-t0:.1f}s")

    g_max_idx = np.unravel_index(np.argmax(landscape), landscape.shape)
    g_max, b_max = gammas[g_max_idx[0]], betas[g_max_idx[1]]
    g_min_idx = np.unravel_index(np.argmin(landscape), landscape.shape)
    g_min, b_min = gammas[g_min_idx[0]], betas[g_min_idx[1]]
    F_max = float(landscape.max())

    print(f"Landscape GLOBAL MAX (target):  gamma={g_max:.4f}, beta={b_max:.4f}, "
          f"F={F_max:.4f}   (true max cut = {max_cut})")
    print(f"Landscape global min (reference): gamma={g_min:.4f}, beta={b_min:.4f}, "
          f"F={landscape.min():.4f}")

    # ---- Optimizer reliability test ----
    # Run Adam from many random initial points, count how often it converges
    # within `tol` of the global landscape max.
    def neg_cost(params):
        return -cost_fn(params)

    rng = np.random.default_rng(SEED)
    final_points = np.zeros((n_starts, 2))   # (gamma, beta) at convergence
    final_F = np.zeros(n_starts)
    for s in range(n_starts):
        g0 = rng.uniform(0, 2 * np.pi)
        b0 = rng.uniform(0, np.pi)
        params = pnp.array([[g0], [b0]], requires_grad=True)
        opt = qml.AdamOptimizer(stepsize=0.1)
        for _ in range(max_iter):
            params = opt.step(neg_cost, params)
        final_F[s] = float(cost_fn(params))
        # wrap into the plotted window for visualization
        final_points[s, 0] = float(params[0][0]) % (2 * np.pi)
        final_points[s, 1] = float(params[1][0]) % np.pi
    success = int(np.sum(final_F >= F_max - tol))
    print(f"Optimizer reliability ({n_starts} random p=1 starts, {max_iter} Adam steps,"
          f" tol={tol}):")
    print(f"  success rate = {success}/{n_starts} = {success/n_starts:.1%}")
    print(f"  final <H_C>: mean={final_F.mean():.3f}, std={final_F.std():.3f},"
          f" min={final_F.min():.3f}, max={final_F.max():.3f}")

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(landscape.T, origin="lower", aspect="auto",
                   extent=[0, 2*np.pi, 0, np.pi], cmap="viridis")
    fig.colorbar(im, ax=ax, label=r"$F(\gamma,\beta)=\langle H_C\rangle$ (= expected cut)")
    # Overlay the converged points: green = success, white = stuck elsewhere
    succ_mask = final_F >= F_max - tol
    if succ_mask.any():
        ax.scatter(final_points[succ_mask, 0], final_points[succ_mask, 1],
                   c="#00ff66", s=25, edgecolors="k", linewidths=0.4,
                   label=f"converged to max ({success}/{n_starts})")
    if (~succ_mask).any():
        ax.scatter(final_points[~succ_mask, 0], final_points[~succ_mask, 1],
                   c="white", s=20, edgecolors="k", linewidths=0.4,
                   label=f"stuck ({n_starts - success}/{n_starts})")
    ax.plot(g_max, b_max, "r*", markersize=18, label="global max (target)")
    ax.plot(g_min, b_min, "rx", markersize=10, label="global min (reference)")
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"$\beta$")
    ax.set_title(f"p=1 QAOA energy landscape (seed={SEED})")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "hw2_p2_landscape.png"), dpi=150)
    plt.close(fig)

    return landscape, (g_max, b_max, F_max), (success, n_starts)


# ===============================================================
# (c) QAOA at p in {1,2,3,4}
# ===============================================================
@dataclass
class QAOAResult:
    p: int
    mode_cut: int           # cut from most-likely bitstring (argmax probs)
    mode_ratio: float
    sample_cut: int         # best cut over n_shots samples
    sample_ratio: float
    gammas: np.ndarray
    betas: np.ndarray
    expected_cut: float     # <H_C> at the optimum
    runtime: float
    history: List[float]


def part_c(G: nx.Graph, max_cut: int, n_restarts: int = 5,
           max_iter: int = 200, n_shots: int = 1024) -> List[QAOAResult]:
    print(f"\n--- (c) QAOA at p in {{1,2,3,4}} ---")
    cost_fn, probs_qnode, n, edges = build_qaoa(G)

    # We MAXIMIZE <H_C> (= expected cut). PennyLane optimizers minimize, so
    # we minimize -cost_fn.
    def neg_cost(params):
        return -cost_fn(params)

    rng = np.random.default_rng(SEED)
    results: List[QAOAResult] = []
    convergence_curves: Dict[int, List[float]] = {}

    for p in [1, 2, 3, 4]:
        best_F = -np.inf
        best_params = None
        best_history: List[float] = []
        t0 = time.time()
        for r in range(n_restarts):
            # gammas in [0, 2*pi), betas in [0, pi) -- match natural periods
            gammas0 = rng.uniform(0, 2 * np.pi, size=p)
            betas0 = rng.uniform(0, np.pi, size=p)
            init = pnp.array(np.stack([gammas0, betas0]), requires_grad=True)
            opt = qml.AdamOptimizer(stepsize=0.1)
            params = init
            history = []
            for it in range(max_iter):
                # step_and_cost returns (new_params, cost_at_OLD_params); the
                # value we store therefore corresponds to the parameters BEFORE
                # this step.  We append the post-final cost after the loop.
                params, val = opt.step_and_cost(neg_cost, params)
                history.append(-float(val))
            F = float(cost_fn(params))   # <H_C> at the FINAL params
            history.append(F)            # close the curve at the true endpoint
            if F > best_F:
                best_F = F
                best_params = np.array(params)
                best_history = history
        runtime = time.time() - t0

        # Full output distribution at the optimized parameters.
        # Clip tiny negative values that can arise from float-roundoff before
        # using probs as a probability vector.
        probs = np.asarray(probs_qnode(pnp.array(best_params, requires_grad=False)))
        probs = np.clip(probs, 0.0, None)
        probs = probs / probs.sum()

        # (i) Mode bitstring (argmax of probs)
        mode_idx = int(np.argmax(probs))
        mode_bits = tuple((mode_idx >> (n - 1 - k)) & 1 for k in range(n))
        mode_cut = cut_value(mode_bits, edges)

        # (ii) Best-of-shots: draw n_shots bitstrings, take max cut
        sample_rng = np.random.default_rng(SEED + p)
        idxs = sample_rng.choice(len(probs), size=n_shots, p=probs)
        sample_cut = 0
        for idx in np.unique(idxs):
            bits = tuple((int(idx) >> (n - 1 - k)) & 1 for k in range(n))
            c = cut_value(bits, edges)
            if c > sample_cut:
                sample_cut = c

        results.append(QAOAResult(
            p=p,
            mode_cut=mode_cut, mode_ratio=mode_cut / max_cut,
            sample_cut=sample_cut, sample_ratio=sample_cut / max_cut,
            gammas=best_params[0], betas=best_params[1],
            expected_cut=best_F, runtime=runtime, history=best_history,
        ))
        convergence_curves[p] = best_history
        print(f"p={p}: <H_C>*={best_F:.4f}, mode cut={mode_cut}/{max_cut} "
              f"(r={mode_cut/max_cut:.3f}), best-of-{n_shots} cut={sample_cut}/{max_cut} "
              f"(r={sample_cut/max_cut:.3f}), time={runtime:.1f}s")
        print(f"    gammas={best_params[0]}, betas={best_params[1]}")

    # Convergence plot
    fig, ax = plt.subplots(figsize=(7, 5))
    for p, h in convergence_curves.items():
        ax.plot(h, label=f"p={p}")
    ax.axhline(max_cut, color="k", ls="--", label=f"optimal = {max_cut}")
    ax.set_xlabel("Adam iteration")
    ax.set_ylabel(r"$\langle H_C\rangle$ (expected cut)")
    ax.set_title("QAOA convergence")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "hw2_p2_convergence.png"), dpi=150)
    plt.close(fig)

    return results


# ===============================================================
# (d) Simulated annealing comparison
# ===============================================================
@dataclass
class SAResult:
    best_cut: int
    approx_ratio: float
    runtime: float
    bits: Tuple[int, ...]


def part_d_sa(G: nx.Graph, max_cut: int, num_reads: int = 1000) -> SAResult:
    print(f"\n--- (d) Simulated annealing ---")
    edges = list(G.edges())
    # Max-Cut as Ising: H = -1/2 sum z_i z_j. dimod minimizes, so we want to
    # minimize H (which == maximize cut, since minimizing -1/2 sum z_i z_j
    # over disagreeing edges).  Actually -1/2 z_i z_j = -1/2 if z_i=z_j,
    # +1/2 if z_i!=z_j.  Sum -> -|E|/2 + cut.  So minimizing H minimizes
    # (-|E|/2 + cut) -> minimizes the cut, NOT maximizes. We need to flip
    # the sign: use J_ij = +1/2 (i.e. H = +1/2 sum z_i z_j).  Then
    # minimizing -> z_i != z_j everywhere it can -> maximizes cut.
    h: Dict[int, float] = {}
    J: Dict[Tuple[int, int], float] = {(i, j): 0.5 for (i, j) in edges}
    bqm = dimod.BinaryQuadraticModel.from_ising(h, J)

    sampler = SimulatedAnnealingSampler()
    t0 = time.time()
    try:
        sampleset = sampler.sample(bqm, num_reads=num_reads, seed=SEED)
    except TypeError:
        # newer dwave.samplers API may not accept `seed`
        sampleset = sampler.sample(bqm, num_reads=num_reads)
    runtime = time.time() - t0

    best = sampleset.first.sample
    n = G.number_of_nodes()
    bits = tuple(0 if best[i] == 1 else 1 for i in range(n))  # spin +1 -> 0
    cut = cut_value(bits, edges)
    print(f"SA best cut = {cut}/{max_cut} (r={cut/max_cut:.3f}), "
          f"time={runtime:.2f}s, num_reads={num_reads}")
    return SAResult(best_cut=cut, approx_ratio=cut / max_cut,
                    runtime=runtime, bits=bits)


# ===============================================================
# Comparison table
# ===============================================================
def comparison_table(max_cut: int, bf_time: float, sa: SAResult,
                     qaoa_results: List[QAOAResult]) -> str:
    def fmt_time(t: float) -> str:
        # Display sub-millisecond timings as "<0.001" so a value of 0.0000 in
        # the table is not mistaken for "no time was recorded".
        return "<0.001" if t < 1e-3 else f"{t:.4f}"

    lines = []
    lines.append("| Method                      | Best cut | Approx ratio | Time (s) |")
    lines.append("|-----------------------------|----------|--------------|----------|")
    lines.append(f"| Classical brute-force       | {max_cut:>8d} | {1.000:>12.3f} | {fmt_time(bf_time):>8s} |")
    lines.append(f"| Simulated annealing         | {sa.best_cut:>8d} | {sa.approx_ratio:>12.3f} | {fmt_time(sa.runtime):>8s} |")
    for r in qaoa_results:
        lines.append(
            f"| QAOA p={r.p} (best-of-shots)     | {r.sample_cut:>8d} | "
            f"{r.sample_ratio:>12.3f} | {fmt_time(r.runtime):>8s} |"
        )
    return "\n".join(lines)


# ===============================================================
# Main
# ===============================================================
def main():
    print("=" * 64)
    print(f"HW2 Problem 2: Max-Cut with QAOA  (seed = {SEED})")
    print("=" * 64)

    G = nx.gnp_random_graph(n=8, p=0.5, seed=SEED)
    print(f"Graph: n={G.number_of_nodes()} nodes, m={G.number_of_edges()} edges")

    max_cut, opt_partitions, bf_time = part_a(G)

    landscape, gmax, reliability = part_b(G, max_cut, grid=50)

    qaoa_results = part_c(G, max_cut, n_restarts=5, max_iter=200)

    sa = part_d_sa(G, max_cut, num_reads=1000)

    table = comparison_table(max_cut, bf_time, sa, qaoa_results)
    print("\n--- Comparison table ---")
    print(table)

    # Persist results
    out = os.path.join(OUT_DIR, "hw2_p2_result.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"HW2 Problem 2 results (seed={SEED})\n")
        f.write(f"Graph: n={G.number_of_nodes()}, m={G.number_of_edges()}\n")
        f.write(f"Edges: {list(G.edges())}\n")
        f.write(f"\n(a) Brute-force max cut = {max_cut}\n")
        f.write(f"    Number of unique optimal partitions = {len(opt_partitions)}\n")
        for p in opt_partitions[:20]:
            f.write(f"      {p}\n")
        f.write(f"    brute-force compute time (excl. plotting) = {bf_time:.6f}s\n")
        f.write(f"\n(b) p=1 landscape: max F = {landscape.max():.4f} at "
                f"(gamma,beta)=({gmax[0]:.4f},{gmax[1]:.4f})\n")
        f.write(f"    landscape min F = {landscape.min():.4f}\n")
        f.write(f"    optimizer reliability: {reliability[0]}/{reliability[1]} "
                f"random p=1 starts converged within tol=0.1 of global max\n")
        f.write(f"\n(c) QAOA results:\n")
        for r in qaoa_results:
            f.write(f"  p={r.p}: <H_C>={r.expected_cut:.4f}, "
                    f"mode cut={r.mode_cut}/{max_cut} (r={r.mode_ratio:.4f}), "
                    f"best-of-shots cut={r.sample_cut}/{max_cut} (r={r.sample_ratio:.4f}), "
                    f"gammas={r.gammas.tolist()}, betas={r.betas.tolist()}, "
                    f"time={r.runtime:.2f}s\n")
        f.write(f"\n(d) SA: cut={sa.best_cut}/{max_cut} (r={sa.approx_ratio:.4f}), "
                f"time={sa.runtime:.2f}s\n")
        f.write("\nComparison table:\n")
        f.write(table + "\n")
    print(f"\nResults written to {out}")


if __name__ == "__main__":
    main()
