"""
GA Final Selection-Only Experiment
-----------------------------------
This script compares selection-only behavior (no crossover, no mutation)
using the enhanced fitness function from ga_knapsack_final.py.

It runs both Roulette and Tournament selection for config_1.txt and config_2.txt.
For each configuration, it tracks:
  - Average population fitness vs. generation
  - Fittest individual's fitness and active genes vs. generation
  - Feasibility counts per generation
  - Overall best fitness and generation
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

print(">>> Running file:", __file__)
print(">>> Working directory:", os.getcwd())
print(">>> sys.path[0]:", sys.path[0])

# -----------------------------------------------------------
# Population initialization
# -----------------------------------------------------------


def get_initial_population(config, seed=1470):
    rng = np.random.default_rng(seed)
    with open(config, "r") as file:
        lines = file.readlines()

    pop_size, n, stop, W = map(int, [lines[i].strip() for i in range(4)])
    S = [tuple(map(int, line.strip().split())) for line in lines[4:]]
    g = 0
    P = rng.integers(0, 2, size=(pop_size, n))
    return P, W, S, g, stop, rng


# -----------------------------------------------------------
# Enhanced fitness function (from ga_knapsack_final.py)
# -----------------------------------------------------------
def enhanced_fitness(P, W, S):
    """Improved fitness: penalizes overweight individuals smoothly."""
    F = []
    for individual in P:
        w = sum(individual[i] * S[i][0] for i in range(len(S)))
        v = sum(individual[i] * S[i][1] for i in range(len(S)))
        if w <= W:
            F.append(v)
        else:
            # Smooth penalty proportional to violation
            penalty = (w - W) / W
            F.append(v / (1 + penalty * 10))
    return np.array(F)


# -----------------------------------------------------------
# Feasibility helper
# -----------------------------------------------------------
def count_feasible(P, W, S):
    """Counts how many individuals satisfy total_weight <= W."""
    wts = np.array([w for (w, v) in S])
    weights = P @ wts
    return int((weights <= W).sum())


# -----------------------------------------------------------
# Selection operators
# -----------------------------------------------------------
def selection_Roulette(P, F, rng):
    total_fitness = np.sum(F)
    if total_fitness == 0:
        idx = rng.integers(0, len(P))
        return P[idx].copy()
    probs = F / total_fitness
    idx = rng.choice(len(P), p=probs)
    return P[idx].copy()


def selection_Tournament(P, F, rng, k=5):
    idxs = rng.integers(0, len(P), size=k)
    best_idx = idxs[np.argmax(F[idxs])]
    return P[best_idx].copy()


# -----------------------------------------------------------
# Selection-only generation
# -----------------------------------------------------------
def get_next_gen(P, W, S, rng, use_tournament=False):
    """Selection-only: no crossover, no mutation."""
    P_next = []
    F = enhanced_fitness(P, W, S)
    while len(P_next) < len(P):
        if use_tournament:
            p1 = selection_Tournament(P, F, rng)
            p2 = selection_Tournament(P, F, rng)
        else:
            p1 = selection_Roulette(P, F, rng)
            p2 = selection_Roulette(P, F, rng)

        if len(P_next) + 2 <= len(P):
            P_next.extend([p1, p2])
        else:
            chosen = p1 if rng.random() < 0.5 else p2
            P_next.append(chosen)
    return np.array(P_next)


# -----------------------------------------------------------
# Experiment runner
# -----------------------------------------------------------
def run_selection_experiment(config_file, use_tournament):
    P, W, S, g, stop, rng = get_initial_population(config_file)
    label = "Tournament" if use_tournament else "Roulette"
    config_name = os.path.splitext(config_file)[0]

    print(
        f"\n=== Enhanced Fitness (Selection-Only, {label}) on {config_file} ===")
    print(f"Capacity={W}, Pop={len(P)}, Items={len(S)}")
    print(f"Initial best fitness={enhanced_fitness(P, W, S).max():.3f}")

    pop_size = len(P)
    gen_list, avg_fit_list, best_fit_list, best_genes_list = [], [], [], []

    feas_start = count_feasible(P, W, S)

    while g < stop:
        F = enhanced_fitness(P, W, S)
        best_idx = int(np.argmax(F))
        best_fit = int(F[best_idx])
        avg_fit = float(np.mean(F))
        active_genes = int(np.sum(P[best_idx]))
        feas_now = count_feasible(P, W, S)

        gen_list.append(g)
        avg_fit_list.append(avg_fit)
        best_fit_list.append(best_fit)
        best_genes_list.append(active_genes)

        if g % 10 == 0:
            print(f"Gen {g:3d}: Feasible {feas_now:>3}/{pop_size} | "
                  f"Avg={avg_fit:7.2f} | Best={best_fit:5d} | Genes={active_genes}")

        P = get_next_gen(P, W, S, rng, use_tournament=use_tournament)
        g += 1

    feas_end = count_feasible(P, W, S)

    # Overall best across generations
    overall_best_idx = int(np.argmax(best_fit_list))
    overall_best_gen = gen_list[overall_best_idx]
    overall_best_fit = best_fit_list[overall_best_idx]
    overall_best_genes = best_genes_list[overall_best_idx]

    print(f"\nStrictly feasible at start: {feas_start}/{pop_size}")
    print(f"Strictly feasible at end:   {feas_end}/{pop_size}")
    print(f"\nOverall best fitness={overall_best_fit}, "
          f"Active genes={overall_best_genes}, Generation={overall_best_gen}")

    # ----------------- Plot (a): Average fitness -----------------
    plt.figure(figsize=(8, 5))
    plt.plot(gen_list, avg_fit_list, linewidth=2)
    plt.xlabel('Generation')
    plt.ylabel('Average Population Fitness')
    plt.title(
        f'Enhanced Selection-Only ({label}) — Avg Fitness ({config_name})')
    plt.grid(True)
    plt.tight_layout()
    fname1 = f'EnhancedSelectionOnly_{config_name}_{label}_AvgFitness.png'
    plt.savefig(fname1, dpi=200)
    print(f"Saved: {fname1}")

    # ----------------- Plot (b): Best fitness & active genes -----------------
    fig, ax1 = plt.subplots(figsize=(8, 5))
    l1, = ax1.plot(gen_list, best_fit_list, linewidth=2, label='Best Fitness')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Best Fitness')
    ax2 = ax1.twinx()
    l2, = ax2.plot(gen_list, best_genes_list, linestyle='--', linewidth=2,
                   color='orange', label='Active Genes (Best)')
    ax2.set_ylabel('Active Genes')
    lines = [l1, l2]
    labels = [line.get_label() for line in lines]
    fig.legend(lines, labels, loc='upper center', ncol=2, frameon=False)
    plt.title(
        f'Enhanced Selection-Only ({label}) — Best Fit & Genes ({config_name})')
    plt.grid(True)
    fig.tight_layout()
    fname2 = f'EnhancedSelectionOnly_{config_name}_{label}_BestFitness&Genes.png'
    plt.savefig(fname2, dpi=200)
    print(f"Saved: {fname2}")
    plt.close('all')


# -----------------------------------------------------------
# Run experiments for both configs
# -----------------------------------------------------------
if __name__ == "__main__":
    for cfg in ["config_1.txt", "config_2.txt"]:
        for use_tournament in [False, True]:
            run_selection_experiment(cfg, use_tournament)
