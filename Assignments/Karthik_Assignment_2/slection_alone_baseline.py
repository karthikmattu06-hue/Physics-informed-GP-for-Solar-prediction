import numpy as np
import matplotlib.pyplot as plt
import os
import sys

print(">>> Running file:", __file__)
print(">>> Current working directory:", os.getcwd())
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
# Fitness function
# -----------------------------------------------------------


def fitness(P, W, S):
    F = []
    for individual in P:
        total_w = sum(individual[i] * S[i][0] for i in range(len(S)))
        total_v = sum(individual[i] * S[i][1] for i in range(len(S)))
        F.append(total_v if total_w <= W else 0)
    return np.array(F)

# -----------------------------------------------------------
# Selection methods
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
# Next generation (Selection only)
# -----------------------------------------------------------


def get_next_gen(P, W, S, rng, use_tournament=False):
    """Selection-only GA step (no crossover, no mutation)."""
    P_next = []
    F = fitness(P, W, S)
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


def run_experiment(config_file, use_tournament):
    P, W, S, g, stop, rng = get_initial_population(config_file)
    label = "Tournament" if use_tournament else "Roulette"

    print(f"\n=== Running Selection-Only GA ({label}) on {config_file} ===")
    print(f"Knapsack capacity: {W}, Population: {len(P)}, Items: {len(S)}")
    print(f"Initial best fitness: {np.max(fitness(P, W, S))}\n")

    gen_list, avg_fit_list, best_fit_list, best_active_genes = [], [], [], []

    while g < stop:
        F = fitness(P, W, S)
        best_idx = int(np.argmax(F))
        best_fit = int(F[best_idx])
        avg_fit = float(np.mean(F))
        active_genes = int(np.sum(P[best_idx]))

        gen_list.append(g)
        avg_fit_list.append(avg_fit)
        best_fit_list.append(best_fit)
        best_active_genes.append(active_genes)

        if g % 10 == 0:
            print(f"Gen {g:3d}: Avg = {avg_fit:7.2f} | "
                  f"Best = {best_fit:4d} | Active genes = {active_genes}")

        P = get_next_gen(P, W, S, rng, use_tournament=use_tournament)
        g += 1

    overall_best_idx = int(np.argmax(best_fit_list))
    print(f"\nOverall best = {best_fit_list[overall_best_idx]} at generation {gen_list[overall_best_idx]} "
          f"({best_active_genes[overall_best_idx]} active genes)\n")

    # -------------------------------------------------------
    # (a) Average population fitness vs generation
    # -------------------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(gen_list, avg_fit_list, linewidth=2)
    plt.xlabel('Generation')
    plt.ylabel('Average Population Fitness')
    plt.title(f'{label} Selection Only — Average Fitness per Generation')
    plt.grid(True)
    plt.tight_layout()

    if "config_1" in config_file:
        fname = f"{label}_SelectOnly_AvgFit.png"
    else:
        fname = f"{label}_SelectOnly_AvgFit_config2.png"
    plt.savefig(fname, dpi=200)
    print(f"Saved: {fname}")

    # -----------------------------------------------------------------
    # (b) Best fitness and active genes per generation
    # -----------------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(8, 5))
    l1, = ax1.plot(gen_list, best_fit_list, linewidth=2, label='Best Fitness')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Best Fitness')

    ax2 = ax1.twinx()
    l2, = ax2.plot(gen_list, best_active_genes, linestyle='--', linewidth=2,
                   label='Active Genes (Best)')
    ax2.set_ylabel('Active Genes')

    lines, labels = [l1, l2], [l1.get_label(), l2.get_label()]
    plt.title(
        f'{label} Selection Only — Best Fitness & Active Genes per Generation')
    ax1.grid(True)
    fig.legend(lines, labels, loc='upper center', ncol=2, frameon=False)
    fig.tight_layout()

    if "config_1" in config_file:
        fname = f"{label}_SelectOnly_BestFit&ActiveGene.png"
    else:
        fname = f"{label}_SelectOnly_BestFit_config2.png"
    fig.savefig(fname, dpi=200)
    print(f"Saved: {fname}\n")

    plt.close('all')


# -----------------------------------------------------------
# Run all experiments
# -----------------------------------------------------------
if __name__ == "__main__":
    for config in ["config_1.txt", "config_2.txt"]:
        run_experiment(config, use_tournament=False)  # Roulette
        run_experiment(config, use_tournament=True)   # Tournament
