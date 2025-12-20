# ============================================================
# Q4: Population Size Exploration using ga_knapsack_final.py
# ============================================================
import numpy as np
import importlib.util
import os
import csv
import time

# -----------------------------
# Load GA module dynamically
# -----------------------------
spec = importlib.util.spec_from_file_location(
    "ga_knapsack_final", "ga_knapsack_final.py")
ga = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ga)

# -----------------------------
# Helper: Run one GA trial
# -----------------------------


def run_ga_trial(config_path, pop_size, seed=0):
    # Load config and override pop_size manually
    P, W, S, g, stop, rng = ga.get_initial_population(config_path, seed=seed)
    n = P.shape[1]
    P = rng.integers(0, 2, size=(pop_size, n))  # overwrite population size

    wts = np.array([w for (w, v) in S])
    vals = np.array([v for (w, v) in S])
    pop_size = len(P)

    mutation_rate = 0.1
    use_tournament = True
    use_crossover = True
    use_mutation = True
    use_descent_repair = True

    max_generations = max(1000, stop)
    stall_limit = 30
    stall = 0
    best_global = 0
    generations = 0
    all_feasible_once = False

    while generations < max_generations:
        weights = P @ wts
        feasible_mask = weights <= W
        feasible_count = feasible_mask.sum()
        F = ga.fitness(P, W, S)
        best = int(F.max())
        avg = float(F.mean())

        # ---- Phase 1 ----
        if not all_feasible_once:
            if feasible_count == pop_size:
                all_feasible_once = True
            P = ga.get_next_gen(P, W, S, rng,
                                mutation_rate=mutation_rate,
                                use_tournament=use_tournament,
                                use_crossover=use_crossover,
                                use_mutation=use_mutation,
                                use_descent_repair=use_descent_repair)
        else:
            # ---- Phase 2 ----
            P = ga.get_next_gen(P, W, S, rng,
                                mutation_rate=mutation_rate,
                                use_tournament=use_tournament,
                                use_crossover=use_crossover,
                                use_mutation=use_mutation,
                                use_descent_repair=False,
                                enforce_feasible_children=True)

            if best > best_global:
                best_global = best
                stall = 0
            else:
                stall += 1

            if stall >= stall_limit:
                break
        generations += 1

    # ---- Final Results ----
    weights = P @ wts
    values = P @ vals
    feasible_idx = np.where(weights <= W)[0]
    if feasible_idx.size == 0:
        return {"val": 0, "wt": 0, "items": 0}

    best_idx = feasible_idx[np.argmax(values[feasible_idx])]
    return {
        "val": int(values[best_idx]),
        "wt": int(weights[best_idx]),
        "items": int(P[best_idx].sum()),
    }

# -----------------------------
# Q4 Experiment
# -----------------------------


def population_study(config, pop_sizes, trials=30):
    results = []
    for pop in pop_sizes:
        vals, wts, items = [], [], []
        print(f"\n=== Population size = {pop} ===")
        for t in range(trials):
            res = run_ga_trial(config, pop, seed=pop * 1000 + t)
            vals.append(res["val"])
            wts.append(res["wt"])
            items.append(res["items"])
        mean_val, std_val = np.mean(vals), np.std(vals)
        max_wt = np.max(wts)
        best_items = items[np.argmax(vals)]
        results.append((pop, mean_val, std_val, max_wt, best_items))
        print(
            f"Value: {mean_val:.2f} ± {std_val:.2f} | Max wt: {max_wt} | Items(best): {best_items}")
    return results


# -----------------------------
# Run for config_1.txt or config_2.txt
# -----------------------------
CONFIG = "config_1.txt"     # or "config_2.txt"
POP_SIZES = [10, 20, 30, 40, 50, 75, 100, 150, 200, 300]

start = time.time()
results = population_study(CONFIG, POP_SIZES, trials=30)
print(f"\nRuntime: {time.time()-start:.1f}s")

# Save results to CSV
out_csv = f"Q4_results_{os.path.splitext(CONFIG)[0]}.csv"
with open(out_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Population Size", "Mean Value",
                    "Std Dev", "Max Weight", "Items (Best)"])
    writer.writerows(results)

print(f"Saved results to {out_csv}")
