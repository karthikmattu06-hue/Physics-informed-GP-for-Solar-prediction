import numpy as np
import os
import sys

print(">>> Running file:", __file__)
print(">>> Current working directory:", os.getcwd())
print(">>> sys.path[0]:", sys.path[0])

# ------------------------
# Population initialization
# ------------------------


def get_initial_population(config, seed=1470):
    """
    Populates variables from config and initializes P at generation 0.
    """
    rng = np.random.default_rng(seed)
    with open(config, "r") as file:
        lines = file.readlines()

    pop_size, n, stop, W = map(int, [lines[i].strip() for i in range(4)])
    S = [tuple(map(int, line.strip().split())) for line in lines[4:]]

    g = 0
    P = rng.integers(0, 2, size=(pop_size, n))  # binary matrix
    return P, W, S, g, stop, rng


# ------------------------
# Fitness functions
# ------------------------
def fitness(P, W, S):
    """
    Computes fitness according to assignment definition:
    fitness = sum(value_i) if total_weight <= W else 0
    """
    F = []
    for individual in P:
        total_weight = sum(individual[i] * S[i][0] for i in range(len(S)))
        total_value = sum(individual[i] * S[i][1] for i in range(len(S)))
        if total_weight <= W:
            F.append(total_value)
        else:
            F.append(0)
    return np.array(F)


def fitness_value_only(P, S):
    """Selection-only fitness: total value regardless of weight."""
    values = np.array([v for (_, v) in S])
    return P @ values


# ------------------------
# Selection
# ------------------------
def selection_Roulette(P, F, rng):
    total_fitness = sum(F)
    if total_fitness == 0:
        idx = rng.integers(0, len(F))
        return P[idx].copy()

    probabilities = [f / total_fitness for f in F]
    idx = rng.choice(len(P), p=probabilities)
    return P[idx].copy()


def selection_Tournament(P, F, rng, k=5):
    idxs = rng.integers(0, len(P), size=k)
    best_idx = idxs[0]
    best_fit = F[best_idx]
    for i in idxs[1:]:
        if F[i] > best_fit:
            best_idx = i
            best_fit = F[i]
    return P[best_idx].copy()


# ------------------------
# Crossover & Mutation
# ------------------------
def crossover_OnePoint(p1, p2, rng):
    n = len(p1)
    point = rng.integers(1, n)
    c1 = np.concatenate((p1[:point], p2[point:]))
    c2 = np.concatenate((p2[:point], p1[point:]))
    return c1, c2


def mutation(individual, mutation_rate, rng, W=None, S=None):
    """
    Mutates an individual by flipping bits with a given mutation rate.
    - For overweight individuals: only accept flips that reduce total weight/value ratio.
    - For feasible (underweight) individuals: mutate freely.
    """
    child = individual.copy()
    if mutation_rate <= 0:
        return child

    wts = np.array([w for (w, v) in S])
    vals = np.array([v for (w, v) in S])

    total_weight = child @ wts
    total_value = child @ vals

    # Compute ratio: higher = less efficient
    current_ratio = np.inf if total_value == 0 else total_weight / total_value

    for i in range(len(child)):
        if rng.random() < mutation_rate:
            proposed = child.copy()
            proposed[i] = 1 - proposed[i]
            new_weight = proposed @ wts
            new_value = proposed @ vals
            new_ratio = np.inf if new_value == 0 else new_weight / new_value

            if total_weight > W:
                # allow only mutations that LOWER the ratio
                if new_ratio < current_ratio:
                    child = proposed
                    total_weight, total_value, current_ratio = new_weight, new_value, new_ratio
            else:
                # underweight: allow all
                child = proposed
                total_weight, total_value, current_ratio = new_weight, new_value, new_ratio

    return child


# ------------------------
# Feasibility repair
# ------------------------
def repair_to_feasible(individual, W, S):
    """Drop highest (w/v) ratio genes until weight <= W."""
    ch = individual.copy()
    wts = np.array([w for (w, v) in S])
    vals = np.array([v if v != 0 else 1e-9 for (w, v) in S])

    while ch @ wts > W:
        active = np.where(ch == 1)[0]
        if active.size == 0:
            break
        worst = active[np.argmax(wts[active] / vals[active])]
        ch[worst] = 0
    return ch


# ------------------------
# Next generation creation
# ------------------------
def get_next_gen(P, W, S, rng,
                 mutation_rate=0.1,
                 use_tournament=True,
                 use_crossover=True,
                 use_mutation=True,
                 use_descent_repair=True,
                 enforce_feasible_children=False,
                 elitism_k=1):
    """
    If enforce_feasible_children=True, every child is projected to feasibility.
    Elitism keeps the top-k current individuals (by strict fitness).
    """
    P_next = []

    # --- Elitism ---
    if elitism_k > 0:
        F_strict = fitness(P, W, S)
        elite_idx = np.argsort(F_strict)[-elitism_k:]
        elites = [P[i].copy() for i in elite_idx]
        P_next.extend(elites)

    # --- Fitness for selection ---
    if enforce_feasible_children:
        F = fitness(P, W, S)
    else:
        F = fitness_value_only(
            P, S) if use_descent_repair else fitness(P, W, S)

    # --- Main reproduction loop ---
    while len(P_next) < len(P):
        # Selection
        if use_tournament:
            p1 = selection_Tournament(P, F, rng)
            p2 = selection_Tournament(P, F, rng)
        else:
            p1 = selection_Roulette(P, F, rng)
            p2 = selection_Roulette(P, F, rng)

        # Crossover
        if use_crossover:
            c1, c2 = crossover_OnePoint(p1, p2, rng)
        else:
            c1, c2 = p1.copy(), p2.copy()

        # Mutation
        if use_mutation:
            c1 = mutation(c1, mutation_rate, rng, W=W, S=S)
            c2 = mutation(c2, mutation_rate, rng, W=W, S=S)

        # Descent repair (soft repair)
        if use_descent_repair and not enforce_feasible_children:
            for c in (c1, c2):
                total_weight = sum(c[i] * S[i][0] for i in range(len(S)))
                if total_weight > W:
                    ratios = [S[i][0] / S[i][1] if c[i] == 1 and S[i][1] != 0 else -1
                              for i in range(len(S))]
                    worst_idx = int(np.argmax(ratios))
                    c[worst_idx] = 0

        # Hard feasibility enforcement (Phase 2)
        if enforce_feasible_children:
            c1 = repair_to_feasible(c1, W, S)
            c2 = repair_to_feasible(c2, W, S)

        # Maintain population size
        remain = len(P) - len(P_next)
        if remain >= 2:
            P_next.extend([c1, c2])
        elif remain == 1:
            P_next.append(c1 if rng.random() < 0.5 else c2)

    return np.array(P_next)


# ------------------------
# Main loop
# ------------------------
if __name__ == "__main__":
    P, W, S, g, stop_from_config, rng = get_initial_population("config_2.txt")

    wts = np.array([w for (w, v) in S])
    vals = np.array([v for (w, v) in S])
    pop_size = len(P)

    mutation_rate = 0.1
    use_tournament = True
    use_crossover = True
    use_mutation = True
    use_descent_repair = True

    max_generations = max(1000, stop_from_config)
    stall_limit = 30
    stall = 0
    best_global = 0
    generations = 0
    all_feasible_once = False

    while generations < max_generations:
        weights = P @ wts
        feasible_mask = weights <= W
        feasible_count = feasible_mask.sum()

        F = fitness(P, W, S)
        best = int(F.max())
        avg = float(F.mean())

        if generations % 10 == 0 or generations == 0:
            print(f"Gen {generations:4d}: Feasible {feasible_count:>3}/{pop_size}, "
                  f"Best = {best}, Avg = {avg:.2f}")

        # ---- Phase 1: Drive toward feasibility ----
        if not all_feasible_once:
            if feasible_count == pop_size:
                all_feasible_once = True
                print("\nAll individuals are feasible! Now monitoring convergence...")
            P = get_next_gen(P, W, S, rng,
                             mutation_rate=mutation_rate,
                             use_tournament=use_tournament,
                             use_crossover=use_crossover,
                             use_mutation=use_mutation,
                             use_descent_repair=use_descent_repair)
        else:
            # ---- Phase 2: Maintain feasibility internally ----
            P = get_next_gen(P, W, S, rng,
                             mutation_rate=mutation_rate,
                             use_tournament=use_tournament,
                             use_crossover=use_crossover,
                             use_mutation=use_mutation,
                             use_descent_repair=False,
                             enforce_feasible_children=True)

            # Stall detection
            if best > best_global:
                best_global = best
                stall = 0
            else:
                stall += 1

            if stall >= stall_limit:
                print(
                    f"\nNo improvement for {stall_limit} generations. Stopping.")
                break

        generations += 1

    # ---- Final report ----
    weights = P @ wts
    values = P @ vals
    feasible_idx = np.where(weights <= W)[0]

    print("\n=== FINAL RESULT ===")
    print(f"Generations run: {generations}")
    print(f"Feasible in final pop: {len(feasible_idx)}/{pop_size}")

    if feasible_idx.size > 0:
        best_idx = feasible_idx[np.argmax(values[feasible_idx])]
        print(f"Best feasible fitness: {int(values[best_idx])}")
        print(f"Total weight: {int(weights[best_idx])} / {W}")
        print("Chromosome:", P[best_idx])
    else:
        print("No feasible individual found.")
