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
# Fitness function (original version)
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


# ------------------------
# Roulette selection
# ------------------------
def selection_Roulette(P, F, rng):
    total_fitness = sum(F)
    if total_fitness == 0:
        idx = rng.integers(0, len(F))
        return P[idx].copy()

    probabilities = [f / total_fitness for f in F]
    idx = rng.choice(len(P), p=probabilities)
    return P[idx].copy()


# ------------------------
# Tournament selection
# ------------------------
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
# One-point crossover
# ------------------------
def crossover_OnePoint(p1, p2, rng):
    n = len(p1)
    point = rng.integers(1, n)
    c1 = np.concatenate((p1[:point], p2[point:]))
    c2 = np.concatenate((p2[:point], p1[point:]))
    return c1, c2


# ------------------------
# Mutation
# ------------------------
def mutation(individual, mutation_rate, rng):
    for i in range(len(individual)):
        if rng.random() < mutation_rate:
            individual[i] = 1 - individual[i]
    return individual


# ------------------------
# Generate next generation
# ------------------------
def get_next_gen(P, W, S, rng,
                 mutation_rate=0.1,
                 use_tournament=False,
                 use_crossover=True,
                 use_mutation=True):
    P_next = []
    F = fitness(P, W, S)

    while len(P_next) < len(P):
        # Selection
        if use_tournament:
            p1 = selection_Tournament(P, F, rng)
            p2 = selection_Tournament(P, F, rng)
        else:
            p1 = selection_Roulette(P, F, rng)
            p2 = selection_Roulette(P, F, rng)

        # Crossover (optional)
        if use_crossover:
            c1, c2 = crossover_OnePoint(p1, p2, rng)
        else:
            c1, c2 = p1.copy(), p2.copy()

        # Mutation (optional)
        if use_mutation:
            c1 = mutation(c1, mutation_rate, rng)
            c2 = mutation(c2, mutation_rate, rng)

        # Maintain population size
        if len(P_next) + 2 <= len(P):
            P_next.extend([c1, c2])
        else:
            chosen = c1 if rng.random() < 0.5 else c2
            P_next.append(chosen)

    return np.array(P_next)


# ------------------------
# Main loop
# ------------------------
if __name__ == "__main__":
    P, W, S, g, stop, rng = get_initial_population("config_1.txt")

    print(f"Initial max fitness: {np.max(fitness(P, W, S))}")
    while g < stop:
        P = get_next_gen(P, W, S, rng,
                         mutation_rate=0.1,
                         use_tournament=False,
                         use_crossover=True,
                         use_mutation=True)
        g += 1
        F = fitness(P, W, S)
        print(
            f"Generation {g}: Best fitness = {np.max(F)}, Average = {np.mean(F):.2f}")
