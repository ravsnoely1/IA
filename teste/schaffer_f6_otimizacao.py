import random

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


# --- Funções Objetivo ---
def alpine2(x):
    return -np.prod(np.sqrt(x) * np.sin(x))

def schaffer_f6(x):
    x1, x2 = x
    num = np.sin(np.sqrt(x1**2 + x2**2))**2 - 0.5
    denom = (1 + 0.001 * (x1**2 + x2**2))**2
    return 0.5 - num / denom

funcoes_objetivo = {
    #"alpine2": alpine2,
    "schaffer_f6": schaffer_f6 
}

# --- Componentes do AG ---
def init_population(size, bounds):
    return np.array([[random.uniform(low, high) for (low, high) in bounds] for _ in range(size)])

def tournament_selection(pop, fitnesses, k=3):
    selected = random.sample(list(zip(pop, fitnesses)), k)
    return max(selected, key=lambda x: x[1])[0]

def crossover(p1, p2, crossover_rate):
    if random.random() < crossover_rate:
        return np.array([random.choice([g1, g2]) for g1, g2 in zip(p1, p2)])
    return p1.copy()

def mutate(individual, bounds, mutation_rate):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = random.uniform(bounds[i][0], bounds[i][1])
    return individual

def genetic_algorithm(func, bounds, pop_size, generations, crossover_rate, mutation_rate):
    pop = init_population(pop_size, bounds)
    history = [pop.copy()]
    best_fitnesses = []

    for _ in range(generations):
        fitnesses = np.array([func(ind) for ind in pop])
        best_fitnesses.append(np.max(fitnesses))
        new_pop = []

        for _ in range(pop_size):
            p1 = tournament_selection(pop, fitnesses)
            p2 = tournament_selection(pop, fitnesses)
            child = crossover(p1, p2, crossover_rate)
            child = mutate(child, bounds, mutation_rate)
            new_pop.append(child)

        pop = np.array(new_pop)
        history.append(pop.copy())

    return history, best_fitnesses

# --- Parâmetros do Algoritmo ---
BOUNDS = [(0.1, 10), (0.1, 10)]
POP_SIZE = 50
GENERATIONS = 50
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.1
func_name = "schaffer_f6" 
func = funcoes_objetivo[func_name]

# --- Executar AG ---
history, fitness_evolution = genetic_algorithm(
    func, BOUNDS, POP_SIZE, GENERATIONS, CROSSOVER_RATE, MUTATION_RATE
)

# --- Gráfico de evolução do fitness ---
plt.figure()
plt.plot(fitness_evolution)
plt.xlabel('Geração')
plt.ylabel('Melhor Fitness')
plt.title(f'Evolução do Fitness - {func_name}')
plt.grid()
plt.tight_layout()
#plt.savefig(f"fitness_{func_name}.png")
plt.show()

# --- GIF da evolução da população ---
history = np.array(history)
x = np.linspace(BOUNDS[0][0], BOUNDS[0][1], 300)
y = np.linspace(BOUNDS[1][0], BOUNDS[1][1], 300)
X, Y = np.meshgrid(x, y)
Z = np.array([[func([xi, yi]) for xi in x] for yi in y])

fig, ax = plt.subplots()
ax.set_xlim(BOUNDS[0])
ax.set_ylim(BOUNDS[1])
ax.set_xlabel('x1')
ax.set_ylabel('x2')
contour = ax.contourf(X, Y, Z, levels=50, cmap=cm.viridis)
scat = ax.scatter([], [], c='red', s=10)

def update(frame):
    pop = history[frame]
    scat.set_offsets(pop)
    ax.set_title(f'{func_name} - Geração {frame}')
    return scat,

# MANTER A ANIMAÇÃO VIVA NA MEMÓRIA
ani = animation.FuncAnimation(fig, update, frames=len(history), interval=200)

# Mostrar a animação
plt.show()

