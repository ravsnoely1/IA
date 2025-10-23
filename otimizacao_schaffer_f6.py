import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm
import random

# --- Parâmetros do Algoritmo Genético --- #
NUM_ITENS = 10
TAMANHO_POP = 20
LIMITE = (-10, 10)
TAXA_MUTACAO = 0.8
TAXA_CRUZAMENTO = 0.1
GERACOES = 50

# --- Função Objetivo: Schaffer F6 --- #
def schaffer_f6(x, y=None):
    # Suporte para entrada como lista [x, y] ou dois argumentos x, y
    if y is None:
        x1, x2 = x
    else:
        x1, x2 = x, y

    numerador = np.sin(np.sqrt(x1**2 + x2**2))**2 - 0.5
    denominador = (1 + 0.001 * (x1**2 + x2**2))**2
    return 0.5 - numerador / denominador


# --- Inicializa população aleatória --- #
def inicializar_populacao(tamanho, limite):
    return np.array([[random.uniform(*limite), random.uniform(*limite)] for _ in range(tamanho)])

# --- Avalia o fitness (qualidade) de um indivíduo --- #
def calcular_fitness(individuo):
    return schaffer_f6(individuo)

# --- Seleção por torneio (k=3) --- #
def selecao_torneio(populacao, k=3):
    selecionados = random.sample(list(populacao), k)
    fitnesses = [calcular_fitness(ind) for ind in selecionados]
    melhor = selecionados[np.argmax(fitnesses)]
    return melhor

# Crossover de um ponto
def crossover(pai1, pai2):
    if random.random() < TAXA_CRUZAMENTO:
        ponto_corte = random.randint(1, NUM_ITENS - 1)
        return np.array([random.choice([g1, g2]) for g1, g2 in zip(pai1, pai2)])
    return pai1.copy()

# --- Mutação de um indivíduo --- #
def mutacao(individuo, limite):
    for i in range(len(individuo)):
        if random.random() < TAXA_CRUZAMENTO:
            individuo[i] = random.uniform(*limite)
    return individuo


# --- Execução principal do algoritmo genético ---
populacao = inicializar_populacao(TAMANHO_POP, LIMITE)
historico_populacao = [populacao.copy()]
historico_melhor_fitness = []
historico_fitness_medio = []

for geracao in range(GERACOES):
    fitnesses = np.array([calcular_fitness(ind) for ind in populacao])
    melhor_fitness = np.max(fitnesses)
    fitness_medio = np.mean(fitnesses)
    historico_melhor_fitness.append(melhor_fitness)
    historico_fitness_medio.append(fitness_medio)

    nova_populacao = []
    for _ in range(TAMANHO_POP):
        pai1 = selecao_torneio(populacao)
        pai2 = selecao_torneio(populacao)
        filho = crossover(pai1, pai2)
        filho = mutacao(filho, LIMITE)
        nova_populacao.append(filho)

    populacao = np.array(nova_populacao)
    historico_populacao.append(populacao.copy())

# --- Melhor solução final ---
fitnesses_finais = np.array([calcular_fitness(ind) for ind in populacao])
indice_melhor = np.argmax(fitnesses_finais)
melhor_solucao = populacao[indice_melhor]
melhor_fitness = fitnesses_finais[indice_melhor]

# --- Gráfico da evolução do fitness ---
plt.figure(figsize=(10, 6))
plt.plot(historico_melhor_fitness, label='Melhor Fitness', linewidth=2)
plt.plot(historico_fitness_medio, label='Fitness Médio', linewidth=2)
plt.xlabel('Geração')
plt.ylabel('Fitness')
plt.title('Evolução do Fitness ao Longo das Gerações')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# --- Animação da evolução da população no plano ---
fig, ax = plt.subplots(figsize=(10, 8))

# Criar malha de pontos para desenhar curvas de nível
x_range = np.linspace(LIMITE[0], LIMITE[1], 200)
y_range = np.linspace(LIMITE[0], LIMITE[1], 200)
X, Y = np.meshgrid(x_range, y_range)
Z = schaffer_f6(X, Y)

# Função que atualiza o gráfico em cada frame
def animar(frame):
    ax.clear()
    contorno = ax.contour(X, Y, Z, levels=10, cmap='viridis', alpha=0.6)
    ax.clabel(contorno, inline=True, fontsize=8)

    populacao = historico_populacao[frame]
    fitnesses = np.array([calcular_fitness(ind) for ind in populacao])
    norm = plt.Normalize(vmin=np.min(fitnesses), vmax=np.max(fitnesses))
    cores = cm.hot(norm(fitnesses))

    ax.scatter(populacao[:, 0], populacao[:, 1], 
               c=cores, s=100, edgecolors='black', linewidth=1, alpha=0.8)

    indice_melhor = np.argmax(fitnesses)
    ax.scatter(populacao[indice_melhor, 0], populacao[indice_melhor, 1], 
               c='blue', s=200, marker='*', edgecolors='white', linewidth=2)

    ax.set_xlim(LIMITE)
    ax.set_ylim(LIMITE)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Geração {frame} - Melhor Fitness: {historico_melhor_fitness[frame]:.6f}')
    ax.grid(True, alpha=0.3)

# Criar animação
animacao = animation.FuncAnimation(fig, animar, frames=len(historico_populacao), interval=200)
plt.show()

# --- Gráfico 3D da superfície da função ---
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
superficie = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, edgecolor='none')

# Melhor ponto encontrado
z_melhor = schaffer_f6(melhor_solucao)
ax.scatter([melhor_solucao[0]], [melhor_solucao[1]], [z_melhor],
           color='red', s=200, marker='*', edgecolors='black', linewidth=2)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('f(x, y)')
ax.set_title('Função Schaffer F6 - Superfície 3D')
fig.colorbar(superficie, ax=ax, shrink=0.5, aspect=5)
plt.tight_layout()
plt.show()
