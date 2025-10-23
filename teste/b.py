import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animacao
from matplotlib import cm
import random

# --- Função Objetivo ---
def schaffer_f6(x, y=None):
    if y is None:
        x1, x2 = x
    else:
        x1, x2 = x, y
    numerador = np.sin(np.sqrt(x1**2 + x2**2))**2 - 0.5
    denominador = (1 + 0.001 * (x1**2 + x2**2))**2
    return 0.5 - numerador / denominador

# --- Parâmetros do Algoritmo ---
tam_pop = 20
limites = (-10, 10)
taxa_mutacao = 0.1
taxa_cruzamento = 0.8
geracoes = 50

# --- Inicialização da População ---
def inicializar_populacao():
    return np.array([[random.uniform(*limites), random.uniform(*limites)] for _ in range(tam_pop)])

# --- Avaliação de Fitness ---
def avaliar_fitness(individuo):
    return schaffer_f6(individuo)

# --- Seleção por Torneio ---
def selecao(populacao, k=3):
    selecionados = random.sample(list(populacao), k)
    fitnesses = [avaliar_fitness(ind) for ind in selecionados]
    return selecionados[np.argmax(fitnesses)]

# --- Cruzamento ---
def cruzamento(pai1, pai2):
    if random.random() < taxa_cruzamento:
        return np.array([random.choice([g1, g2]) for g1, g2 in zip(pai1, pai2)])
    return pai1.copy()

# --- Mutação ---
def mutacao(individuo):
    for i in range(len(individuo)):
        if random.random() < taxa_mutacao:
            individuo[i] = random.uniform(*limites)
    return individuo

# --- Evolução ---
populacao = inicializar_populacao()
historico_populacao = [populacao.copy()]
historico_melhor_fitness = []
historico_fitness_medio = []

for _ in range(geracoes):
    fitnesses = np.array([avaliar_fitness(ind) for ind in populacao])
    melhor = np.max(fitnesses)
    media = np.mean(fitnesses)
    historico_melhor_fitness.append(melhor)
    historico_fitness_medio.append(media)

    nova_populacao = []
    for _ in range(tam_pop):
        pai1 = selecao(populacao)
        pai2 = selecao(populacao)
        filho = cruzamento(pai1, pai2)
        filho = mutacao(filho)
        nova_populacao.append(filho)

    populacao = np.array(nova_populacao)
    historico_populacao.append(populacao.copy())

# --- Melhor Solução ---
fitness_final = [avaliar_fitness(ind) for ind in populacao]
idx_melhor = np.argmax(fitness_final)
melhor_solucao = populacao[idx_melhor]
melhor_fitness = fitness_final[idx_melhor]

# --- Gráfico de Evolução do Fitness ---
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

# --- Animação da Evolução da População ---
figura, eixo = plt.subplots(figsize=(10, 8))
intervalo_x = np.linspace(limites[0], limites[1], 200)
intervalo_y = np.linspace(limites[0], limites[1], 200)
X, Y = np.meshgrid(intervalo_x, intervalo_y)
Z = schaffer_f6(X, Y)

def animar(frame):
    eixo.clear()
    contorno = eixo.contour(X, Y, Z, levels=10, cmap='viridis', alpha=0.6)
    eixo.clabel(contorno, inline=True, fontsize=8)

    populacao_atual = historico_populacao[frame]
    fitnesses = np.array([avaliar_fitness(ind) for ind in populacao_atual])
    normalizar = plt.Normalize(vmin=np.min(fitnesses), vmax=np.max(fitnesses))
    cores = cm.hot(normalizar(fitnesses))

    eixo.scatter(populacao_atual[:, 0], populacao_atual[:, 1],
                 c=cores, s=100, edgecolors='black', linewidth=1, alpha=0.8)

    idx_melhor = np.argmax(fitnesses)
    eixo.scatter(populacao_atual[idx_melhor, 0], populacao_atual[idx_melhor, 1],
                 c='blue', s=200, marker='*', edgecolors='white', linewidth=2)

    eixo.set_xlim(limites)
    eixo.set_ylim(limites)
    eixo.set_xlabel('X')
    eixo.set_ylabel('Y')
    eixo.set_title(f'Geração {frame} - Melhor Fitness: {historico_melhor_fitness[frame]:.6f}')
    eixo.grid(True, alpha=0.3)

anim = animacao.FuncAnimation(figura, animar, frames=len(historico_populacao), interval=200)
plt.show()

# --- Superfície 3D ---
figura3d = plt.figure(figsize=(12, 8))
eixo3d = figura3d.add_subplot(111, projection='3d')
superficie = eixo3d.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, edgecolor='none')

melhor_z = schaffer_f6(melhor_solucao)
eixo3d.scatter([melhor_solucao[0]], [melhor_solucao[1]], [melhor_z],
               color='red', s=200, marker='*', edgecolors='black', linewidth=2)

eixo3d.set_xlabel('X')
eixo3d.set_ylabel('Y')
eixo3d.set_zlabel('f(x,y)')
eixo3d.set_title('Função Schaffer F6 - Superfície 3D')
figura3d.colorbar(superficie, ax=eixo3d, shrink=0.5, aspect=5)
plt.tight_layout()
plt.show()

