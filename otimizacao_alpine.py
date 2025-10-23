import random
import matplotlib.pyplot as plt
import matplotlib.animation as animacao
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # Necessário para gráficos 3D

# --- Parâmetros do Algoritmo Genético --- #
NUM_ITENS = 10
LIMITES = [(0.1, 10), (0.1, 10)]  # Intervalo de busca para cada variável (x1 e x2)
TAM_POP = 50                     # Tamanho da população
GERACOES = 50                    # Número de gerações
TAXA_CRUZAMENTO = 0.8          # Probabilidade de cruzamento entre indivíduos
TAXA_MUTACAO = 0           # Probabilidade de mutação por gene
nome_funcao = "alpine2"      # Nome da função objetivo escolhida


# --- Funções Objetivo alpine2 ---#
def alpine2(x):
    # Função Alpine 2: problema clássico de otimização contínua
    return -np.prod(np.sqrt(x) * np.sin(x))

# def schaffer_f6(x):
#     # Função Schaffer F6: função multimodal com muitos ótimos locais
#     x1, x2 = x
#     numerador = np.sin(np.sqrt(x1**2 + x2**2))**2 - 0.5
#     denominador = (1 + 0.001 * (x1**2 + x2**2))**2
#     return 0.5 - numerador / denominador

# Dicionário com funções disponíveis
funcoes_objetivo = {
    "alpine2": alpine2
    #,"schaffer_f6": schaffer_f6
}

# Seleciona a função com base no nome escolhido
funcao = funcoes_objetivo[nome_funcao]

# --- Funções do Algoritmo Genético --- #

def inicializar_populacao(tamanho, limites):
    # Gera uma população inicial aleatória dentro dos limites definidos
    return np.array([
        [random.uniform(minimo, maximo) for (minimo, maximo) in limites]
        for _ in range(tamanho)
    ])

def calcular_fitness(populacao, aptidoes, k=3):
    # Seleção por torneio: escolhe k indivíduos aleatórios e retorna o melhor
    selecionados = random.sample(list(zip(populacao, aptidoes)), k)
    return max(selecionados, key=lambda x: x[1])[0]


# Crossover de um ponto
def crossover(pai1, pai2):
    if random.random() < TAXA_CRUZAMENTO:
        ponto_corte = random.randint(1, NUM_ITENS - 1)
        return np.array([random.choice([g1, g2]) for g1, g2 in zip(pai1, pai2)])
    return pai1.copy()

def mutacao(individuo, limites):
    # Aplica mutação a cada gene do indivíduo com uma certa taxa
    for i in range(len(individuo)):
        if random.random() < TAXA_MUTACAO:
            individuo[i] = random.uniform(limites[i][0], limites[i][1])
    return individuo


def algoritmo_genetico(funcao, limites, tam_pop, geracoes):
    # Função principal do AG
    populacao = inicializar_populacao(tam_pop, limites)
    historico_pop = [populacao.copy()]  # Armazena a população a cada geração
    historico_melhores = []             # Guarda o melhor fitness por geração

    for _ in range(geracoes):
        aptidoes = np.array([funcao(ind) for ind in populacao])
        historico_melhores.append(np.max(aptidoes))  # Melhor da geração
        nova_populacao = []

        for _ in range(tam_pop):
            # Seleciona dois pais
            pai1 = calcular_fitness(populacao, aptidoes)
            pai2 = calcular_fitness(populacao, aptidoes)
            # Gera filho por cruzamento e mutação
            filho = crossover(pai1, pai2)
            filho = mutacao(filho, limites)
            nova_populacao.append(filho)

        # Atualiza população e armazena
        populacao = np.array(nova_populacao)
        historico_pop.append(populacao.copy())

    return historico_pop, historico_melhores

# --- Execução do Algoritmo Genético --- #
historico_pop, evolucao_fitness = algoritmo_genetico(
    funcao, LIMITES, TAM_POP, GERACOES
)

# --- Visualização dos Resultados --- #

# Cria figura com dois gráficos lado a lado
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Gráfico da evolução do melhor fitness
ax1.plot(evolucao_fitness)
ax1.set_xlabel('Geração')
ax1.set_ylabel('Melhor Fitness')
ax1.set_title('Convergência do Algoritmo Genético')
ax1.grid(True)

# Prepara grade de pontos para visualização da função objetivo
x = np.linspace(0, 10, 100)
y = np.linspace(0, 10, 100)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)

# Calcula valores da função Alpine 2 sobre a grade
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z[i, j] = alpine2([X[i, j], Y[i, j]])

# Obtém a melhor solução encontrada
melhor_index = np.argmax(evolucao_fitness)         # Índice da geração com melhor resultado
melhor_solucao = historico_pop[melhor_index][0]    # Melhor indivíduo dessa geração
melhor_z = funcao(melhor_solucao)                  # Avaliação da função nessa solução

# Mostra a solução no terminal
print(f'melhor_index: {melhor_index}')
print(f'melhor_solucao: {melhor_solucao}')
print(f'melhor_z: {melhor_z}')

# Gráfico 3D da função com destaque para a melhor solução
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)

# Marca a melhor solução encontrada em vermelho
ax2.scatter([melhor_solucao[0]], [melhor_solucao[1]], [melhor_z],
            color='red', s=100, marker='*', edgecolors='black', label='Melhor solução')

# Marca o ótimo global conhecido para Alpine2
ax2.scatter(4.75693772, 7.9572246, 6.119377517145012, 
            color='yellow', s=100, marker='o', label='Ótimo global')

ax2.set_xlabel('x1')
ax2.set_ylabel('x2')
ax2.set_zlabel('f(x)')
ax2.set_title('Função Alpine 2 com solução encontrada')
ax2.legend()

plt.tight_layout()
plt.show()
