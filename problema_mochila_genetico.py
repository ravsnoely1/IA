import random
import time
import matplotlib.pyplot as plt
import numpy as np

# Configuração dos parâmetros
NUM_ITENS = 60
POP_SIZE = 25
CAPACIDADE_MOCHILA = 40
GERACOES = 100
TAXA_CROSSOVER = 0.8
TAXA_MUTACAO = 1
TORNEIO_SIZE = 3

inicio = time.perf_counter()

# Gera itens aleatórios com peso e valor
def gerar_itens_aleatorios():
    itens = []
    for i in range(NUM_ITENS):
        peso = random.randint(1, 15)  # Peso entre 1 e 15
        valor = random.randint(5, 50)  # Valor entre 5 e 50
        itens.append({
            'peso': peso,
            'valor': valor,
            'nome': f'Item {i+1}'
        })
    return itens

# Gera um indivíduo (representação binária)
def generate_individual():
    return [random.randint(0, 1) for _ in range(NUM_ITENS)]

# Inicialização da população
def generate_populacao():
    return [generate_individual() for _ in range(POP_SIZE)]

# Função de avaliação (fitness)
def fitness(individuo, itens):
    peso_total = 0
    valor_total = 0
    
    # Percorre o cromossomo do indivíduo
    for gene, item in zip(individuo, itens):
        # Se o bit estiver setado, o item será incluído na mochila
        if gene:
            peso_total += item['peso']
            valor_total += item['valor']
    
    # Se excedeu a capacidade máxima
    if peso_total > CAPACIDADE_MOCHILA:
        return 0  # Penalização por ultrapassar capacidade
    
    # Quanto maior o valor, melhor o fitness
    return valor_total

# Seleção por torneio
def selecao_torneio(populacao, itens):
    selecionados = random.sample(populacao, TORNEIO_SIZE)
    melhor = max(selecionados, key=lambda ind: fitness(ind, itens))
    return melhor

# Crossover de um ponto
def crossover(individuo1, individuo2):
    if random.random() < TAXA_CROSSOVER:
        ponto_corte = random.randint(1, NUM_ITENS - 1)
        return individuo1[:ponto_corte] + individuo2[ponto_corte:]
    return individuo1.copy()

# Mutação
def mutacao(individuo):
    return [bit if random.random() > TAXA_MUTACAO else 1 - bit for bit in individuo]

# Referencia
# https://colab.research.google.com/github/duducosmos/problemadamochila/blob/main/ProblemaDaMochila.ipynb#scrollTo=Agi1CwySoA0w

# Algoritmo Genético Principal
def algoritmo_genetico():
    # Gera itens aleatórios
    itens = gerar_itens_aleatorios()
      
    # Estatísticas para gráficos
    melhores_fitness = []
    fitness_medio = []
    pior_fitness = []
    
    # Inicializa população usando generate_populacao
    populacao = generate_populacao()
    
    # Evolução
    for geracao in range(GERACOES):
        # Avalia fitness de toda população
        fitness_pop = [fitness(ind, itens) for ind in populacao]
        
        # Armazena estatísticas
        melhores_fitness.append(max(fitness_pop))
        fitness_medio.append(np.mean(fitness_pop))
        pior_fitness.append(min(fitness_pop))
        
        # Encontra melhor indivíduo
        melhor_idx = fitness_pop.index(max(fitness_pop))
        melhor_individuo = populacao[melhor_idx]
        
        # Nova população
        nova_populacao = []
        
        # Elitismo - mantém o melhor indivíduo
        nova_populacao.append(melhor_individuo.copy())
        
        # Gera resto da população
        while len(nova_populacao) < POP_SIZE:
            # Seleção
            pai1 = selecao_torneio(populacao, itens)
            pai2 = selecao_torneio(populacao, itens)
            
            # Crossover
            filho1 = crossover(pai1, pai2)
            
            # Mutação
            filho1 = mutacao(filho1)
            
            # Adiciona filhos à nova população
            nova_populacao.extend([filho1])
        
        # Ajusta tamanho da população
        populacao = nova_populacao[:POP_SIZE]
    
    # Avaliação final
    fitness_final = [fitness(ind, itens) for ind in populacao]
    melhor_idx = fitness_final.index(max(fitness_final))
    melhor_solucao = populacao[melhor_idx]
    
    return melhor_solucao, melhores_fitness, fitness_medio, pior_fitness, itens

# Execução do algoritmo
if __name__ == "__main__":
    print("=== Problema da Mochila com Algoritmo Genético ===\n")
    print(f"Capacidade da Mochila: {CAPACIDADE_MOCHILA}")
    print(f"Número de Itens: {NUM_ITENS}")
    print(f"Tamanho da População: {POP_SIZE}")
    print(f"Gerações: {GERACOES}\n")
    
    # Executa o algoritmo
    melhor_solucao, melhores_fitness, fitness_medio, pior_fitness, itens = algoritmo_genetico()
    
    # Mostra resultados
    print("\n=== Resultado Final ===")
    print(f"Melhor solução (cromossomo): {melhor_solucao}")
    
    peso_total = sum(itens[i]['peso'] for i, gene in enumerate(melhor_solucao) if gene)
    valor_total = sum(itens[i]['valor'] for i, gene in enumerate(melhor_solucao) if gene)
    
    print(f"Valor total: {valor_total}")
    print(f"Peso total: {peso_total}")
    
   
    print("Imprimindo o resultado")
    #print("Melhor combinação:", melhor_combinacao)
    print("Soma dos valores:", valor_total)
    print("Soma dos pesos:", CAPACIDADE_MOCHILA)


    # Código a ser medido
    for i in range(1000000):
        pass

    fim = time.perf_counter()
    tempo_execucao = fim - inicio
    print(f"Tempo de execução: {tempo_execucao:.4f} segundos")
    
