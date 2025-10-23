import math
import random
import matplotlib.pyplot as plt

import numpy as np

POP_SIZE = 5
QTD_GENES = 6
N_DIMENSOES = 2  # n=2 conforme especificado
MAX_GERACOES = 50
TOLERANCIA = 0.05
MAXIMO_GLOBAL = 7.88  # Para n=2: 2.808^2 ≈ 7.88

#Inicializando a população. Primeiramente o algoritmo gera uma quantidade de individuos binários.
#Em seguida, para cada individuo é gerado aleatoriamente um valor que determina o quão bom esse indivíduo é para a solução
#O próximo passo é gerar uma população com a quantidade de indivíduos gerados e por fim, é graddo aleatoriamente uma aptidão para
#todos os individuos da população

def gera_individuos():
    return [random.randint(0, 1) for _ in range(QTD_GENES)]

def gera_aptidao():
    return [round(random.random(), 2) for _ in range(QTD_GENES*2)]

# Inicialização da população
def gera_populacao():
    return [gera_individuos() for _ in range(POP_SIZE)]

#Gera uma quantidade de aptidão do tamanho da populaçao
def gera_aptidao_pop():
    return[gera_aptidao() for _ in range(POP_SIZE)]

#A partir do momento que temos uma população e para cada individuo temos 2 valores de aptidão, calculamos um valor
#mais real com o uso da função Alpine2
def alpine2(x):
    return np.prod(np.sqrt(x) * np.sin(x))

#Realizando a clonagem
def clone_candidatos(candidatos, fitnessCandidatos):
    qtde = len(fitnessCandidatos)
    clonesApt = []
    clonesCand = []

    # Vamos iterar enquanto houver valores para remover
    while qtde > 0 and len(fitnessCandidatos) > 0:
        maior = max(fitnessCandidatos)
        indice_maior = fitnessCandidatos.index(maior)

        #print(f"Índice do maior valor: {indice_maior}")

        # Corrigido: selecionar o candidato correspondente ao índice
        elementoAClonar = candidatos[indice_maior]
        cloneCand = np.repeat([elementoAClonar], qtde, axis=0)  # repetir o vetor
        cloneFit = np.repeat(maior, qtde)

        clonesCand.extend(cloneCand.tolist())
        clonesApt.extend(cloneFit.tolist())

        # Remover os elementos já clonados
        candidatos.pop(indice_maior)
        fitnessCandidatos.pop(indice_maior)

        qtde = len(fitnessCandidatos)

    return clonesCand, clonesApt

#Na mutação altero os bits dos candidatos uma quantidade de vezes que a taxa de mutação permite,
#considerando aptidão menor >> taxa maior
def mutar_candidatos(candidatos, fitnessCandidatos):
    taxa_mutacao = 1.00
    mutarCand = []
    mutarApt = []

    while taxa_mutacao > 0 and len(fitnessCandidatos) > 0:
        menor = min(fitnessCandidatos)
        indice_menor = fitnessCandidatos.index(menor)

        elementoAMutar = candidatos[indice_menor][:]  # cópia do indivíduo

        num_genes = len(elementoAMutar)
        num_mutacoes = math.ceil(num_genes * taxa_mutacao)

        # Seleciona posições únicas para mutação
        indices_mutar = random.sample(range(num_genes), num_mutacoes)

        for i in indices_mutar:
            if elementoAMutar[i] == 1:
                elementoAMutar[i] = 0
            else:
                elementoAMutar[i] = 1

        mutarCand.append(elementoAMutar)
        mutarApt.append(menor)

        # Remove os elementos já mutados
        candidatos.pop(indice_menor)
        fitnessCandidatos.pop(indice_menor)
        taxa_mutacao = taxa_mutacao - 0.10

    return mutarCand, mutarApt


def metadinamica(candidatos, fitnessCandidatos, n2=2): 
    """Substitui n2 indivíduos de menor aptidão por novos indivíduos"""
    if len(candidatos) == 0:
        return candidatos, fitnessCandidatos
        
    if n2 >= len(candidatos):
        n2 = max(1, len(candidatos) - 1)
    
    candidatos_copia = [ind[:] for ind in candidatos]
    fitness_copia = fitnessCandidatos[:]
    
    # Encontrar os n2 piores
    fitness_ordenados = sorted(enumerate(fitness_copia), key=lambda x: x[1])
    indices_piores = [i for i, _ in fitness_ordenados[:n2]]
    
    # Substituir pelos novos
    for idx in indices_piores:
        novo_individuo = gera_individuos()
        candidatos_copia[idx] = novo_individuo
        fitness_copia[idx] = alpine2(novo_individuo)
    
    return candidatos_copia, fitness_copia

def calc_int_metadinamica(geracao):
    if geracao < MAX_GERACOES: 
        return max(1, POP_SIZE//5)
    else: 
        return max (2, POP_SIZE//2)

def executa_algoritmo():

    print("="*60)
    print("CLONALG")
    print("="*60)    
    #Avaliar fitness
    fitness_melhores = []
    maiorFitness = []
    candidatos_qt = []
    melhor_fitness_historico = []

    geracao = 0  

    #Calcula a fitness para cada individuo da população
    populacao = gera_populacao()
    #print("população",populacao)
    aptidaoPop = gera_aptidao_pop()
    #print("aptidao da pop",aptidaoPop)

    for idx, apt in enumerate(aptidaoPop):
        fitness_total = 0
        for i in range(0, len(apt), 2):
            par = apt[i:i + 2]
            fitness = alpine2(par)
            fitness_total += fitness
        fitness_melhores.append(fitness_total)
        candidatos_qt.append(populacao[idx])  # associar o indivíduo correspondente
        maiorFitness.append(fitness_total)  

    while geracao < MAX_GERACOES:
        print(f'geracao {geracao +1}')

         # pode ser útil manter os maiores aqui também

        # Selecionar o melhor indivíduo e seu índice
        melhor_valor = max(fitness_melhores)
        melhor_fitness_historico.append(melhor_valor)
        #indice_melhor = fitness_melhores.index(melhor_valor)

        #critérios de parada
        v = MAXIMO_GLOBAL - int(melhor_valor)
        if v <= TOLERANCIA:
            print(f'\nmaximo global encontrado {melhor_valor}')
            break
        
        #verifica melhoria do fitness como criterio de parada
        if len(melhor_fitness_historico) >= 20:
            ultimos_20 = melhor_fitness_historico[-20:]
            melhoria = max(ultimos_20) - min(ultimos_20)
            if (melhoria < 0.001):
                print(f'algoritmo estagnou, melhoria {melhoria}')
                break
        
        clonesInd, clonesAptidao = clone_candidatos(candidatos_qt, fitness_melhores)

        print("clonesInd", clonesInd)
        print("clonesAptidao", clonesAptidao)

        print("mutação")
        candMut,aptDoCandMut = mutar_candidatos(clonesInd, clonesAptidao)
        #Adicionando esses individuos na população inicial e seus fitness junto com os fitness da população inicial
        
        populacao.extend(candMut)
        maiorFitness.extend(aptDoCandMut)
        
        # Resultado final
        print("População atualizada:")
        print(populacao)
        
        print("\nFitness atualizada:")
        print(maiorFitness)
        
        print("tamanho populacao",len(populacao))
        print("tamanho Fitness",len(maiorFitness))
        
        print(f'metadinamica')
        int_metadinamica = calc_int_metadinamica(geracao)
        cand_metadinamica, fitness_metadinamica = metadinamica(populacao, maiorFitness, n2=int_metadinamica)

        #selecao
        todos_candidatos = candidatos_qt + cand_metadinamica
        todos_fitness = fitness_melhores + fitness_metadinamica

        #junta os candidatos e fitness
        pares = list(zip(todos_candidatos, todos_fitness))
        pares_ordenados = sorted(pares, reverse=True)
        
        candidatos_qt = [par[0] for par in pares_ordenados[:POP_SIZE]]
        fitness_melhores = [par[1] for par in pares_ordenados[:POP_SIZE]]
        
        geracao += 1


    print("\n" + "="*60)
    print("RESULTADO FINAL")
    print("="*60)
    
    melhor_final = max(fitness_melhores)
    melhor_indice_final = fitness_melhores.index(melhor_final)
    melhor_individuo_final = candidatos_qt[melhor_indice_final]
    
    print(f"Gerações executadas: {geracao}")
    print(f"Melhor fitness encontrado: {melhor_final:.6f}")

    print(f"Melhor solução: [{melhor_individuo_final[0]:.6f}, {melhor_individuo_final[1]:.6f}]")

    melhores_fitness = sorted(todos_fitness)
    
    # Mostrar toda a população final
    print(f'fitness melhores {todos_fitness}')

   # Plot the fitness improvement over generations
    plt.figure(figsize=(10, 6))
    plt.plot(melhores_fitness, marker='o', color='blue', label='Best Fitness per Generation')
    plt.xlabel('Generations')
    plt.ylabel('')
    plt.title('')
    plt.grid(True)
    plt.show()

    
    return candidatos_qt, fitness_melhores, melhor_fitness_historico
    
if __name__ == "__main__":
    candidatos_finais, fitness_finais, historico = executa_algoritmo()