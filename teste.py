import math
import random

import numpy as np

POP_SIZE = 6
QTD_GENES = 5
N_DIMENSOES = 2
MAX_GERACOES = 50
TOLERANCIA = 0.05
MAXIMO_GLOBAL = 7.88
LIMITE_INF = 0
LIMITE_SUP = 10

def gera_individuos():
    """Gera indivíduo binário"""
    return [random.randint(0, 1) for _ in range(QTD_GENES)]

def binario_para_real(individuo_binario, limite_inf=LIMITE_INF, limite_sup=LIMITE_SUP):
    """
    CORREÇÃO CRÍTICA: Converte binário para real corretamente
    Cada dimensão usa metade dos genes
    """
    if len(individuo_binario) % N_DIMENSOES != 0:
        # Ajustar se necessário
        genes_por_dim = len(individuo_binario) // N_DIMENSOES
    else:
        genes_por_dim = len(individuo_binario) // N_DIMENSOES
    
    valores_reais = []
    
    for dim in range(N_DIMENSOES):
        inicio = dim * genes_por_dim
        fim = inicio + genes_por_dim
        
        # Pegar os bits para esta dimensão
        bits_dimensao = individuo_binario[inicio:fim]
        
        # Converter binário para decimal
        valor_decimal = 0
        for i, bit in enumerate(reversed(bits_dimensao)):
            valor_decimal += bit * (2 ** i)
        
        # Normalizar para o intervalo [limite_inf, limite_sup]
        max_decimal = (2 ** genes_por_dim) - 1
        if max_decimal > 0:
            valor_real = limite_inf + (valor_decimal / max_decimal) * (limite_sup - limite_inf)
        else:
            valor_real = limite_inf
            
        valores_reais.append(valor_real)
    
    return valores_reais

def alpine2(x):
    """
    CORREÇÃO CRÍTICA: Função Alpine2 robusta
    """
    try:
        x_array = np.array(x)
        # Proteção contra valores problemáticos
        x_abs = np.maximum(np.abs(x_array), 1e-10)
        resultado = np.prod(np.sqrt(x_abs) * np.sin(x_array))
        return max(0, resultado)  # Garantir que não seja negativo
    except:
        return 0.0

def calcular_fitness_individuo(individuo_binario):
    """
    CORREÇÃO CRÍTICA: Calcula fitness corretamente
    1. Converte binário para real
    2. Aplica função Alpine2
    """
    valores_reais = binario_para_real(individuo_binario)
    return alpine2(valores_reais)

def gera_populacao():
    """Gera população inicial"""
    return [gera_individuos() for _ in range(POP_SIZE)]

def calcular_fitness_populacao(populacao):
    """
    CORREÇÃO CRÍTICA: Calcula fitness para toda população
    """
    return [calcular_fitness_individuo(individuo) for individuo in populacao]

def clone_candidatos(candidatos, fitnessCandidatos):
    """
    CORREÇÃO: Clonagem sem destruir listas originais
    """
    if not candidatos or not fitnessCandidatos:
        return [], []
    
    # Fazer cópias para não alterar originais
    candidatos_copia = [ind[:] for ind in candidatos]
    fitness_copia = fitnessCandidatos[:]
    
    qtde = len(fitness_copia)
    clonesApt = []
    clonesCand = []
    
    while qtde > 0 and len(fitness_copia) > 0:
        maior = max(fitness_copia)
        indice_maior = fitness_copia.index(maior)
        
        elementoAClonar = candidatos_copia[indice_maior]
        
        # Clonar qtde vezes
        for _ in range(qtde):
            clonesCand.append(elementoAClonar[:])  # Cópia profunda
            clonesApt.append(maior)
        
        # Remover elementos processados
        candidatos_copia.pop(indice_maior)
        fitness_copia.pop(indice_maior)
        qtde = len(fitness_copia)
    
    return clonesCand, clonesApt

def mutar_candidatos(candidatos, fitnessCandidatos):
    """
    CORREÇÃO: Mutação e recálculo de fitness
    """
    if not candidatos or not fitnessCandidatos:
        return [], []
    
    taxa_mutacao = 1.00
    mutarCand = []
    mutarApt = []
    
    # Fazer cópias
    candidatos_copia = [ind[:] for ind in candidatos]
    fitness_copia = fitnessCandidatos[:]
    
    while taxa_mutacao > 0 and len(fitness_copia) > 0:
        menor = min(fitness_copia)
        indice_menor = fitness_copia.index(menor)
        
        elementoAMutar = candidatos_copia[indice_menor][:]
        
        num_genes = len(elementoAMutar)
        num_mutacoes = max(1, math.ceil(num_genes * taxa_mutacao))
        
        # Mutação
        indices_mutar = random.sample(range(num_genes), min(num_mutacoes, num_genes))
        for i in indices_mutar:
            elementoAMutar[i] = 1 - elementoAMutar[i]  # Flip bit
        
        # CORREÇÃO CRÍTICA: Recalcular fitness após mutação
        novo_fitness = calcular_fitness_individuo(elementoAMutar)
        
        mutarCand.append(elementoAMutar)
        mutarApt.append(novo_fitness)
        
        # Remover elementos processados
        candidatos_copia.pop(indice_menor)
        fitness_copia.pop(indice_menor)
        taxa_mutacao -= 0.10
    
    return mutarCand, mutarApt

def metadinamica(candidatos, fitnessCandidatos, n2=2):
    """
    CORREÇÃO: Metadinâmica com recálculo correto de fitness
    """
    if len(candidatos) == 0:
        return candidatos, fitnessCandidatos
        
    if n2 >= len(candidatos):
        n2 = max(1, len(candidatos) - 1)
    
    # Fazer cópias
    candidatos_copia = [ind[:] for ind in candidatos]
    fitness_copia = fitnessCandidatos[:]
    
    # Encontrar os n2 piores
    fitness_ordenados = sorted(enumerate(fitness_copia), key=lambda x: x[1])
    indices_piores = [i for i, _ in fitness_ordenados[:n2]]
    
    # Substituir pelos novos
    for idx in indices_piores:
        novo_individuo = gera_individuos()
        candidatos_copia[idx] = novo_individuo
        # CORREÇÃO CRÍTICA: Calcular fitness corretamente
        fitness_copia[idx] = calcular_fitness_individuo(novo_individuo)
    
    return candidatos_copia, fitness_copia

def calc_int_metadinamica(geracao):
    """Calcula intensidade da metadinâmica"""
    if geracao < MAX_GERACOES * 0.7: 
        return max(1, POP_SIZE//5)
    else: 
        return max(2, POP_SIZE//3)

def executa_algoritmo():
    """
    ALGORITMO PRINCIPAL - VERSÃO CORRIGIDA
    """
    print("="*60)
    print("CLONALG CORRIGIDO")
    print("="*60)
    
    # INICIALIZAÇÃO CORRETA
    populacao = gera_populacao()
    print("População inicial:")
    for i, ind in enumerate(populacao):
        valores_reais = binario_para_real(ind)
        print(f"  Ind {i+1}: {ind} → Real: [{valores_reais[0]:.3f}, {valores_reais[1]:.3f}]")
    
    # CORREÇÃO CRÍTICA: Calcular fitness corretamente desde o início
    fitness_melhores = calcular_fitness_populacao(populacao)
    candidatos_qt = [ind[:] for ind in populacao]  # Cópia profunda
    
    print("\nFitness inicial:")
    for i, (ind, fit) in enumerate(zip(candidatos_qt, fitness_melhores)):
        valores_reais = binario_para_real(ind)
        print(f"  Ind {i+1}: fitness = {fit:.6f}, valores = [{valores_reais[0]:.3f}, {valores_reais[1]:.3f}]")
    
    melhor_fitness_historico = []
    geracao = 0
    
    # CICLO PRINCIPAL
    while geracao < MAX_GERACOES:
        print(f'\n--- GERAÇÃO {geracao + 1} ---')
        
        # Melhor indivíduo atual
        melhor_valor = max(fitness_melhores)
        melhor_indice = fitness_melhores.index(melhor_valor)
        melhor_individuo = candidatos_qt[melhor_indice]
        melhor_valores_reais = binario_para_real(melhor_individuo)
        
        melhor_fitness_historico.append(melhor_valor)
        
        print(f"Melhor fitness: {melhor_valor:.6f}")
        print(f"Melhor solução: [{melhor_valores_reais[0]:.3f}, {melhor_valores_reais[1]:.3f}]")
        
        # CRITÉRIOS DE PARADA
        if abs(melhor_valor - MAXIMO_GLOBAL) <= TOLERANCIA:
            print(f'\n🎉 MÁXIMO GLOBAL ENCONTRADO! {melhor_valor:.6f}')
            break
        
        if len(melhor_fitness_historico) >= 20:
            ultimos_20 = melhor_fitness_historico[-20:]
            melhoria = max(ultimos_20) - min(ultimos_20)
            if melhoria < 0.001:
                print(f'⚠️ Algoritmo estagnado, melhoria: {melhoria:.6f}')
                break
        
        # CLONAGEM
        print("Realizando clonagem...")
        clonesInd, clonesAptidao = clone_candidatos(candidatos_qt[:], fitness_melhores[:])
        print(f"Clones gerados: {len(clonesInd)}")
        
        # MUTAÇÃO
        print("Realizando mutação...")
        candMut, aptDoCandMut = mutar_candidatos(clonesInd, clonesAptidao)
        print(f"Candidatos mutados: {len(candMut)}")
        
        # METADINÂMICA
        print("Aplicando metadinâmica...")
        int_metadinamica = calc_int_metadinamica(geracao)
        cand_metadinamica, fitness_metadinamica = metadinamica(candMut, aptDoCandMut, n2=int_metadinamica)
        
        # SELEÇÃO
        print("Realizando seleção...")
        todos_candidatos = candidatos_qt + cand_metadinamica
        todos_fitness = fitness_melhores + fitness_metadinamica
        
        # Ordenar por fitness (decrescente)
        pares = list(zip(todos_candidatos, todos_fitness))
        pares_ordenados = sorted(pares, key=lambda x: x[1], reverse=True)
        
        # Selecionar os melhores
        candidatos_qt = [par[0] for par in pares_ordenados[:POP_SIZE]]
        fitness_melhores = [par[1] for par in pares_ordenados[:POP_SIZE]]
        
        # Mostrar população atual
        print("População atual:")
        for i, (ind, fit) in enumerate(zip(candidatos_qt, fitness_melhores)):
            valores_reais = binario_para_real(ind)
            print(f"  {i+1}: fitness = {fit:.6f}, valores = [{valores_reais[0]:.3f}, {valores_reais[1]:.3f}]")
        
        geracao += 1
    
    # RESULTADO FINAL
    print("\n" + "="*60)
    print("RESULTADO FINAL")
    print("="*60)
    
    melhor_final = max(fitness_melhores)
    melhor_indice_final = fitness_melhores.index(melhor_final)
    melhor_individuo_final = candidatos_qt[melhor_indice_final]
    melhor_valores_finais = binario_para_real(melhor_individuo_final)
    
    print(f"Gerações executadas: {geracao}")
    print(f"Melhor fitness encontrado: {melhor_final:.6f}")
    print(f"Máximo global teórico: {MAXIMO_GLOBAL}")
    print(f"Diferença: {abs(melhor_final - MAXIMO_GLOBAL):.6f}")
    print(f"Melhor solução binária: {melhor_individuo_final}")
    print(f"Melhor solução real: [{melhor_valores_finais[0]:.6f}, {melhor_valores_finais[1]:.6f}]")
    print(f"Ponto ótimo teórico: [7.917, 7.917]")
    
    if abs(melhor_final - MAXIMO_GLOBAL) <= TOLERANCIA:
        print("✅ MÁXIMO GLOBAL ENCONTRADO!")
    else:
        print("⚠️ NÃO ENCONTROU MÁXIMO GLOBAL")
    
    # Mostrar população final
    print(f"\n📊 POPULAÇÃO FINAL ({len(candidatos_qt)} indivíduos):")
    for i, (ind, fit) in enumerate(zip(candidatos_qt, fitness_melhores)):
        valores_reais = binario_para_real(ind)
        print(f"  {i+1:2d}: {ind} → [{valores_reais[0]:7.4f}, {valores_reais[1]:7.4f}] → fitness: {fit:.6f}")
    
    return candidatos_qt, fitness_melhores, melhor_fitness_historico

def testar_conversao():
    """Função para testar a conversão binário-real"""
    print("="*50)
    print("TESTE DE CONVERSÃO BINÁRIO → REAL")
    print("="*50)
    
    # Alguns exemplos
    exemplos = [
        [0, 0, 0, 0, 0],  # Mínimo
        [1, 1, 1, 1, 1],  # Máximo
        [1, 0, 1, 0, 1],  # Meio termo
        [0, 1, 1, 1, 0],  # Outro exemplo
    ]
    
    for binario in exemplos:
        real = binario_para_real(binario)
        fitness = calcular_fitness_individuo(binario)
        print(f"Binário: {binario} → Real: [{real[0]:.3f}, {real[1]:.3f}] → Fitness: {fitness:.6f}")

if __name__ == "__main__":
    # Teste de conversão
    testar_conversao()
    
    # Executar algoritmo
    candidatos_finais, fitness_finais, historico = executa_algoritmo()