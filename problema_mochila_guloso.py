import random
import time
from itertools import combinations

inicio = time.perf_counter()

#Variáveis de decisão
# Parâmetros
num_itens = 60
capacidade_mochila = 40

# Geração de itens
itens = [{'peso': random.randint(1, 20), 'valor': random.randint(10, 100)} for _ in range(num_itens)]


# Definido uma população através do algoritmo de mochila (por força bruta)
#Cada individuo da população será uma mochila
def Guloso_forca_bruta(itens, capacidade):
    n = len(itens)
    melhor_valor = 0
    melhor_combinacao = None
    capacidade_usada = None
    for r in range(n + 1):
        for combinacao in combinations(itens, r):
            peso_total = sum(item['peso'] for item in combinacao)
            valor_total = sum(item['valor'] for item in combinacao)
            if peso_total <= capacidade and valor_total > melhor_valor:
                melhor_valor = valor_total
                melhor_combinacao = combinacao
                capacidade_usada = peso_total
    return melhor_valor, melhor_combinacao, capacidade_usada

melhor_valor, melhor_combinacao, capacidade_usada = Guloso_forca_bruta(itens, capacidade_mochila)


print("Imprimindo o resultado")
#print("Melhor combinação:", melhor_combinacao)
print("Soma dos valores:", melhor_valor)
print("Soma dos pesos:", capacidade_usada)


# Código a ser medido
for i in range(1000000):
    pass

fim = time.perf_counter()
tempo_execucao = fim - inicio
print(f"Tempo de execução: {tempo_execucao:.4f} segundos")

