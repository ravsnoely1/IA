import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Função objetivo: Rastrigin (n = 2)
def rastrigin(x):
    A = 10
    return A * len(x) + sum([(xi**2 - A * np.cos(2 * np.pi * xi)) for xi in x])

def criterios_parada_adicionais( historico_gbest, particulas, tolerancia=1e-6):
    # Frequência de atualização do gbest
    if len(historico_gbest) > 10:
        ultimas_10 = historico_gbest[-10:]
        if max(ultimas_10) - min(ultimas_10) < tolerancia:
            return True, "Gbest não melhorou"
    
    # Soma das velocidades próxima de zero
    soma_velocidades = sum(np.linalg.norm(p.velocidade) for p in particulas)
    if soma_velocidades < tolerancia:
        return True, "Partículas paradas"
    
    # Raio da colônia
    posicoes = [p.posicao for p in particulas]
    if len(posicoes) > 1:
        distancias = [np.linalg.norm(p1 - p2) for p1 in posicoes for p2 in posicoes]
        raio = max(distancias)
        if raio < tolerancia:
            return True, "Colônia convergiu"
    
    return False, ""

# Classe que representa uma partícula
class Particula:
    def __init__(self, limites):
        self.posicao = np.random.uniform(limites[0], limites[1], 2)
        self.velocidade = np.random.uniform(-1, 1, 2)
        self.melhor_posicao = np.copy(self.posicao)
        self.melhor_valor = rastrigin(self.posicao)

    def atualizar_velocidade(self, melhor_global, inercia, c1, c2):
        r1, r2 = np.random.rand(2), np.random.rand(2)
        cognitivo = c1 * r1 * (self.melhor_posicao - self.posicao)
        social = c2 * r2 * (melhor_global - self.posicao)
        self.velocidade = inercia * self.velocidade + cognitivo + social
        if (self.velocidade == 0).any():
            print("Particulas imoveis")
            return 0
    

    def atualizar_posicao(self, limites):
        self.posicao += self.velocidade
        self.posicao = np.clip(self.posicao, limites[0], limites[1])
        valor = rastrigin(self.posicao)
        if valor < self.melhor_valor:
            self.melhor_posicao = np.copy(self.posicao)
            self.melhor_valor = valor

# Função principal do PSO
def executar_pso(num_particulas=30, num_iteracoes=100, c1=2.0, c2=2.0, w_max=0.9, w_min=0.4):
    limites = [-5.12, 5.12]
    particulas = [Particula(limites) for _ in range(num_particulas)]
    melhor_global = particulas[0].melhor_posicao
    melhor_valor_global = rastrigin(melhor_global)
    historico_posicoes = []
    melhor_global_controle= []

    for k in range(num_iteracoes):
        # Redução linear da inércia
        w = w_max - k * ((w_max - w_min) / num_iteracoes)
        for p in particulas:
            p.atualizar_velocidade(melhor_global, w, c1, c2)
            p.atualizar_posicao(limites)
            if rastrigin(p.posicao) < melhor_valor_global:
                melhor_global = np.copy(p.posicao)
                melhor_valor_global = rastrigin(melhor_global)
        
        historico_posicoes.append([p.posicao.copy() for p in particulas])
        criterios_parada_adicionais(melhor_global, particulas, tolerancia=1e-6)

    return historico_posicoes, melhor_global

# Executa o PSO
posicoes, melhor = executar_pso()

# Gera o gráfico de contorno da função Rastrigin
x = np.linspace(-5.12, 5.12, 400)
y = np.linspace(-5.12, 5.12, 400)
X, Y = np.meshgrid(x, y)
Z = 10 * 2 + (X**2 - 10 * np.cos(2 * np.pi * X)) + (Y**2 - 10 * np.cos(2 * np.pi * Y))

fig, ax = plt.subplots(figsize=(8, 6))
contorno = ax.contourf(X, Y, Z, levels=50, cmap='viridis')
pontos = ax.scatter([], [], c='red')

def atualizar(frame):
    dados = np.array(posicoes[frame])
    pontos.set_offsets(dados)
    ax.set_title(f'Iteração {frame + 1}')
    return pontos,




melhor_valor_obtido = rastrigin(melhor)
print(f'Melhor posição encontrada: {melhor}')
print(f'Valor da função em melhor posição: {melhor_valor_obtido:.6f}')
print(f'Diferença em relação ao mínimo: {abs(melhor_valor_obtido - 0):.6f}')
m = abs(melhor_valor_obtido - 0)

# Adiciona o ponto do mínimo global (0, 0)
ax.plot(m, marker='*', color='white', markersize=15, label=f'Mínimo Global {m:.6f}')
ax.legend()
# Cria animação
animacao = FuncAnimation(fig, atualizar, frames=len(posicoes), interval=100)
plt.colorbar(contorno)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Distribuição das Partículas na Função Rastrigin')
plt.show()

##  c1 e c2 na convergência do algoritmo.
def executar_pso_com_hist(c1=2.0, c2=2.0, num_iteracoes=100):
    limites = [-5.12, 5.12]
    num_particulas = 30
    w_max = 0.9
    w_min = 0.4
    particulas = [Particula(limites) for _ in range(num_particulas)]
    melhor_global = particulas[0].melhor_posicao
    melhor_valor_global = rastrigin(melhor_global)
    historico_valores = []
    criterios_atingidos = []
    janela_estagnacao = []

    for k in range(num_iteracoes):
        w = w_max - k * ((w_max - w_min) / num_iteracoes)
        for p in particulas:
            p.atualizar_velocidade(melhor_global, w, c1, c2)
            p.atualizar_posicao(limites)
            if rastrigin(p.posicao) < melhor_valor_global:
                melhor_global = np.copy(p.posicao)
                melhor_valor_global = rastrigin(melhor_global)
            janela_estagnacao.append(melhor_valor_global)

        if k >= num_iteracoes:
            criterios_atingidos.append("Máximo de iterações atingido")
        
        # N = 100 # número de elementos finais a verificar
        # if len(janela_estagnacao) >= N and len(set(janela_estagnacao[-N:])) == 1:
        #     print("Últimos valores são iguais. Interrompendo o loop.")
        #     break # isso deve estar dentro de um loop


        historico_valores.append(melhor_valor_global)

    return historico_valores

# Combinações de c1 e c2 para testar
combinacoes = [(1, 1), (1, 2), (2, 1), (2, 2), (2.5, 2.5)]
resultados = []

# Executa PSO para cada combinação
for c1, c2 in combinacoes:
    historico = executar_pso_com_hist(c1, c2)
    resultados.append((f"c1={c1}, c2={c2}", historico))

# Plotagem dos resultados
plt.figure(figsize=(12, 6))
for label, valores in resultados:
    plt.plot(valores, label=label)
plt.title('Comparação do Efeito de c1 e c2 na Convergência do PSO')
plt.xlabel('Iterações')
plt.ylabel('Melhor valor da função (mínimo encontrado)')
plt.legend()
plt.grid(True)
plt.show()
