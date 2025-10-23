import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import warnings
warnings.filterwarnings('ignore')

# Função objetivo: Rastrigin (n = 2)
def rastrigin(x):
    """
    Função Rastrigin - função multimodal com muitos mínimos locais
    Mínimo global em (0, 0) com valor f(0,0) = 0
    """
    A = 10
    return A * len(x) + sum([(xi**2 - A * np.cos(2 * np.pi * xi)) for xi in x])

# Classe que representa uma partícula
class Particula:
    def __init__(self, limites, dimensoes=2):
        """
        Inicializa uma partícula com posição e velocidade aleatórias
        """
        self.posicao = np.random.uniform(limites[0], limites[1], dimensoes)
        self.velocidade = np.random.uniform(-1, 1, dimensoes)
        self.melhor_posicao = np.copy(self.posicao)
        self.melhor_valor = rastrigin(self.posicao)
        self.dimensoes = dimensoes

    def atualizar_velocidade(self, melhor_global, inercia, c1, c2):
        """
        Atualiza velocidade usando a equação clássica do PSO
        v = w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)
        """
        r1, r2 = np.random.rand(self.dimensoes), np.random.rand(self.dimensoes)
        cognitivo = c1 * r1 * (self.melhor_posicao - self.posicao)
        social = c2 * r2 * (melhor_global - self.posicao)
        self.velocidade = inercia * self.velocidade + cognitivo + social

    def atualizar_posicao(self, limites):
        """
        Atualiza posição e verifica se é a melhor já encontrada pela partícula
        """
        self.posicao += self.velocidade
        # Mantém partículas dentro dos limites
        self.posicao = np.clip(self.posicao, limites[0], limites[1])
        
        # Avalia fitness da nova posição
        valor = rastrigin(self.posicao)
        
        # Atualiza pbest se necessário
        if valor < self.melhor_valor:
            self.melhor_posicao = np.copy(self.posicao)
            self.melhor_valor = valor

class PSO:
    def __init__(self, num_particulas=30, c1=2.0, c2=2.0, w_max=0.9, w_min=0.4, limites=[-5.12, 5.12]):
        """
        Inicializa o otimizador PSO
        """
        self.num_particulas = num_particulas
        self.c1 = c1  # Fator cognitivo
        self.c2 = c2  # Fator social
        self.w_max = w_max  # Inércia máxima
        self.w_min = w_min  # Inércia mínima
        self.limites = limites
        
        # Inicializa população aleatoriamente (Item 1)
        self.particulas = [Particula(limites) for _ in range(num_particulas)]
        
        # Inicializa gbest
        self.melhor_global = self.particulas[0].melhor_posicao.copy()
        self.melhor_valor_global = self.particulas[0].melhor_valor
        
        # Encontra o melhor inicial
        for p in self.particulas:
            if p.melhor_valor < self.melhor_valor_global:
                self.melhor_global = p.melhor_posicao.copy()
                self.melhor_valor_global = p.melhor_valor
        
        # Históricos para análise
        self.historico_posicoes = []
        self.historico_gbest = []
        self.historico_fitness = []

    def verificar_criterios_parada(self, iteracao, max_iteracoes, tolerancia=1e-6, janela_estagnacao=20):
        """
        Verifica múltiplos critérios de parada
        """
        criterios_atingidos = []
        
        # 1. Número máximo de iterações
        if iteracao >= max_iteracoes:
            criterios_atingidos.append("Máximo de iterações atingido")
        
        # 2. Frequência de atualização do gbest (estagnação)
        if len(self.historico_gbest) >= janela_estagnacao:
            ultimos_valores = self.historico_gbest[-janela_estagnacao:]
            melhoria = abs(max(ultimos_valores) - min(ultimos_valores))
            if melhoria < tolerancia:
                criterios_atingidos.append(f"Gbest estagnado por {janela_estagnacao} iterações")
        
        # 3. Soma das velocidades próxima de zero
        soma_velocidades = sum(np.linalg.norm(p.velocidade) for p in self.particulas)
        if soma_velocidades < tolerancia * self.num_particulas:
            criterios_atingidos.append("Partículas praticamente paradas")
        
        # 4. Raio da colônia (convergência espacial)
        posicoes = np.array([p.posicao for p in self.particulas])
        if len(posicoes) > 1:
            # Calcula distância entre partículas mais distantes
            distancias = []
            for i in range(len(posicoes)):
                for j in range(i+1, len(posicoes)):
                    dist = np.linalg.norm(posicoes[i] - posicoes[j])
                    distancias.append(dist)
            raio_colonia = max(distancias) if distancias else 0
            
            if raio_colonia < tolerancia * 10:  # Tolerância maior para raio
                criterios_atingidos.append("Colônia convergiu espacialmente")
        
        # 5. Objetivo atingido (opcional - muito próximo do ótimo)
        if abs(self.melhor_valor_global) < tolerancia:
            criterios_atingidos.append("Objetivo global atingido")
        
        return criterios_atingidos

    def executar(self, max_iteracoes=101, tolerancia=1e-6, verbose=True, salvar_historico=True):
        """
        Executa o algoritmo PSO completo
        """
        print(f"🚀 Iniciando PSO com {self.num_particulas} partículas")
        print(f"📊 Melhor valor inicial: {self.melhor_valor_global:.6f}")
        print("-" * 60)
        
        iteracao = 0
        
        while iteracao < max_iteracoes:
            # Calcula inércia com redução linear
            w = self.w_max - iteracao * ((self.w_max - self.w_min) / max_iteracoes)           
            # Para cada partícula
            for p in self.particulas:
                # Item 5: Atualizar velocidade e posição
                p.atualizar_velocidade(self.melhor_global, w, self.c1, self.c2)
                p.atualizar_posicao(self.limites)
                
                # Item 2: Avaliar fitness (feito em atualizar_posicao)
                # Item 3: Pbest já atualizado em atualizar_posicao
                
                # Item 4: Atualizar gbest se necessário
                if p.melhor_valor < self.melhor_valor_global:
                    self.melhor_global = p.melhor_posicao.copy()
                    self.melhor_valor_global = p.melhor_valor
            
            # Salva histórico se solicitado
            if salvar_historico:
                self.historico_posicoes.append([p.posicao.copy() for p in self.particulas])
                self.historico_gbest.append(self.melhor_valor_global)
                self.historico_fitness.append([p.melhor_valor for p in self.particulas])
            
            # Mostra progresso
            if verbose and (iteracao % 20 == 0 or iteracao < 10):
                print(f"Iteração {iteracao:3d}: Melhor = {self.melhor_valor_global:.8f} | Inércia = {w:.3f}")
            if (iteracao == 25 or iteracao == 50 or iteracao == 100):
                print(f'valor de w {w:.3f} na iteração {iteracao}') 
            
            # Verifica critérios de parada
            criterios = self.verificar_criterios_parada(iteracao, max_iteracoes, tolerancia)
            if criterios:
                print(f"\n⏹️  Parada na iteração {iteracao}")
                for criterio in criterios:
                    print(f"   • {criterio}")
                break
            
            iteracao += 1
        
        print(f"🎯 Melhor posição: [{self.melhor_global[0]:.6f}, {self.melhor_global[1]:.6f}]")
        print(f"📈 Melhor valor: {self.melhor_valor_global:.8f}")
        print(f"🔄 Total de iterações: {iteracao}")
        
        return {
            'melhor_posicao': self.melhor_global,
            'melhor_valor': self.melhor_valor_global,
            'iteracoes': iteracao,
            'historico_posicoes': self.historico_posicoes,
            'historico_gbest': self.historico_gbest,
            'historico_fitness': self.historico_fitness
        }


    def criar_animacao(self, intervalo=100, salvar_arquivo=None):
        """
        Cria animação do movimento das partículas
        """
        if not self.historico_posicoes:
            print("❌ Nenhum histórico de posições disponível.")
            return
        
        # Prepara o contorno da função Rastrigin
        x = np.linspace(self.limites[0], self.limites[1], 200)
        y = np.linspace(self.limites[0], self.limites[1], 200)
        X, Y = np.meshgrid(x, y)
        Z = 10 * 2 + (X**2 - 10 * np.cos(2 * np.pi * X)) + (Y**2 - 10 * np.cos(2 * np.pi * Y))
        
        fig, ax = plt.subplots(figsize=(10, 8))
        contorno = ax.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.8)
        
        # Elementos da animação
        pontos = ax.scatter([], [], c='red', s=50, alpha=0.7, edgecolors='black', linewidths=0.5)
        ponto_gbest = ax.scatter([], [], c='white', s=200, marker='*', edgecolors='black', linewidths=2)
        texto_info = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=12, 
                            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        def atualizar_frame(frame):
            if frame < len(self.historico_posicoes):
                # Atualiza posições das partículas
                dados = np.array(self.historico_posicoes[frame])
                pontos.set_offsets(dados)
                
                # Atualiza posição do gbest
                ponto_gbest.set_offsets([self.melhor_global])
                
                # Atualiza informações
                valor_atual = self.historico_gbest[frame] if frame < len(self.historico_gbest) else self.melhor_valor_global
                info_texto = f'Iteração: {frame + 1}\nMelhor Valor: {valor_atual:.6f}\nGbest: ({self.melhor_global[0]:.3f}, {self.melhor_global[1]:.3f})'
                texto_info.set_text(info_texto)
            
            return pontos, ponto_gbest, texto_info
        
        # Configurações do plot
        ax.set_xlim(self.limites[0], self.limites[1])
        ax.set_ylim(self.limites[0], self.limites[1])
        ax.set_xlabel('x₁', fontsize=12)
        ax.set_ylabel('x₂', fontsize=12)
        ax.set_title('PSO na Função Rastrigin - Evolução das Partículas', fontsize=14)
        
        # Adiciona colorbar
        plt.colorbar(contorno, ax=ax, shrink=0.8)
        
        # Marca o mínimo global teórico
        ax.plot(0, 0, 'yellow', marker='X', markersize=15, markeredgecolor='black', 
                markeredgewidth=2, label='Mínimo Global (0,0)')
        ax.legend(loc='upper right')
        
        # Cria animação
        frames = len(self.historico_posicoes)
        animacao = FuncAnimation(fig, atualizar_frame, frames=frames, interval=intervalo, 
                                blit=False, repeat=True)
        
        if salvar_arquivo:
            print(f"💾 Salvando animação como {salvar_arquivo}...")
            animacao.save(salvar_arquivo, writer='pillow', fps=10)
        
        plt.show()
        return animacao

def comparar_parametros():
    """
    Compara diferentes combinações de parâmetros c1 e c2
    """
    print("🔬 Comparando diferentes parâmetros c1 e c2...")
    
    combinacoes = [
        (1.0, 1.0, "Exploração Baixa"),
        (1.0, 2.0, "Favorece Social"),
        (2.0, 1.0, "Favorece Cognitivo"),
        (2.0, 2.0, "Balanceado Padrão"),
        (2.5, 2.5, "Exploração Alta"),
        (0.5, 2.5, "Muito Social")
    ]
    
    resultados = []
    
    for c1, c2, descricao in combinacoes:
        print(f"  Testando {descricao} (c1={c1}, c2={c2})...")
        
        pso = PSO(num_particulas=30, c1=c1, c2=c2, w_max=0.9, w_min=0.4)
        resultado = pso.executar(max_iteracoes=101, verbose=False)
        
        resultados.append({
            'c1': c1,
            'c2': c2,
            'descricao': descricao,
            'melhor_valor': resultado['melhor_valor'],
            'iteracoes': resultado['iteracoes'],
            'historico': resultado['historico_gbest']
        })
    
    # Plotagem dos resultados
    plt.figure(figsize=(12, 6))
    for res in resultados:
        plt.plot(res['historico'], label=f"{res['descricao']} (c1={res['c1']}, c2={res['c2']})", linewidth=2)
    plt.title('Comparação do Efeito de c1 e c2 na Convergência do PSO')
    plt.xlabel('Iterações')
    plt.ylabel('Melhor valor da função (mínimo encontrado)')
    plt.legend()
    plt.grid(True)
    plt.show()

# Função principal para demonstração
def main():
    """
    Demonstração completa do PSO implementado
    """
    print("=" * 80)
    print("🐛 PARTICLE SWARM OPTIMIZATION - FUNÇÃO RASTRIGIN")
    print("=" * 80)
    
    # Execução principal
    print("\n1️⃣ EXECUÇÃO PRINCIPAL DO PSO")
    pso = PSO(num_particulas=30, c1=2.0, c2=2.0, w_max=0.9, w_min=0.4)
    resultado = pso.executar(max_iteracoes=101, tolerancia=1e-6, verbose=True)
    

    # Comparação de parâmetros
    print("\n3️⃣ COMPARAÇÃO DE PARÂMETROS")
    comparar_parametros()
    
    # Animação (opcional - descomente para ver)
    print("\n4️⃣ ANIMAÇÃO DO ENXAME")
    print("🎬 Criando animação...")
    pso.criar_animacao(intervalo=100)
    
    print("\n" + "=" * 80)
    print("✅ DEMONSTRAÇÃO COMPLETA FINALIZADA!")
    print("=" * 80)

if __name__ == "__main__":
    main()