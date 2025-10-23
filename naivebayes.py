from collections import \
    defaultdict  # Estrutura que evita erro com dicionário vazio

import pandas as pd  # Biblioteca para trabalhar com tabelas (dataframes)


# Classe do nosso modelo de Naive Bayes
class NaiveBayes:
    def __init__(self):
        self.probabilidades_classes = {}  # Ex: {"Sim": 0.6, "Não": 0.4}
        self.probabilidades_caracteristicas = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        self.classes_possiveis = []  # Ex: ["Sim", "Não"]
        self.colunas_caracteristicas = []  # Ex: ["Tempo", "Temperatura", ...]

    def treinar(self, dados, nome_coluna_alvo):
        self.colunas_caracteristicas = [col for col in dados.columns if col != nome_coluna_alvo]
        self.classes_possiveis = dados[nome_coluna_alvo].unique().tolist()
        total_linhas = len(dados)

        for classe in self.classes_possiveis:
            dados_da_classe = dados[dados[nome_coluna_alvo] == classe]
            self.probabilidades_classes[classe] = len(dados_da_classe) / total_linhas

            print(f'\nqtde de condições {classe}: {len(dados_da_classe)} de {total_linhas}\n{dados_da_classe}')

            for coluna in self.colunas_caracteristicas:
                valores_possiveis = dados[coluna].unique()
                for valor in valores_possiveis:
                    contagem = len(dados_da_classe[dados_da_classe[coluna] == valor])
                    #print(contagem)

                    # AQUI é feita a suavização de Laplace
                    prob = (contagem + 1) / (len(dados_da_classe) + len(valores_possiveis))
                    #print(prob)

                    self.probabilidades_caracteristicas[coluna][valor][classe] = prob
                    #print(self.probabilidades_caracteristicas[coluna][valor][classe])

    def calcular_probabilidades(self, entrada):
        chances = {}
        for classe in self.classes_possiveis:
            # Começamos com a probabilidade da classe
            prob = self.probabilidades_classes[classe]
            for coluna, valor in entrada.items():
                prob *= self.probabilidades_caracteristicas[coluna].get(valor, {}).get(classe, 1e-6)
            chances[classe] = prob

        total = sum(chances.values())
        if total > 0:
            return {classe: chances[classe] / total for classe in self.classes_possiveis}
        else:
            return chances

    def prever(self, entrada):
        probs = self.calcular_probabilidades(entrada)
        # Retorna a classe com a maior probabilidade
        return max(probs, key=probs.get)

# Lê os dados de um arquivo .txt
def carregar_dados_txt(caminho_arquivo):
    try:
        with open(caminho_arquivo, 'r', encoding='utf-8') as f:
            conteudo = f.read()

        inicio = conteudo.find("dados = [")
        if inicio == -1:
            raise ValueError("O arquivo não contém 'dados = ['")

        fim = conteudo.find("]", inicio) + 1
        lista_dados = eval(conteudo[inicio:].split("=", 1)[1].strip())

        colunas = ['Tempo', 'Temperatura', 'Umidade', 'Vento', 'Jogar']
        df = pd.DataFrame(lista_dados, columns=colunas)
        print(f"✔ Dados carregados com sucesso: {df.shape[0]} registros")
        return df
    except Exception as erro:
        print(f"Erro ao carregar os dados: {erro}")
        return None

# Função para receber os dados do usuário no terminal
def receber_dados_do_usuario(colunas):
    print("\nDigite os valores para prever se deve jogar golfe:")
    entrada = {}
    for coluna in colunas:
        valor = input(f"{coluna}: ").strip().capitalize()
        entrada[coluna] = valor
    return entrada

# Função principal
def main():
    print("=== CLASSIFICADOR NAIVE BAYES - GOLFE ===")

    caminho_arquivo = "C:/Users/ravilla.moreira/Downloads/Base_dados_golfe.txt"
    dados = carregar_dados_txt(caminho_arquivo)

    if dados is None:
        return

    modelo = NaiveBayes()
    modelo.treinar(dados, nome_coluna_alvo='Jogar')

    while True:
        print("\n1. Fazer uma previsão")
        print("2. Sair")
        escolha = input("Escolha uma opção: ").strip()
        
        if escolha == '1':
            entrada = receber_dados_do_usuario(modelo.colunas_caracteristicas)
            probabilidades = modelo.calcular_probabilidades(entrada)
            resposta = modelo.prever(entrada)

            print(f"\n➡ Resultado: Jogar Golfe = {resposta}")
            print("Probabilidades:")
            for classe, p in probabilidades.items():
                print(f"  {classe}: {p:.4f} ({p*100:.2f}%)")

        elif escolha == '2':
            print("Até mais!")
            break
        else:
            print("Opção inválida. Tente novamente.")

# Executa o programa
if __name__ == "__main__":
    main()