import numpy as np
import csv
import random
import matplotlib.pyplot as plt

# Função para carregar o dataset Iris de um arquivo CSV
def carregar_iris_csv(caminho="CEFET/IA/iris3.txt"):
    X, y = [], []
    with open(caminho, 'r') as f:
        leitor = csv.reader(f)
        for linha in leitor:
            if not linha or linha[0].startswith("sepal"):  # pular cabeçalho
                continue
            X.append([float(x) for x in linha[:4]])
            if linha[4] == "Iris-setosa":
                y.append(0)
            elif linha[4] == "Iris-versicolor":
                y.append(1)
            else:
                y.append(2)
    return np.array(X), np.array(y)

# Dividir manualmente os dados em treino e teste
def dividir_dados(X, y, proporcao_teste=0.2):
    indices = list(range(len(X)))
    random.shuffle(indices)
    corte = int(len(X) * (1 - proporcao_teste))
    indices_treino = indices[:corte]
    indices_teste = indices[corte:]
    return X[indices_treino], X[indices_teste], y[indices_treino], y[indices_teste]

# Métrica de acurácia
def acuracia(y_verdadeiro, y_previsto):
    return np.mean(y_verdadeiro == y_previsto)

# Classificação com base na distância euclidiana
def classificar(anticorpo, X):
    return np.linalg.norm(X - anticorpo, axis=1)

# Carregar dados
X, y = carregar_iris_csv("CEFET/IA/iris3.txt")
X_train, X_test, y_train, y_test = dividir_dados(X, y)

# Parâmetros
n_anticorpos = 50
n_clones = 10
n_geracoes = 20
taxa_mutacao = 0.1

# Inicializar anticorpos
anticorpos = [np.random.rand(X.shape[1]) for _ in range(n_anticorpos)]

historico_acuracia = []

# Treinamento via CLONALG com monitoramento da acurácia
for _ in range(n_geracoes):
    afinidades = []
    for anticorpo in anticorpos:
        distancias = classificar(anticorpo, X_train)
        afinidade = 1 / (np.mean(distancias) + 1e-6)
        afinidades.append(afinidade)

    indices_melhores = np.argsort(afinidades)[-n_clones:]
    melhores_anticorpos = [anticorpos[i] for i in indices_melhores]

    novos_anticorpos = []
    for anticorpo in melhores_anticorpos:
        for _ in range(n_clones):
            clone = anticorpo + taxa_mutacao * np.random.randn(X.shape[1])
            novos_anticorpos.append(clone)

    anticorpos = melhores_anticorpos + novos_anticorpos
    anticorpos = anticorpos[:n_anticorpos]

    # Acurácia com melhor anticorpo da geração atual
    melhor = anticorpos[0]
    distancias = classificar(melhor, X_test)

    # Classificação
    y_pred_temp = []
    for d in distancias:
        pos = sum(1 for od in distancias if od < d)
        p = pos / len(distancias)
        if p < 0.33:
            y_pred_temp.append(2)
        elif p < 0.66:
            y_pred_temp.append(1)
        else:
            y_pred_temp.append(0)

    historico_acuracia.append(acuracia(y_test, y_pred_temp))


# Contar o número de classificações corretas
acertos = sum(yt == yp for yt, yp in zip(y_test, y_pred))

# # Mostrar resultados
# print(f"Número de acertos: {acertos} de {len(y_test)} exemplos de teste")
# print(f"Acurácia (manual): {acc:.2f}")


# Plotar gráfico da evolução da acurácia
plt.figure(figsize=(8, 5))
plt.plot(range(1, n_geracoes + 1), historico_acuracia, marker='o', color='blue')
plt.title("Evolução da Acurácia por Geração")
plt.xlabel("Geração")
plt.ylabel("Acurácia no Teste")
plt.grid(True)
plt.tight_layout()
plt.show()