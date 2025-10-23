import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Substitua pelo caminho do seu arquivo CSV
df = pd.read_csv("CEFET/IA/heart.csv", sep=";")

# Suponha que a última coluna seja o alvo
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Normalização (opcional dependendo dos dados)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Função para avaliar modelo por 30 execuções
def avalia_modelo(clf, X, y, n_execucoes=30):
    accs = []
    for _ in range(n_execucoes):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        accs.append(accuracy_score(y_val, y_pred))
    return np.mean(accs), np.std(accs)

# Árvore de decisão
dt = DecisionTreeClassifier()
media_dt, std_dt = avalia_modelo(dt, X_scaled, y)
print(f"Acurácia Árvore de Decisão: {media_dt:.2f} (std = {std_dt:.2f})")

# Floresta aleatória
rf = RandomForestClassifier()
media_rf, std_rf = avalia_modelo(rf, X_scaled, y)
print(f"Acurácia Floresta Aleatória: {media_rf:.2f} (std = {std_rf:.2f})")

# Variação da acurácia com profundidade
max_depths = list(range(1, X.shape[1]+1))
depth_results = []

for depth in max_depths:
    rf = RandomForestClassifier(max_depth=depth)
    acc_mean, acc_std = avalia_modelo(rf, X_scaled, y)
    depth_results.append((depth, acc_mean, acc_std))

# Resultado por profundidade
for depth, acc_mean, acc_std in depth_results:
    print(f"Profundidade: {depth}, Acurácia: {acc_mean:.2f}, STD: {acc_std:.2f}")
