import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# 1. Definição das variáveis fuzzy
# Aumentando a granularidade para uma melhor visualização da superfície
comida = ctrl.Antecedent(np.linspace(0, 1, 100), 'comida')
servico = ctrl.Antecedent(np.linspace(0, 1, 100), 'servico')
gorjeta = ctrl.Consequent(np.linspace(0, 20, 100), 'gorjeta')

# 2. Funções de pertinência
comida['ruim'] = fuzz.trimf(comida.universe, [0.0, 0.0, 0.5])
comida['boa'] = fuzz.trimf(comida.universe, [0.25, 0.5, 0.75])
comida['saborosa'] = fuzz.trimf(comida.universe, [0.5, 1.0, 1.0])

servico['ruim'] = fuzz.trimf(servico.universe, [0.0, 0.0, 0.5])
servico['aceitavel'] = fuzz.trimf(servico.universe, [0.25, 0.5, 0.75])
servico['otima'] = fuzz.trimf(servico.universe, [0.5, 1.0, 1.0])

gorjeta['pequena'] = fuzz.trimf(gorjeta.universe, [0, 0, 10])
gorjeta['media'] = fuzz.trimf(gorjeta.universe, [5, 10, 15])
gorjeta['alta'] = fuzz.trimf(gorjeta.universe, [10, 20, 20])

# Visualização das funções de pertinência (opcional, para depuração)
comida.view()
servico.view()
gorjeta.view()
plt.show()

# 3. Regras de decisão (exemplos próprios)
rule1 = ctrl.Rule(comida['ruim'] & servico['ruim'], gorjeta['pequena'])
rule2 = ctrl.Rule(comida['ruim'] & servico['aceitavel'], gorjeta['pequena'])
rule3 = ctrl.Rule(comida['ruim'] & servico['otima'], gorjeta['media'])

rule4 = ctrl.Rule(comida['boa'] & servico['ruim'], gorjeta['pequena'])
rule5 = ctrl.Rule(comida['boa'] & servico['aceitavel'], gorjeta['media'])
rule6 = ctrl.Rule(comida['boa'] & servico['otima'], gorjeta['alta'])

rule7 = ctrl.Rule(comida['saborosa'] & servico['ruim'], gorjeta['media'])
rule8 = ctrl.Rule(comida['saborosa'] & servico['aceitavel'], gorjeta['alta'])
rule9 = ctrl.Rule(comida['saborosa'] & servico['otima'], gorjeta['alta'])

# 4. Sistema fuzzy e simulação
sistema = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
simulador = ctrl.ControlSystemSimulation(sistema)

# Exemplo de uso com valores específicos
print("--- Exemplo de Simulação ---")
simulador.input['comida'] = 0.8  # Exemplo: comida saborosa
simulador.input['servico'] = 0.9  # Exemplo: serviço ótimo
simulador.compute()
print(f"Para Comida=0.8 e Serviço=0.9, Gorjeta sugerida: {simulador.output['gorjeta']:.2f}%")

print("\n--- Gorjetas Mínima e Máxima Pelo Sistema Fuzzy ---")
# Para encontrar a gorjeta mínima possível pelo sistema, usamos os menores inputs
simulador.input['comida'] = comida.universe.min()
simulador.input['servico'] = servico.universe.min()
simulador.compute()
min_gorjeta_sugerida = simulador.output['gorjeta']
print(f"Menor gorjeta possível sugerida pelo sistema (Comida=0, Serviço=0): {min_gorjeta_sugerida:.2f}%")

# Para encontrar a gorjeta máxima possível pelo sistema, usamos os maiores inputs
simulador.input['comida'] = comida.universe.max()
simulador.input['servico'] = servico.universe.max()
simulador.compute()
max_gorjeta_sugerida = simulador.output['gorjeta']
print(f"Maior gorjeta possível sugerida pelo sistema (Comida=1, Serviço=1): {max_gorjeta_sugerida:.2f}%")

print(f"A maior gorjeta sugerida é igual a 20%? {'Sim' if max_gorjeta_sugerida >= 19.9 else 'Não'} (Considerando pequena margem de erro)")


# 5. Geração da Superfície 3D
# Criar uma nova instância de simulação para o plot 3D para evitar interferência com os exemplos
simulador_plot = ctrl.ControlSystemSimulation(sistema)

comida_vals = np.linspace(0, 1, 30) # Reduzir para um plot mais rápido
servico_vals = np.linspace(0, 1, 30)
output = np.zeros((len(comida_vals), len(servico_vals)))

for i, c_val in enumerate(comida_vals):
    for j, s_val in enumerate(servico_vals):
        simulador_plot.input['comida'] = c_val
        simulador_plot.input['servico'] = s_val
        simulador_plot.compute()
        output[i, j] = simulador_plot.output['gorjeta']

# Converter para meshgrid para o plot
comida_mesh, servico_mesh = np.meshgrid(comida_vals, servico_vals)

# Plot 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(comida_mesh, servico_mesh, output, cmap='viridis', rstride=1, cstride=1) # Adicionado rstride e cstride para melhor visualização
ax.set_xlabel('Qualidade da Comida')
ax.set_ylabel('Qualidade do Serviço')
ax.set_zlabel('Gorjeta (%)')
ax.set_title('Superfície de Decisão Fuzzy - Gorjeta')
plt.tight_layout()
plt.show()