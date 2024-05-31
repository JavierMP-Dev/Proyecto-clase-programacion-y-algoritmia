import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time

# Parámetros
n_values = [100, 200, 400]
p_values = [0.1, 0.3, 0.5, 0.7, 0.9]
num_graphs = 5

def generate_random_graphs(n, p, num_graphs):
    graphs = []
    for _ in range(num_graphs):
        G = nx.erdos_renyi_graph(n, p)
        graphs.append(G)
    return graphs

def largest_connected_component_fraction(G):
    largest_cc = max(nx.connected_components(G), key=len)
    return len(largest_cc) / len(G.nodes)

def degree_histogram(G):
    degrees = [deg for _, deg in G.degree()]
    return degrees

def global_clustering_coefficient(G):
    return nx.transitivity(G)

# Resultados
fraction_largest_cc = {}
degree_histograms = {}
global_clustering_coefficients = {}
execution_times = []

# Procesamiento
for n in n_values:
    for p in p_values:
        graphs = generate_random_graphs(n, p, num_graphs)
        
        fractions = []
        histograms = []
        clustering_coeffs = []
        
        start_time = time.time()
        
        for G in graphs:
            # Fracción de la componente conectada más grande
            fractions.append(largest_connected_component_fraction(G))
            
            # Histograma de grados
            histograms.extend(degree_histogram(G))
            
            # Coeficiente de clusterización global
            clustering_coeffs.append(global_clustering_coefficient(G))
        
        execution_time = time.time() - start_time
        execution_times.append(execution_time)
        
        fraction_largest_cc[(n, p)] = np.mean(fractions)
        degree_histograms[(n, p)] = histograms
        global_clustering_coefficients[(n, p)] = np.mean(clustering_coeffs)

# Graficar resultados
fig, axes = plt.subplots(len(n_values), len(p_values), figsize=(20, 15))

for i, n in enumerate(n_values):
    for j, p in enumerate(p_values):
        ax = axes[i, j]
        
        # Histograma de grados
        ax.hist(degree_histograms[(n, p)], bins=range(max(degree_histograms[(n, p)]) + 1), alpha=0.75, edgecolor='black')
        ax.set_title(f'n={n}, p={p}')
        ax.set_xlabel('Grado')
        ax.set_ylabel('Frecuencia')

plt.tight_layout()
plt.show()

# Graficar fracción de nodos en la componente conectada más grande
fig, ax = plt.subplots()
for n in n_values:
    fractions = [fraction_largest_cc[(n, p)] for p in p_values]
    ax.plot(p_values, fractions, label=f'n={n}')

ax.set_title('Fracción de nodos en la componente conectada más grande')
ax.set_xlabel('p')
ax.set_ylabel('Fracción')
ax.legend()
plt.show()

# Graficar coeficiente de clusterización global
fig, ax = plt.subplots()
for n in n_values:
    clustering_coeffs = [global_clustering_coefficients[(n, p)] for p in p_values]
    ax.plot(p_values, clustering_coeffs, label=f'n={n}')

ax.set_title('Coeficiente de clusterización global')
ax.set_xlabel('p')
ax.set_ylabel('Coeficiente de clusterización')
ax.legend()
plt.show()

# Imprimir tiempos de ejecución
for i, (n, p) in enumerate([(n, p) for n in n_values for p in p_values]):
    print(f'n={n}, p={p}: Tiempo de ejecución = {execution_times[i]:.2f} segundos')
git 