import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import skfuzzy as fuzz
from scipy.spatial.distance import cdist


# Fonction pour calculer l'indice de Dunn
def calculate_dunn_index(X, labels):
    clusters = np.unique(labels)
    inter_cluster_distances = []
    intra_cluster_diameters = []

    # Calcul des distances inter-clusters
    for i in clusters:
        for j in clusters:
            if i != j:
                cluster_i = X[labels == i]
                cluster_j = X[labels == j]
                dist = cdist(cluster_i, cluster_j, metric='euclidean')
                inter_cluster_distances.append(np.min(dist))

    # Calcul des diamètres intra-cluster
    for i in clusters:
        cluster_i = X[labels == i]
        dist = cdist(cluster_i, cluster_i, metric='euclidean')
        intra_cluster_diameters.append(np.max(dist))

    # Indice de Dunn
    dunn_index = np.min(inter_cluster_distances) / np.max(intra_cluster_diameters)
    return dunn_index


# Classe pour gérer les jeux de données
class DatasetManager:
    def __init__(self, n_samples):
        self.n_samples = n_samples
        self.datasets = self._create_datasets()

    def _create_datasets(self):
        noisy_circles = datasets.make_circles(
            n_samples=self.n_samples, factor=0.2, noise=0.05, random_state=170
        )
        return {"noisy_circles": noisy_circles}

    def get_dataset(self, name):
        return self.datasets.get(name)


# Classe pour gérer les algorithmes de clustering
class ClusteringManager:
    def __init__(self, n_clusters=2):
        self.n_clusters = n_clusters

    def fit_predict(self, algorithm_name, X):
        if algorithm_name == "Fuzzy C-Means":
            # Utilisation de Fuzzy C-Means de scikit-fuzzy
            cntr, u, _, _, _, _, _ = fuzz.cmeans(X.T, self.n_clusters, 2, error=0.005, maxiter=1000)
            # Choisir les classes basées sur l'appartenance maximale
            y_pred = np.argmax(u, axis=0)
            return y_pred
        elif algorithm_name == "KMeans":
            # Utilisation de KMeans de sklearn
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
            kmeans.fit(X)
            return kmeans.labels_


# Classe pour la visualisation
class PlotManager:
    def __init__(self, X, clustering_results, dunn_indices):
        self.X = X
        self.clustering_results = clustering_results
        self.dunn_indices = dunn_indices

    def plot(self):
        plt.figure(figsize=(10, 6))
        for i, (name, y_pred) in enumerate(self.clustering_results.items()):
            colors = np.array(
                [
                    "#377eb8", "#ff7f00", "#4daf4a", "#f781bf", "#a65628",
                    "#984ea3", "#999999", "#e41a1c", "#dede00",
                ]
            )
            plt.subplot(1, 2, i + 1)
            plt.scatter(self.X[:, 0], self.X[:, 1], s=10, color=colors[y_pred])
            plt.title(f"{name} Clustering\nDunn Index: {self.dunn_indices[name]:.4f}", size=15)
            plt.xlim(-2.5, 2.5)
            plt.ylim(-2.5, 2.5)
            plt.xticks(())
            plt.yticks(())
        plt.tight_layout()
        plt.show()


# Main
if __name__ == "__main__":
    n_samples = 1500
    n_clusters = 6

    # Gestion des données
    dataset_manager = DatasetManager(n_samples)
    X, _ = dataset_manager.get_dataset("noisy_circles")

    # Normalisation des données
    X = StandardScaler().fit_transform(X)

    # Gestion du clustering
    clustering_manager = ClusteringManager(n_clusters)
    clustering_results = {}
    dunn_indices = {}

    # Comparaison entre KMeans et Fuzzy C-Means
    for algorithm_name in ["KMeans", "Fuzzy C-Means"]:
        start_time = time.time()
        y_pred = clustering_manager.fit_predict(algorithm_name, X)
        end_time = time.time()

        # Calcul de l'indice de Dunn pour chaque résultat de clustering
        dunn_index = calculate_dunn_index(X, y_pred)
        clustering_results[algorithm_name] = y_pred
        dunn_indices[algorithm_name] = dunn_index

        print(f"{algorithm_name} clustering completed in {end_time - start_time:.2f}s")
        print(f"{algorithm_name} Dunn Index: {dunn_index:.4f}")

    # Visualisation
    plot_manager = PlotManager(X, clustering_results, dunn_indices)
    plot_manager.plot()
