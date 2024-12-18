import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from skfuzzy.cluster import cmeans
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# Génération des données
n_samples = 1500
random_state = 170
transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]

X, y = make_blobs(n_samples=n_samples, random_state=random_state)
X_aniso = np.dot(X, transformation)  # Anisotropic blobs
X_varied, y_varied = make_blobs(
    n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state
)  # Unequal variance
X_filtered = np.vstack(
    (X[y == 0][:500], X[y == 1][:100], X[y == 2][:10])
)  # Unevenly sized blobs
y_filtered = [0] * 500 + [1] * 100 + [2] * 10

datasets = [(X, y), (X_aniso, y), (X_varied, y_varied), (X_filtered, y_filtered)]
titles = [
    "Mixture of Gaussian Blobs",
    "Anisotropically Distributed Blobs",
    "Unequal Variance",
    "Unevenly Sized Blobs",
]

from scipy.spatial.distance import cdist
import numpy as np

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


# Calcul de l'indice de Dunn pour K-Means
for i, (data, ground_truth) in enumerate(datasets):
    kmeans = KMeans(n_clusters=3, random_state=random_state)
    kmeans_labels = kmeans.fit_predict(data)
    dunn_kmeans = calculate_dunn_index(data, kmeans_labels)
    print(f"Indice de Dunn (K-Means) pour {titles[i]} : {dunn_kmeans:.4f}")



# Fonction pour appliquer Fuzzy C-Means
def apply_fcm(X, n_clusters):
    cntr, u, _, _, _, _, _ = cmeans(X.T, c=n_clusters, m=2.0, error=0.005, maxiter=1000)
    labels = np.argmax(u, axis=0)
    return labels

# Création des graphiques K-Means
fig_kmeans, axs_kmeans = plt.subplots(2, 2, figsize=(12, 12), constrained_layout=True)
fig_kmeans.suptitle("Résultats de K-Means", fontsize=16, y=0.95)

for i, (data, ground_truth) in enumerate(datasets):
    row, col = divmod(i, 2)
    kmeans = KMeans(n_clusters=3, random_state=random_state)
    kmeans_labels = kmeans.fit_predict(data)
    axs_kmeans[row, col].scatter(data[:, 0], data[:, 1], c=kmeans_labels, cmap="viridis")
    axs_kmeans[row, col].set_title(titles[i])

# Création des graphiques Fuzzy C-Means
fig_fcm, axs_fcm = plt.subplots(2, 2, figsize=(12, 12), constrained_layout=True)
fig_fcm.suptitle("Résultats de Fuzzy C-Means", fontsize=16, y=0.95)

for i, (data, ground_truth) in enumerate(datasets):
    row, col = divmod(i, 2)
    fcm_labels = apply_fcm(data, n_clusters=3)
    axs_fcm[row, col].scatter(data[:, 0], data[:, 1], c=fcm_labels, cmap="viridis")
    axs_fcm[row, col].set_title(titles[i])

# Calcul de l'indice de Dunn pour FCM
for i, (data, ground_truth) in enumerate(datasets):
    fcm_labels = apply_fcm(data, n_clusters=3)
    dunn_fcm = calculate_dunn_index(data, fcm_labels)
    print(f"Indice de Dunn (FCM) pour {titles[i]} : {dunn_fcm:.4f}")

plt.show()
