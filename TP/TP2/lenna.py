import numpy as np
from PIL import Image
import skfuzzy as fuzz
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt


# Fonction pour calculer l'indice de Dunn avec centroids
def calculate_dunn_index(X, labels):
    clusters = np.unique(labels)
    inter_cluster_distances = []
    intra_cluster_diameters = []

    # Calcul des distances inter-clusters basées sur les centroids
    centroids = []
    for i in clusters:
        cluster_i = X[labels == i]
        centroid_i = np.mean(cluster_i, axis=0)
        centroids.append(centroid_i)
    
    centroids = np.array(centroids)

    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            dist = np.linalg.norm(centroids[i] - centroids[j])
            inter_cluster_distances.append(dist)

    # Calcul des diamètres intra-cluster basés sur les centroids
    for i in clusters:
        cluster_i = X[labels == i]
        centroid_i = np.mean(cluster_i, axis=0)
        dist = np.max(np.linalg.norm(cluster_i - centroid_i, axis=1))
        intra_cluster_diameters.append(dist)

    # Indice de Dunn
    dunn_index = np.min(inter_cluster_distances) / np.max(intra_cluster_diameters)
    return dunn_index


# Fonction pour effectuer un clustering KMeans
def kmeans_clustering(image, n_clusters=3):
    # Convertir l'image en tableau numpy
    img_array = np.array(image)

    # Vérifier la forme de l'image
    print(f"Forme de l'image : {img_array.shape}")

    # Si l'image est en niveaux de gris (1 canal), pas besoin de dupliquer pour obtenir 3 canaux
    if len(img_array.shape) == 2:  # Image en niveaux de gris
        pixels = img_array.reshape((-1, 1))  # Les pixels ont une seule valeur de luminance
    else:
        pixels = img_array.reshape((-1, 3))  # Si c'est une image RGB, la transformer en 3 valeurs

    # Normalisation des pixels (valeurs entre 0 et 1)
    pixels = pixels / 255.0

    # Appliquer KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(pixels)

    # Récupérer les labels et l'image segmentée
    labels = kmeans.labels_
    kmeans_image = labels.reshape(img_array.shape)  # Reshape pour l'image segmentée

    # Calculer l'indice de Dunn pour KMeans
    dunn_index = calculate_dunn_index(pixels, labels)
    
    return kmeans_image, kmeans, dunn_index


# Fonction pour effectuer un clustering FCM
def fcm_clustering(image, n_clusters=3):
    # Convertir l'image en tableau numpy
    img_array = np.array(image)

    # Vérifier la forme de l'image
    print(f"Forme de l'image : {img_array.shape}")

    # Si l'image est en niveaux de gris (1 canal), pas besoin de dupliquer pour obtenir 3 canaux
    if len(img_array.shape) == 2:  # Image en niveaux de gris
        pixels = img_array.reshape((-1, 1))  # Les pixels ont une seule valeur de luminance
    else:
        pixels = img_array.reshape((-1, 3))  # Si c'est une image RGB, la transformer en 3 valeurs

    # Normalisation des pixels (valeurs entre 0 et 1)
    pixels = pixels / 255.0

    # Appliquer Fuzzy C-Means
    cntr, u, _, _, _, _, _ = fuzz.cmeans(pixels.T, n_clusters, 2, error=0.005, maxiter=1000)
    
    # Choisir les classes basées sur l'appartenance maximale
    labels = np.argmax(u, axis=0)
    
    # Reshaping pour l'image segmentée
    fcm_image = labels.reshape(img_array.shape)  # Reshape pour l'image segmentée

    # Calculer l'indice de Dunn pour FCM
    dunn_index = calculate_dunn_index(pixels, labels)
    
    return fcm_image, u, dunn_index


# Exemple d'utilisation avec une image en niveaux de gris
image = Image.open('TP2/Lenna_gray.jpg')  # Remplacer 'Lenna_gray.jpg' par votre image en niveaux de gris

# Effectuer un clustering avec KMeans
kmeans_image, kmeans_model, kmeans_dunn_index = kmeans_clustering(image, n_clusters=228)
print(f"KMeans Dunn Index: {kmeans_dunn_index:.4f}")  # Afficher l'indice de Dunn pour KMeans

# Effectuer un clustering avec Fuzzy C-Means
fcm_image, fcm_membership, fcm_dunn_index = fcm_clustering(image, n_clusters=228)
print(f"Fuzzy C-Means Dunn Index: {fcm_dunn_index:.4f}")  # Afficher l'indice de Dunn pour FCM

# Affichage des résultats
plt.subplot(1, 2, 1)
plt.imshow(kmeans_image, cmap='gray')
plt.title('KMeans Clustering')

plt.subplot(1, 2, 2)
plt.imshow(fcm_image, cmap='gray')
plt.title('Fuzzy C-Means Clustering')

plt.show()
