import matplotlib.pyplot as plt
from math import sqrt
from typing import List
import random
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import silhouette_score


class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Point({self.x}, {self.y})"

def Kmeans(points: List[Point], centres: List[Point]):
    max_iterations = 10 
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'cyan', 'magenta', 'yellow']

    plt.figure(figsize=(8, 6))  # Créer la fenêtre pour le graphique
    for iteration in range(max_iterations):
        classe = {centre: [] for centre in centres}
        
        for pt in points:
            dist_min = float('inf')
            centre_min = None
            for centre in centres:
                dist = sqrt((pt.x - centre.x)**2 + (pt.y - centre.y)**2)
                if dist < dist_min:
                    dist_min = dist
                    centre_min = centre
            classe[centre_min].append(pt)

        # Effacer l'ancienne figure pour la mise à jour
        plt.clf()

        # Affichage des points associés à chaque centre
        for idx, (centre, points_in_class) in enumerate(classe.items()):
            x_vals = [pt.x for pt in points_in_class]
            y_vals = [pt.y for pt in points_in_class]
            plt.scatter(x_vals, y_vals, c=colors[idx], label=f'Centre {centre} (Cluster {idx+1})', alpha=0.6)

        # Affichage des centres de cluster
        centre_x_vals = [centre.x for centre in centres]
        centre_y_vals = [centre.y for centre in centres]
        plt.scatter(centre_x_vals, centre_y_vals, c='black', marker='X', label='Centres', s=200)

        # Titre et légende
        plt.title(f"K-means - Iteration {iteration + 1}")
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend(loc='upper right')
        plt.grid(True)

        new_centres = []
        for centre in centres:
            if classe[centre]:
                moyenne_x = sum(pt.x for pt in classe[centre]) / len(classe[centre])
                moyenne_y = sum(pt.y for pt in classe[centre]) / len(classe[centre])
                new_centres.append(Point(moyenne_x, moyenne_y))
            else:
                new_centres.append(centre)

        if all(abs(new.x - old.x) < 1e-4 and abs(new.y - old.y) < 1e-4 for new, old in zip(new_centres, centres)):
            print("Convergence atteinte!")
            break

        centres = new_centres 

        plt.pause(3)  # Pause de 3 secondes

    plt.show()


# Exemple d'utilisation avec des points supplémentaires
points = [Point(random.uniform(0, 10), random.uniform(0, 10)) for _ in range(100)]
points2 = [[random.uniform(0, 10), random.uniform(0, 10)] for _ in range(100)]


# Centres initiaux pour le k-means
centres = [
    Point(2, 2), Point(5, 5), Point(8, 3), Point(10, 10)
]

# Appel de la fonction kmeans avec la liste étendue de points
Kmeans(points, centres)

# Appliquer K-means avec 3 clusters
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(points2)

# Afficher les résultats
plt.scatter([p[0] for p in points2], [p[1] for p in points2], c=kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', marker='X')
plt.title('K-means Clustering with scikit-learn')
plt.show()



