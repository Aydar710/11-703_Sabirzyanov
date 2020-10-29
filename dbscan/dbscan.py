import math
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs
import seaborn as sns

blobs_X, blobs_y = make_blobs(n_samples=100, centers=3, n_features=2, cluster_std=2, random_state=42)


def calc_neighbors(collection, point, distFunc, eps):
    neighbors = []
    for i in range(len(collection)):
        if distFunc(collection[i], point) < eps:
            neighbors.append(i)

    return neighbors


def distance(point1, point2):
    sum = 0
    for i in range(len(point1)):
        sum += (point2[i] - point1[i]) ** 2
    return math.sqrt(sum)


def dbscan(x, eps, minPts, distFunc="euclidean"):
    if distFunc == "euclidean":
        distFunc = distance

    labels = [None] * len(x)
    clusters = 0

    for i in range(len(x)):
        if labels[i] is not None:
            continue

        neighbors = calc_neighbors(x, x[i], distFunc, eps)

        if len(neighbors) < minPts:
            labels[i] = -1
            continue

        labels[i] = clusters

        j = 0
        while j < len(neighbors):
            p = neighbors[j]
            if labels[p] == -1:
                labels[p] = clusters

            if labels[p] is not None:
                j += 1
                continue

            labels[p] = clusters
            new_neighbors = calc_neighbors(x, x[p], distFunc, eps)
            if len(new_neighbors) >= minPts:
                for n in new_neighbors:
                    if n not in neighbors:
                        neighbors.append(n)
            j += 1
        clusters += 1
    return labels


clusters = dbscan(blobs_X, eps=2, minPts=5)
sns.scatterplot(blobs_X[:, 0], blobs_X[:, 1], hue=clusters)
plt.show()
