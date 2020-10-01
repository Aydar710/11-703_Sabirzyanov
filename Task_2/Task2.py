import matplotlib.pyplot as plt
import numpy as np
import requests


def draw(x, y, x_cc, y_cc):
    for i in range(0, len(x_cc)):
        plt.scatter(x, y)
    plt.scatter(x_cc, y_cc)
    plt.show()


def calculate_clusters(x_cc, y_cc, x, y):
    clusters = []
    for i in range(0, n):
        r = dist(x_cc[0], y_cc[0], x[i], y[i])
        for j in range(0, k):
            if r > dist(x_cc[j], y_cc[j], x[i], y[i]):
                numb = j
                r = dist(x_cc[j], y_cc[j], x[i], y[i])
                clusters.append(numb)
    return clusters


def dist(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


resp = requests.get('http://data.kzn.ru:8082/api/v0/dynamic_datasets/bus.json')
response_jsons = resp.json()
random_index = np.random.randint(0, len(response_jsons))
x = response_jsons[random_index].get('data')
print(x)

n, k = 100, 4
x = np.random.randint(1, 100, n)
y = np.random.randint(1, 100, n)

x_c = np.mean(x)
y_c = np.mean(y)
R = 0

for i in range(0, n):
    if dist(x_c, y_c, x[i], y[i]) > R:
        R = dist(x_c, y_c, x[i], y[i])

x_cc = [R * np.cos(2 * np.pi * i / k) + x_c for i in range(k)]
y_cc = [R * np.sin(2 * np.pi * i / k) + y_c for i in range(k)]

clusters = calculate_clusters(x_cc, y_cc, x, y)

draw(x, y, x_cc, y_cc)
