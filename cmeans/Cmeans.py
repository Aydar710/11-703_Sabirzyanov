import numpy as np
import matplotlib.pyplot as plt

n = 88
x = np.random.randint(1, 100, n)
y = np.random.randint(1, 100, n)

x_c = np.mean(x)
y_c = np.mean(y)


def dist(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def c_means(k, max_iter):
    m, eps = 2, 0.2
    u = np.zeros([n, k])
    u_new = np.zeros([n, k])

    def check(clusters, k):
        for i in range(0, len(clusters)):
            for j in range(0, k):
                if clusters[i] == j:
                    u[i][j] = 1
                else:
                    u[i][j] = 0

    def clusterize(x_cc, y_cc, x, y):
        for i in range(0, n):
            r = dist(x_cc[0], y_cc[0], x[i], y[i])
            numb = 0
            for j in range(0, k):
                if r < dist(x_cc[j], y_cc[j], x[i], y[i]):
                    numb = j
                    r = dist(x_cc[j], y_cc[j], x[i], y[i])
                if j == k - 1:
                    clust[i] = numb

    R = 0
    for i in range(0, n):
        if dist(x_c, y_c, x[i], y[i]) > R:
            R = dist(x_c, y_c, x[i], y[i])

    x_cc = [R * np.cos(2 * np.pi * i / k) + x_c for i in range(k)]
    y_cc = [R * np.sin(2 * np.pi * i / k) + y_c for i in range(k)]

    clust = [0] * n

    clusterize(x_cc, y_cc, x, y)
    check(clust, k)

    for dik in range(max_iter):
        u_sum = [0] * n
        for i in range(0, n):
            for j in range(0, k):
                u_sum[i] += u[i][j]

        for i in range(0, n):
            for j in range(0, k):
                sum = 0
                for l in range(0, k):
                    if clust[i] == l:
                        sum += (dist(x_cc[j], y_cc[j], x[i], y[i]) / dist(x_cc[l], y_cc[l], x[i], y[i])) ** (
                                2 / (m - 1))
                u_new[i][j] = 1 / sum

        for i in range(0, k):
            x_downSum = 0
            x_upSum = 0
            y_downSum = 0
            y_upSum = 0
            for j in range(0, n):
                if clust[j] == i:
                    x_downSum += (u_sum[j]) ** m
                    x_upSum += (x[j] * ((u_sum[j]) ** m))

                    y_downSum += (u_sum[j]) ** m
                    y_upSum += (y[j] * ((u_sum[j]) ** m))
            x_cc[i] = x_upSum / x_downSum
            y_cc[i] = y_upSum / y_downSum

        f = True
        for i in range(0, n):
            for j in range(0, k):
                if u[i][j] - u_new[i][j] < eps:
                    f = False
        if f:
            break

        for i in range(0, n):
            for j in range(0, k):
                u[i][j] = u_new[i][j]

    print(u)
    plt.scatter(x, y, color='orange')
    plt.scatter(x_cc, y_cc, color='r')
    plt.show()


c_means(3, 5)
