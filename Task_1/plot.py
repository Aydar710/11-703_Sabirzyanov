import pandas
import matplotlib.pyplot as plt
import numpy as np


def read_csv():
    return pandas.read_csv("HP.csv")


def draw(dataset):
    y_pos = np.arange(len(dataset.name))

    plt.barh(y_pos, dataset.age, align='center', alpha=0.5)
    plt.yticks(y_pos, dataset.name)
    plt.xlabel('age')
    plt.title('Harry Potter character ages')

    plt.show()


data = read_csv()
draw(data)
