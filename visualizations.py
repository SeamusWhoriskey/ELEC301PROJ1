import numpy as np
import matplotlib.pyplot as plt


def show_onb(dk):

    print(dk.shape)
    x = np.arange(dk.shape[0])
    plt.stem(x, dk)
    plt.show()


