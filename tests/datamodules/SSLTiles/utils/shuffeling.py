import numpy as np


def shuffle_horizontal(number_array: np.ndarray):
    # for each row
    for i in range(number_array.shape[0]):
        # get permutation of number of columns
        permutation = np.random.permutation(number_array.shape[1])
        # rearange colum in row
        number_array[i, :] = number_array[i, permutation]


def shuffle_vertical(number_array: np.ndarray):
    for i in range(number_array.shape[1]):
        permutation = np.random.permutation(number_array.shape[0])
        number_array[:, i] = number_array[permutation, i]
