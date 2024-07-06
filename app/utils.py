import numpy as np


##Упорядочевает координаты баудинг бокса
def normalized_coords(coords):
    ind1 = np.argmin(coords[:, 0])
    first = np.array(coords[ind1])
    coords = np.delete(coords, ind1, axis=0)

    ind2 = np.argmin(coords[:, 0])
    first = np.append(first, coords[ind2])
    first = first.reshape((2, 2))
    coords = np.delete(coords, ind2, axis=0)

    second = coords
    ind1 = np.argmin(first[:, 1])

    x1 = first[ind1]
    first = np.delete(first, ind1, axis=0)

    y1 = first[0]
    ind2 = np.argmin(second[:, 1])
    x2 = second[ind2]
    second = np.delete(second, ind2, axis=0)
    y2 = second[0]

    return x1, y1, x2, y2