from matplotlib.image import imread, imsave
import matplotlib.pyplot as plt
import numpy as np

K = 16
ITERATION = 30


def main():
    image = imread('../data/peppers-large.tiff')
    print(f'image.shape{image.shape}')
    data = image.reshape(-1, 3)
    mu, z = initialize(data)
    pre_mu = None
    it = 0
    threshold = 1e-5
    while it < ITERATION and (pre_mu is None or ((pre_mu - mu) ** 2).sum() < threshold):
        pre_mu = mu.copy()
        z = iterate(data, mu)
    mu = mu.round().astype(np.uint8)
    compressed_image = mu[z]
    imsave('./output/p05_compressed.tiff', compressed_image.reshape(image.shape))



def initialize(image):
    m, n = image.shape
    mu = np.empty((K, 3))
    for i in range(0, K):
        mu[i] = image[(2 ** i) % m]
    z = np.zeros((K, 3))
    return mu, z


def iterate(image, mu):
    z = ((image - mu.reshape(K, 1, 3)) ** 2).sum(axis=2).argmin(axis=0)
    print(z.shape)
    for i in range(0, K):
        mu[i] = image[z == i, :].sum(axis=0) / (z == i).sum()
    return z


if __name__ == '__main__':
    main()







