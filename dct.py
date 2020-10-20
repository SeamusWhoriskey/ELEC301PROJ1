import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.image as im

pi = math.pi


def fDCT(image, onb):

    return np.matmul(onb, image)


def compress_image_DCT(image, thresh, onb):
    out = np.zeros(image.shape)
    for n in range(image.shape[2]):
        im_c = fDCT(image[:, :, n], onb)
        im_c[(im_c ** 2) ** 0.5 < thresh] = 0
        out[:, :, n] = im_c
    return out


def iDCT(compressed_image, onb):

    return np.matmul(onb, compressed_image)


def decompress_image_DCT(compressed_image, onb):
    out = np.zeros(compressed_image.shape)
    for n in range(compressed_image.shape[2]):
        out[:, :, n] = iDCT(compressed_image[:, :, n], onb)
    return out


def create_onb(N):
    d = np.zeros((N, N))
    for k in range(N):
        for n in range(N):
            d[n, k] = (math.cos(pi/N * (n+0.5) * k))
        norm = np.linalg.norm(d[:, k])
        if norm != 0:
            d[:, k] = d[:, k]/norm

    return d


def show_onb(dk):

    print(dk.shape)
    x = np.arange(dk.shape[0])
    plt.stem(x, dk)
    plt.show()


def show_compression(image, t):
    onb = create_onb(image.shape[0])
    x = compress_image_DCT(image, t, onb)
    y = decompress_image_DCT(x, onb.conjugate())
    plt.subplot(121)
    plt.title("Original image")
    plt.imshow(image, interpolation='nearest', cmap='gray')
    plt.subplot(122)
    plt.title("Image after compression with threshold = " + str(t))
    plt.imshow(y, interpolation='nearest', cmap='gray')
    plt.show()


show_compression(im.imread('test_images/richb8.png'), 0.01)
