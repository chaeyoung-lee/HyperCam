"""
Binary Spatter Code (BSC) Implementation
"""
import numpy as np


class BSC:
    @classmethod
    def bind(cls, xs):
        return np.bitwise_xor.reduce(xs)

    @classmethod
    def bundle(cls, xs):
        return 1 * (np.sum(xs, axis=0) >= (len(xs) / 2))

    @classmethod
    def bundle_with_weights(cls, xs, weights):
        return 1 * (np.matmul(weights, xs) >= (np.sum(weights) / 2))

    @classmethod
    def permute(cls, x, i):
        return np.roll(x, i)

    @classmethod
    def distance(cls, x1, x2):
        return np.divide(np.sum(np.bitwise_xor(x1, x2)), len(x1))

    @classmethod
    def wta(cls, hv_image, table):
        result = []
        for i in range(len(table)):  # i = class label
            result.append(BSC.distance(table[i], hv_image))
        return np.argmin(result)
