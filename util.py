import numpy as np


def se(x, y):
    return (x - y) ** 2


def mse(x, y):
    return se(x, y).mean()


def sse(x, y):
    return se(x, y).sum()