import numpy as np


def _derangement(_i):
    while any((perm := np.random.permutation(_i)) == _i):
        pass
    return perm


def derangement(*shape):
    """
    Return a random derangement array of shape which permutes the last axis.
    """
    return np.apply_along_axis(func1d=_derangement, axis=-1, arr=np.broadcast_to(np.arange(shape[-1]), shape))
