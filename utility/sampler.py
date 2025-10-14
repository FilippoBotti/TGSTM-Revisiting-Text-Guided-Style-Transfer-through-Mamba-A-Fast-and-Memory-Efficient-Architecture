import numpy as np
from torch.utils import data


def InfiniteSampler(n):
    # i = 0
    i = 0
    order = np.arange(n)   # fixed order, no randomness
    while True:
        yield order[i]
        i += 1
        if i >= n:
            i = 0


class InfiniteSamplerWrapper(data.sampler.Sampler):
    def __init__(self, data_source):
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(InfiniteSampler(self.num_samples))

    def __len__(self):
        return 2 ** 31
