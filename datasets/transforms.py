import numpy as np
import torch


class NoiseTransform(object):
    
    def __init__(self, noise_scale):
        self.noise_scale = noise_scale

    def __call__(self, sample):
        morlet = sample
        return morlet + np.random.normal(0, self.noise_scale, morlet.shape)


class ToTensor(object):

    def __call__(self, sample):
        morlet = sample
        return torch.from_numpy(morlet.copy())