from math import log2
from typing import Iterable

import numpy
import pandas
from hics.contrast_measure import HiCS
from numpy import ndarray
from numpy.random import randint, rand
from pynverse import inversefunc
import matplotlib.pyplot as plt

def test():
    dataset_size = 10000

    def code_length(p: float) -> float:
        if p == 0:
            return 0

        return -p * log2(p)

    def mutual_information_of_partial_cause_feature(target_vs_noise: float = .5) -> float:
        return 2 - 2 * code_length(1 / 4 * (1 + target_vs_noise)) - 2 * code_length(1 / 4 * (1 - target_vs_noise))

    target_vs_noise_for_partial_cause_feature = inversefunc(mutual_information_of_partial_cause_feature, domain=[0, 1], image=[0, 1])

    def combination_count(bits: int) -> int:
        return 1 << bits

    def random_values(bits: int, count: int = dataset_size) -> ndarray:
        return randint(0, combination_count(bits=bits), count)

    def random_merged(values1: ndarray, values2: ndarray, values1_share: float = .5):
        return numpy.where(rand(len(values1)) < values1_share, values1, values2)

    def partial_averages(values: Iterable[float]) -> Iterable[float]:
        sum = 0
        for index, value in enumerate(values):
            sum += value
            count = index + 1
            yield sum / count

    def joined_with_noise(values: ndarray, noise_bits: int):
        return [f"{value},{extra_noise}" for value, extra_noise in zip(values, random_values(bits=noise_bits))]

    target = random_values(bits=1)
    noise = random_values(bits=1)
    cause = target

    mutual_information = .1
    target_share = target_vs_noise_for_partial_cause_feature(mutual_information)

    feature_count = 1000

    merged = random_merged(values1=target, values2=noise, values1_share=target_share)
    partial_cause = joined_with_noise(merged, noise_bits=3)

    features = {'target': target, 'cause': cause, 'noise': noise, 'partial_cause': partial_cause}

    data = pandas.DataFrame(features)
    hics = HiCS(data, alpha=.001, iterations=1, categorical_features=['target', 'cause', 'noise', 'partial_cause'])

    partial_cause_contrasts = []
    for iteration in range(100):
        partial_cause_contrasts.append(hics.calculate_contrast(['partial_cause'], 'target'))

    plt.ylabel("estimated mutual information / bits")
    plt.xlabel("iteration")
    plt.plot(list(partial_averages(partial_cause_contrasts)))
    plt.show()

    cause_contrast = hics.calculate_contrast(['cause'], 'target')
    noise_contrast = hics.calculate_contrast(['noise'], 'target')

    print(f"{partial_cause_contrasts[-1]} should be {mutual_information}.")
    print(f"{cause_contrast} should be 1.")
    print(f"{noise_contrast} should be 0.")

test()
