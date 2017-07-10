from math import exp, log2
from typing import List

import numpy as np
import pandas
from matplotlib import pyplot as plt
from numpy import ndarray
from numpy.random.mtrand import randint, rand, shuffle
from pynverse import inversefunc


def generate_data(correlation_distribution: List[float], dataset_size=10000,
                  show_distribution: bool = False):
    if show_distribution:
        plt.plot(correlation_distribution)
        plt.show()

    def combination_count(bits: int) -> int:
        return 1 << bits

    def random_values(bits: int, count: int = dataset_size) -> ndarray:
        return randint(0, combination_count(bits=bits), count)

    def random_merged(values1: ndarray, values2: ndarray, values1_share: float = .5):
        return np.where(rand(len(values1)) < values1_share, values1, values2)

    def joined_with_noise(values: ndarray, noise_bits: int):
        return [f"{value},{extra_noise}" for value, extra_noise in zip(values, random_values(bits=noise_bits))]

    target = random_values(bits=1)

    def feature(mutual_information: float):
        target_share = target_vs_noise_for_partial_cause_feature(mutual_information)
        return joined_with_noise(random_merged(target, random_values(bits=1), target_share), noise_bits=3)

    features = {'target': target}

    for i in range(len(correlation_distribution)):
        features[str(i)] = feature(correlation_distribution[i])

    df = pandas.DataFrame(features)

    shuffled = list(df.columns)
    shuffle(shuffled)
    return df.reindex_axis(shuffled, axis=1)


def plateau_distribution(total_feature_count: int, relevant_mutual_information: float = 1.0) -> List[float]:
    return [((relevant_mutual_information - x / 10) if x < .3 else exp(-10 * x)) for x in
            np.linspace(0, 1, num=total_feature_count)]


def code_length(p: float) -> float:
    if p == 0:
        return 0

    return -p * log2(p)


def mutual_information_of_partial_cause_feature(target_vs_noise: float = .5) -> float:
    return 2 - 2 * code_length(1 / 4 * (1 + target_vs_noise)) - 2 * code_length(1 / 4 * (1 - target_vs_noise))


target_vs_noise_for_partial_cause_feature = inversefunc(mutual_information_of_partial_cause_feature, domain=[0, 1],
                                                        image=[0, 1])


def two_relevant_features_distribution(total_feature_count: int) -> List[float]:
    return [1, 0.95] + ([0] * (total_feature_count - 2))
