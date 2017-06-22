from math import log2
from typing import Iterable

import matplotlib.pyplot as plt
import numpy
import pandas
from hics.contrast_measure import HiCS
from numpy import ndarray
from numpy.random import randint, rand, choice, shuffle
from pynverse import inversefunc


def code_length(p: float) -> float:
    if p == 0:
        return 0

    return -p * log2(p)


def mutual_information_of_partial_cause_feature(target_vs_noise: float = .5) -> float:
    return 2 - 2 * code_length(1 / 4 * (1 + target_vs_noise)) - 2 * code_length(1 / 4 * (1 - target_vs_noise))


target_vs_noise_for_partial_cause_feature = inversefunc(mutual_information_of_partial_cause_feature, domain=[0, 1],
                                                        image=[0, 1])


def test(relevant_feature_count=10, total_feature_count=1000, iterations=1000):
    data = test_dataset(relevant_feature_count=relevant_feature_count, noise_feature_count=total_feature_count - relevant_feature_count)
    features_with_target = data.columns.values
    features = list(filter(lambda i: i != 'target', features_with_target))
    shuffle(features)
    hics = HiCS(data, alpha=.001, iterations=1, categorical_features=features_with_target)

    class ValuesWithAverage:
        def __init__(self):
            self.values = []
            self.average = 0

        def append(self, value: float):
            self.values.append(value)
            self.average = (self.average + value) / (len(self.values) + 1)

        def partial_averages(self) -> Iterable[float]:
            sum = 0
            for index, value in enumerate(self.values):
                sum += value
                count = index + 1
                yield sum / count

    def run_hics():
        mutual_by_feature = dict([(feature, ValuesWithAverage()) for feature in features])

        chosen_relevant_feature_counts = []

        for iteration in range(iterations):
            feature = choice(features)
            mutual_by_feature[feature].append(hics.calculate_contrast([feature], 'target'))
            items = list(mutual_by_feature.items())
            chosen = sorted(items, key=lambda x: x[1].average, reverse=True)[:relevant_feature_count]
            chosen_relevant_feature_count = len([c for c in chosen if c[0].startswith("mutual")])
            chosen_relevant_feature_counts.append(chosen_relevant_feature_count)
            print(f"Iteration {iteration}, chosen relevant features: {chosen_relevant_feature_count}.")

        plt.ylabel("relevant chosen features")
        plt.xlabel("iteration")
        plt.plot(chosen_relevant_feature_counts)
        plt.show()

    run_hics()


def test_dataset(relevant_feature_count: int, noise_feature_count: int, dataset_size=10000,
                 mutual_information=.1):
    def combination_count(bits: int) -> int:
        return 1 << bits

    def random_values(bits: int, count: int = dataset_size) -> ndarray:
        return randint(0, combination_count(bits=bits), count)

    def random_merged(values1: ndarray, values2: ndarray, values1_share: float = .5):
        return numpy.where(rand(len(values1)) < values1_share, values1, values2)

    def joined_with_noise(values: ndarray, noise_bits: int):
        return [f"{value},{extra_noise}" for value, extra_noise in zip(values, random_values(bits=noise_bits))]

    target = random_values(bits=1)
    features = {'target': target}
    for i in range(noise_feature_count):
        features[f"noise-{i}"] = random_values(bits=1)
    target_share = target_vs_noise_for_partial_cause_feature(mutual_information)
    for i in range(relevant_feature_count):
        merged = random_merged(target, random_values(bits=1), target_share)
        features[f"mutual-{i}"] = joined_with_noise(merged, noise_bits=3)

    return pandas.DataFrame(features)


test()
