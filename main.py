from math import log2, sqrt, log
from typing import Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy
import pandas
from hics.contrast_measure import HiCS
from numpy import ndarray
from numpy.random import randint, rand, shuffle, random
from random import choice
from pynverse import inversefunc


def code_length(p: float) -> float:
    if p == 0:
        return 0

    return -p * log2(p)


def mutual_information_of_partial_cause_feature(target_vs_noise: float = .5) -> float:
    return 2 - 2 * code_length(1 / 4 * (1 + target_vs_noise)) - 2 * code_length(1 / 4 * (1 - target_vs_noise))


target_vs_noise_for_partial_cause_feature = inversefunc(mutual_information_of_partial_cause_feature, domain=[0, 1],
                                                        image=[0, 1])


def test(relevant_feature_count=10, total_feature_count=200, iterations=500):
    data = test_dataset(relevant_feature_count=relevant_feature_count,
                        noise_feature_count=total_feature_count - relevant_feature_count)
    features_with_target = data.columns.values  # type:List[str]
    features = list(filter(lambda i: i != 'target', features_with_target))
    shuffle(features)
    hics = HiCS(data, alpha=.001, iterations=1, categorical_features=features_with_target)

    class ValuesWithAverage:
        def __init__(self):
            self.values = []
            self.sum = 0.0

        def append(self, value: float):
            self.values.append(value)
            self.sum += value

        @property
        def average(self):
            l = len(self.values)
            if (l == 0):
                return 0

            return self.sum / l

        def partial_averages(self) -> Iterable[float]:
            sum = 0
            for index, value in enumerate(self.values):
                sum += value
                count = index + 1
                yield sum / count

    def run_hics(steered: Optional[bool] = True, steered_exploration_weight=1) -> List[float]:
        mutual_by_feature = dict([(feature, ValuesWithAverage()) for feature in features])

        chosen_relevant_feature_counts = []

        for iteration in range(iterations):
            items = list(mutual_by_feature.items())
            chosen = sorted(items, key=lambda x: x[1].average, reverse=True)[:relevant_feature_count]
            chosen_relevant_feature_count = len([c for c in chosen if c[0].startswith("mutual")])
            chosen_relevant_feature_counts.append(chosen_relevant_feature_count)

            def priority(x):
                return (x[1].average if steered else 0) + steered_exploration_weight * sqrt(
                    log(iteration + 1) / (len(x[1].values) + 1))

            feature, value = max(items, key=priority) if steered is not None else choice(items)

            print(
                f"Iteration {iteration}, chosen relevant features: {chosen_relevant_feature_count}, average {value.average:0.3f} in priority {priority((feature, value)):0.3f}.")

            mutual_by_feature[feature].append(hics.calculate_contrast([feature], 'target'))

        return chosen_relevant_feature_counts

    plt.ylabel("chosen relevant features")
    plt.xlabel("iteration")
    # plt.plot(run_hics(steered=None), label="Random")
    plt.plot(run_hics(steered=True), label="Steered")
    plt.plot(run_hics(steered=False), label="Flat")
    plt.legend()
    plt.show()


def test_dataset(relevant_feature_count: int, noise_feature_count: int, dataset_size=10000,
                 mutual_information=.5, slight_mutual_information=.45, noise_mutual_information=0):
    def combination_count(bits: int) -> int:
        return 1 << bits

    def random_values(bits: int, count: int = dataset_size) -> ndarray:
        return randint(0, combination_count(bits=bits), count)

    def random_merged(values1: ndarray, values2: ndarray, values1_share: float = .5):
        return numpy.where(rand(len(values1)) < values1_share, values1, values2)

    def joined_with_noise(values: ndarray, noise_bits: int):
        return [f"{value},{extra_noise}" for value, extra_noise in zip(values, random_values(bits=noise_bits))]

    target = random_values(bits=1)

    def feature(mutual_information: float):
        target_share = target_vs_noise_for_partial_cause_feature(mutual_information)
        return joined_with_noise(random_merged(target, random_values(bits=1), target_share), noise_bits=3)

    features = {'target': target}
    for i in range(int(noise_feature_count * 8 / 10)):
        features[f"noise-{i}"] = feature(noise_mutual_information)

    for i in range(int(noise_feature_count * 2 / 10)):
        features[f"noise2-{i}"] = feature(slight_mutual_information)

    target_share = target_vs_noise_for_partial_cause_feature(mutual_information)
    for i in range(relevant_feature_count):
        features[f"mutual-{i}"] = joined_with_noise(random_merged(target, random_values(bits=1), target_share),
                                                    noise_bits=3)

    return pandas.DataFrame(features)


test()
