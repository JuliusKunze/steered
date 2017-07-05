from math import log2, sqrt, log, exp
from pathlib import Path
from random import choice
from time import strftime
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas
from hics.contrast_measure import HiCS
from numpy import ndarray
from numpy.random import randint, rand, shuffle
from pynverse import inversefunc


def code_length(p: float) -> float:
    if p == 0:
        return 0

    return -p * log2(p)


def mutual_information_of_partial_cause_feature(target_vs_noise: float = .5) -> float:
    return 2 - 2 * code_length(1 / 4 * (1 + target_vs_noise)) - 2 * code_length(1 / 4 * (1 - target_vs_noise))


target_vs_noise_for_partial_cause_feature = inversefunc(mutual_information_of_partial_cause_feature, domain=[0, 1],
                                                        image=[0, 1])


def timestamp() -> str:
    return strftime('%Y%m%d-%H%M%S')


def test(relevant_feature_count=10, total_feature_count=200, iterations=1000, runs=10, steered_exploration_weight=1):
    correlation_distribution = plateau_distribution(total_feature_count)
    data = generate_data(relevant_feature_count=relevant_feature_count,
                         correlation_distribution=correlation_distribution)
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
            if l == 0:
                return 0

            return self.sum / l

    def run_hics(steered: Optional[bool] = True, steered_exploration_weight=steered_exploration_weight) -> List[float]:
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
                f"Iteration {iteration}, chosen relevant features: {chosen_relevant_feature_count}, "
                f"average {value.average:0.3f} in priority {priority((feature, value)):0.3f}.")

            mutual_by_feature[feature].append(hics.calculate_contrast([feature], 'target'))

        return chosen_relevant_feature_counts

    plt.ylabel("chosen relevant features")
    plt.xlabel("iteration")

    # plt.plot(run_hics(steered=None), label="Random")

    def plot_average_and_deviation_by_time(steered: bool):
        color = "b" if steered else "r"
        label = "Steered" if steered else "Flat"

        mutual_information_by_run_by_time = np.array([run_hics(steered=steered) for _ in range(runs)])
        average = np.average(mutual_information_by_run_by_time, axis=0)
        deviation = np.std(mutual_information_by_run_by_time, axis=0)

        plt.fill_between(range(len(average)),
                         list(average - deviation),
                         list(average + deviation),
                         color=color, alpha=.3)
        plt.plot(average, label=label, color=color)

    plot_average_and_deviation_by_time(steered=True)
    plot_average_and_deviation_by_time(steered=False)

    plt.legend()

    directory = Path(".") / "plots"
    directory.mkdir(exist_ok=True)
    fig = plt.gcf()
    fig.savefig(str(directory / (timestamp() + ".png")))

    plt.show()


def generate_data(relevant_feature_count: int, correlation_distribution: List[float], dataset_size=10000,
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
        features[f"{'mutual' if i < relevant_feature_count else 'noise'}-{i}"] = feature(
            correlation_distribution[i])

    return pandas.DataFrame(features)


def plateau_distribution(total_feature_count: int) -> List[float]:
    return [((1 - x / 10) if x < .3 else exp(-10 * x)) for x in np.linspace(0, 1, num=total_feature_count)]


def two_relevant_features_distribution(total_feature_count: int) -> List[float]:
    return [1, 0.95] + ([0] * (total_feature_count - 2))


test()
