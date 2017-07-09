import random
from math import log2, sqrt, log, exp
from pathlib import Path
from time import strftime
from typing import List, Tuple, Dict, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas
from hics.contrast_measure import HiCS
from numpy import ndarray
from numpy.random import randint, rand, shuffle
from pynverse import inversefunc

from stats import ValuesWithStats


def timestamp() -> str:
    return strftime('%Y%m%d-%H%M%S')


def code_length(p: float) -> float:
    if p == 0:
        return 0

    return -p * log2(p)


def mutual_information_of_partial_cause_feature(target_vs_noise: float = .5) -> float:
    return 2 - 2 * code_length(1 / 4 * (1 + target_vs_noise)) - 2 * code_length(1 / 4 * (1 - target_vs_noise))


target_vs_noise_for_partial_cause_feature = inversefunc(mutual_information_of_partial_cause_feature, domain=[0, 1],
                                                        image=[0, 1])


class Items:
    def __init__(self, mutual_by_feature: Dict[str, ValuesWithStats], true_correlation_distribution: List[float],
                 num_features_to_select: int, iteration: int, name: str):
        self.name = name
        self.true_correlation_distribution = true_correlation_distribution
        self.iteration = iteration
        self.items = list(mutual_by_feature.items())  # type:List[Tuple[str, ValuesWithStats]]
        self.sorted_features = sorted(self.items, key=lambda x: x[1].average, reverse=True)
        self.selected = self.sorted_features[:num_features_to_select]
        self.non_selected = self.sorted_features[num_features_to_select:]
        self.selected_relevant_feature_count = len([s for s in self.selected if int(s[0]) < num_features_to_select])
        self.items_by_index = sorted(self.items, key=lambda x: int(x[0]))

    def show_plot(self, color='red'):
        plt.plot(self.true_correlation_distribution, color='blue')

        plt.ylabel('mutual information')
        plt.xlabel('feature')


        average = np.array([value.average for feature, value in self.items_by_index])
        deviation = np.array([value.average_as_gaussian.standard_deviation for feature, value in self.items_by_index])

        plt.fill_between(range(len(average)),
                         list(average - deviation),
                         list(average + deviation),
                         color=color, alpha=.15)
        plt.plot(average, label=self.name, c=color)
        plt.show()


class Strategy:
    def __init__(self, choose: Callable[[Items], Tuple[str, ValuesWithStats]], name: str):
        self.name = name
        self.choose = choose


def random_strategy():
    def choose_random(items: Items) -> Tuple[str, ValuesWithStats]:
        return random.choice(items)

    return Strategy(choose_random, name='random')


def gaussian_strategy():
    def choose_by_gaussian(items: Items) -> Tuple[str, ValuesWithStats]:
        selected_nonselected_pairs = [(s, n) for n in items.non_selected for s in items.selected]

        def loss(pair):
            s, n = pair

            return (
            n[1].average_as_gaussian - s[1].average_as_gaussian).expected_value_if_truncated_of_negative_mapped_to_0

        s, n = max(selected_nonselected_pairs, key=loss)

        return s if s[1].variance_of_average > n[1].variance_of_average else n

    return Strategy(choose_by_gaussian, name='gaussian')


def exploitation_strategy(exploitation: float):
    def choose_by_exploitation(items: Items) -> Tuple[str, ValuesWithStats]:
        def priority(x):
            feature, stats = x

            return stats.average * exploitation + sqrt(log(items.iteration + 1) / (stats.count + 1))

        return max(items.items, key=priority)

    return Strategy(choose_by_exploitation, name=f'exploit{exploitation}')


def test(strategies: List[Strategy], num_features_to_select=10, total_feature_count=100, iterations=1000, runs=1):
    correlation_distribution = plateau_distribution(total_feature_count)

    def run_hics(data, strategy: Strategy) -> List[float]:
        features_with_target = data.columns.values  # type:List[str]
        features = list(filter(lambda i: i != 'target', features_with_target))
        hics = HiCS(data, alpha=.001, iterations=1, categorical_features=features_with_target)

        mutual_by_feature = dict(
            [(feature, ValuesWithStats()) for feature in features])  # type:Dict[str, ValuesWithStats]

        selected_relevant_feature_counts = []

        for iteration in range(iterations):
            items = Items(mutual_by_feature=mutual_by_feature, num_features_to_select=num_features_to_select,
                          iteration=iteration, true_correlation_distribution=correlation_distribution,
                          name=strategy.name)

            if iteration % 100 == 0:
                items.show_plot()

            selected_relevant_feature_counts.append(items.selected_relevant_feature_count)

            feature, value = strategy.choose(items)

            print(f"Iteration {iteration}, chosen relevant features: {items.selected_relevant_feature_count}")

            mutual_by_feature[feature].append(hics.calculate_contrast([feature], 'target'))

        return selected_relevant_feature_counts

    def plot_average_and_deviation_by_time(strategy: Strategy, data_for_runs, color):
        mutual_information_by_run_by_time = np.array(
            [run_hics(data=data, strategy=strategy) for data in data_for_runs])
        average = np.average(mutual_information_by_run_by_time, axis=0)
        deviation = np.std(mutual_information_by_run_by_time, axis=0)

        plt.fill_between(range(len(average)),
                         list(average - deviation),
                         list(average + deviation),
                         color=color, alpha=.15)
        plt.plot(average, label=strategy.name, c=color)

    data_for_runs = [generate_data(correlation_distribution=correlation_distribution) for _ in range(runs)]

    plt.ylabel('chosen relevant features')
    plt.xlabel('iteration')

    colors = plt.cm.rainbow(np.linspace(0, 1, len(strategies)))
    # TODO plot_average_and_deviation_by_time(steered=None, label="Random")
    for strategy, color in zip(strategies, colors):
        plot_average_and_deviation_by_time(strategy=strategy, data_for_runs=data_for_runs, color=color)

    plt.legend(loc=2)

    title = f"{total_feature_count}features_{runs}runs"
    plt.title(title)

    directory = Path(".") / "plots"
    directory.mkdir(exist_ok=True)
    fig = plt.gcf()
    fig.savefig(str(directory / f"{timestamp()}_{title}.svg"))

    plt.show()


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


def plateau_distribution(total_feature_count: int) -> List[float]:
    return [((1 - x / 10) if x < .3 else exp(-10 * x)) for x in np.linspace(0, 1, num=total_feature_count)]


def two_relevant_features_distribution(total_feature_count: int) -> List[float]:
    return [1, 0.95] + ([0] * (total_feature_count - 2))


if __name__ == '__main__':
    exploit_strats = [exploitation_strategy(exploitation) for exploitation in (0, .5, 1, 1.5, 2, 2.5, 3)]

    test(strategies=[gaussian_strategy(), exploitation_strategy(1.5), exploitation_strategy(0)])
