import random
from math import sqrt, log
from typing import Callable, Tuple, Dict, List

import numpy as np
from matplotlib import pyplot as plt

from stats import RandomVariableSamples


class Items:
    """
    Information that is needed allow a decision which feature should be evaluated next using HiCS.
    """
    def __init__(self, mutual_by_feature: Dict[str, RandomVariableSamples], true_correlation_distribution: List[float],
                 num_features_to_select: int, iteration: int, name: str):
        self.name = name
        self.true_correlation_distribution = true_correlation_distribution
        self.iteration = iteration
        self.items = list(mutual_by_feature.items())  # type:List[Tuple[str, RandomVariableSamples]]
        self.sorted_features = sorted(self.items, key=lambda x: x[1].mean, reverse=True)
        self.selected = self.sorted_features[:num_features_to_select]
        self.non_selected = self.sorted_features[num_features_to_select:]
        self.selected_relevant_feature_count = len([s for s in self.selected if int(s[0]) < num_features_to_select])
        self.items_by_index = sorted(self.items, key=lambda x: int(x[0]))

    def show_plot(self, color='red'):
        plt.ylabel('mutual information')
        plt.xlabel('feature')

        means = np.array([value.mean for feature, value in self.items_by_index])
        standard_deviations = np.array([value.mean_as_gaussian.standard_deviation for feature, value in self.items_by_index])

        markersize = 2
        plt.plot(range(len(means)), self.true_correlation_distribution, linestyle="None",
                 marker="o",
                 markersize=markersize,
                 color='blue')

        standard_deviations[standard_deviations == np.inf] = .5

        plt.errorbar(range(len(means)),
                     list(means),
                     list(standard_deviations),
                     linestyle="None",
                     elinewidth=markersize / 2,
                     ecolor='red',
                     marker="o",
                     markersize=markersize,
                     color=color)

        plt.show()


class Strategy:
    """
    A decision strategy for which feature should be evaluated next using HiCS.
    """
    def __init__(self, choose: Callable[[Items], Tuple[str, RandomVariableSamples]], name: str):
        self.name = name
        self.choose = choose


def random_strategy():
    def choose_random(items: Items) -> Tuple[str, RandomVariableSamples]:
        return random.choice(items)

    return Strategy(choose_random, name='random')


def gaussian_strategy():
    def choose_by_gaussian(items: Items) -> Tuple[str, RandomVariableSamples]:
        selected_nonselected_pairs = [(s, n) for n in items.non_selected for s in items.selected]

        def loss(pair):
            s, n = pair

            return (
                n[1].mean_as_gaussian - s[1].mean_as_gaussian).expected_value_if_truncated_of_negative_mapped_to_0

        s, n = max(selected_nonselected_pairs, key=loss)

        return s if s[1].variance_of_mean > n[1].variance_of_mean else n

    return Strategy(choose_by_gaussian, name='gaussian')


def exploitation_strategy(exploitation: float):
    def choose_by_exploitation(items: Items) -> Tuple[str, RandomVariableSamples]:
        def priority(x):
            feature, stats = x

            return stats.mean * exploitation + sqrt(log(items.iteration + 1) / (stats.count + 1))

        return max(items.items, key=priority)

    return Strategy(choose_by_exploitation, name=f'exploit{exploitation}')
