import random
from math import sqrt, log
from pathlib import Path
from typing import Callable, Tuple, Dict

import numpy as np
from matplotlib import pyplot as plt

from stats import RandomVariableSamples
from util import timestamp


class Items:
    """
    Information that is needed allow a decision which feature should be evaluated next using HiCS.
    """

    def __init__(self, relevance_by_feature: Dict[str, RandomVariableSamples],
                 true_relevance_by_feature: Dict[str, float],
                 num_features_to_select: int, iteration: int, name: str):
        self.true_relevance_by_feature = true_relevance_by_feature
        self.num_features_to_select = num_features_to_select
        self.name = name

        self.iteration = iteration
        self.relevance_by_feature = relevance_by_feature  # type:Dict[str, RandomVariableSamples]
        self.sorted_features = sorted(list(relevance_by_feature.items()), key=lambda x: x[1].mean, reverse=True)
        self.selected = self.sorted_features[:num_features_to_select]
        self.non_selected = self.sorted_features[num_features_to_select:]

        self.true_relevances = sorted(self.true_relevance_by_feature.items(), key=lambda x: x[1], reverse=True)
        self.features = [feature for feature, _ in self.true_relevances]
        self.relevant_features = set(
            feature for feature, correlation in self.true_relevances[:self.num_features_to_select])
        self.num_selected_relevant_features = len(
            set(feature for feature, correlation in self.selected).intersection(self.relevant_features))

    def save_plot(self, color='red'):
        plt.ylabel('mutual information')
        plt.xlabel('feature')

        means = np.array([self.relevance_by_feature[feature].mean for feature in self.features])
        standard_deviations = np.array(
            [self.relevance_by_feature[feature].mean_as_gaussian.standard_deviation for feature in self.features])

        markersize = 2
        plt.plot(range(len(means)), [true_relevance for f, true_relevance in self.true_relevances], linestyle="None",
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

        plt.savefig(str(Path('.') / 'plots' / f'{timestamp()}.png'))
        plt.clf()


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

        return max(items.sorted_features, key=priority)

    return Strategy(choose_by_exploitation, name=f'exploit{exploitation}')
