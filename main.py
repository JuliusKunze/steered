from pathlib import Path
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
from hics.contrast_measure import HiCS

from data import generate_data, linearly_relevant_features
from stats import RandomVariableSamples
from strategy import Strategy, exploitation_strategy, Items, gaussian_strategy
from util import timestamp

plt.rcParams["figure.figsize"] = (19.20 / 2, 10.80 / 2)


def main(relevance_distribution: List[float], strategies: List[Strategy], num_features_to_select=10,
         iterations=500, runs=20, alpha=.001):
    def run_hics(data, strategy: Strategy, plot_step=iterations) -> List[float]:
        features_with_target = data.columns.values  # type:List[str]
        features = list(filter(lambda i: i != 'target', features_with_target))
        hics = HiCS(data, alpha=alpha, iterations=1, categorical_features=features_with_target)
        initial_hics = HiCS(data, alpha=alpha, iterations=500, categorical_features=features_with_target)

        relevance_by_feature = dict(
            [(feature, RandomVariableSamples()) for feature in features])  # type:Dict[str, RandomVariableSamples]

        true_relevance_by_feature = dict(
            [(feature, initial_hics.calculate_contrast([feature], 'target')) for feature in features])

        selected_relevant_feature_counts = []

        for iteration in range(iterations):
            items = Items(relevance_by_feature=relevance_by_feature, num_features_to_select=num_features_to_select,
                          iteration=iteration, true_relevance_by_feature=true_relevance_by_feature,
                          name=strategy.name)

            if iteration % plot_step == plot_step - 1:
                items.save_plot()

            selected_relevant_feature_counts.append(items.num_selected_relevant_features)

            feature, value = strategy.choose(items)

            print(f"Iteration {iteration}, chosen relevant features: {items.num_selected_relevant_features}")

            relevance_by_feature[feature].append(hics.calculate_contrast([feature], 'target'))

        return selected_relevant_feature_counts

    def plot_summary(relevance_by_run_by_time_by_strategy: Dict[str, np.ndarray]):
        plt.ylabel('chosen relevant features')
        plt.xlabel('iteration')
        items = list(relevance_by_run_by_time_by_strategy.items())
        colors = plt.cm.rainbow(np.linspace(0, 1, len(items)))
        for (strategy, mutual_information_by_run_by_time), color in zip(items, colors):
            average = np.average(mutual_information_by_run_by_time, axis=0)
            deviation = np.std(mutual_information_by_run_by_time, axis=0)

            plt.fill_between(range(len(average)),
                             list(average - deviation),
                             list(average + deviation),
                             color=color, alpha=.15)
            plt.plot(average, label=strategy.name, c=color)
        plt.legend(loc=2)
        title = f"{len(relevance_distribution)}features_{runs}runs"
        plt.title(title)
        directory = Path(".") / "plots"
        directory.mkdir(exist_ok=True)
        fig = plt.gcf()
        fig.savefig(str(directory / f"{timestamp()}_{title}.png"))
        plt.show()

    data_for_runs = [generate_data(correlation_distribution=relevance_distribution) for _ in range(runs)]

    mutual_information_by_run_by_time_by_strategy = dict([(strategy, np.array(
        [run_hics(data=data, strategy=strategy) for data in data_for_runs])) for strategy in strategies])

    plot_summary(mutual_information_by_run_by_time_by_strategy)


def show_distribution(distribution: List[float]):
    plt.plot(distribution)
    plt.show()


if __name__ == '__main__':
    exploit_strategies = [exploitation_strategy(exploitation) for exploitation in (0, .5, 1, 1.5, 2, 2.5, 3)]

    distribution = linearly_relevant_features() + [.2] * 80

    # show_distribution(distribution)

    main(relevance_distribution=distribution,
         strategies=[exploitation_strategy(0), gaussian_strategy(), exploitation_strategy(1.5)])
