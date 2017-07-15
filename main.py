from pathlib import Path
from time import strftime
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
from hics.contrast_measure import HiCS

from data import generate_data, plateau_distribution, linearly_relevant_features
from stats import ValuesWithStats
from strategy import Strategy, exploitation_strategy, Items, gaussian_strategy

plt.rcParams["figure.figsize"] = (19.20 / 2, 10.80 / 2)


def timestamp() -> str:
    return strftime('%Y%m%d-%H%M%S')


def test(correlation_distribution: List[float], strategies: List[Strategy], num_features_to_select=10,
         iterations=1000, runs=3):
    def run_hics(data, strategy: Strategy, plot_step=iterations * 2) -> List[float]:
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

            if iteration % plot_step == plot_step - 1:
                items.show_plot()

            selected_relevant_feature_counts.append(items.selected_relevant_feature_count)

            feature, value = strategy.choose(items)

            print(f"Iteration {iteration}, chosen relevant features: {items.selected_relevant_feature_count}")

            mutual_by_feature[feature].append(hics.calculate_contrast([feature], 'target'))

        return selected_relevant_feature_counts

    def plot_summary(mutual_information_by_run_by_time_by_strategy: Dict[str, np.ndarray]):
        plt.ylabel('chosen relevant features')
        plt.xlabel('iteration')
        items = list(mutual_information_by_run_by_time_by_strategy.items())
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
        title = f"{len(correlation_distribution)}features_{runs}runs"
        plt.title(title)
        directory = Path(".") / "plots"
        directory.mkdir(exist_ok=True)
        fig = plt.gcf()
        fig.savefig(str(directory / f"{timestamp()}_{title}.svg"))
        plt.show()

    data_for_runs = [generate_data(correlation_distribution=correlation_distribution) for _ in range(runs)]

    mutual_information_by_run_by_time_by_strategy = dict([(strategy, np.array(
        [run_hics(data=data, strategy=strategy) for data in data_for_runs])) for strategy in strategies])

    plot_summary(mutual_information_by_run_by_time_by_strategy)


if __name__ == '__main__':
    exploit_strategies = [exploitation_strategy(exploitation) for exploitation in (0, .5, 1, 1.5, 2, 2.5, 3)]

    distribution = linearly_relevant_features() + [.2] * 180

    test(correlation_distribution=distribution,
         strategies=[gaussian_strategy(), exploitation_strategy(1.5), exploitation_strategy(0)])
