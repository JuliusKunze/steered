from pathlib import Path
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas
import sklearn
from pandas import DataFrame

import real_data
from steered import select_features
from strategy import Strategy, exploitation_strategy, gaussian_strategy
from synthetic_data import generate_data
from util import timestamp

plt.rcParams["figure.figsize"] = (19.20 / 2, 10.80 / 2)

import random


def choose_random(X, N):
    return list(map(lambda _: random.choice(X), range(N)))


def main(data_for_runs: List[DataFrame], strategies: List[Strategy], num_features_to_select,
         iterations, true_relevances=None, runs=1):
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
        title = f"{len(true_relevances)}features_{runs}runs"
        plt.title(title)
        directory = Path(".") / "plots"
        directory.mkdir(exist_ok=True)
        fig = plt.gcf()
        fig.savefig(str(directory / f"{timestamp()}_{title}.pdf"))
        plt.show()

    mutual_information_by_run_by_time_by_strategy = dict([(strategy, np.array(
        [select_features(data=data, strategy=strategy, num_features_to_select=num_features_to_select,
                         iterations=iterations,
                         plot_step=iterations, true_relevances=true_relevances)[1] for data in data_for_runs])) for
                                                          strategy in strategies])

    plot_summary(mutual_information_by_run_by_time_by_strategy)


def show_distribution(distribution: List[float]):
    plt.plot(distribution)
    plt.show()


def synthetic():
    exploit_strategies = [exploitation_strategy(exploitation) for exploitation in (0, .5, 1, 1.5, 2, 2.5, 3)]
    distribution = [.6] * 5 + [.585] * 10 + [0] * 20  # linearly_relevant_features() + [.2] * 80
    data_for_runs = [generate_data(relevance_distribution=distribution) for _ in range(1)]
    # show_distribution(distribution)
    main(data_for_runs, num_features_to_select=5, iterations=100, true_relevances=distribution,
         strategies=[gaussian_strategy(), exploitation_strategy(0)])  # exploitation_strategy(1.5)])


def real(strategy=exploitation_strategy(), iterations=1000):
    bundle = sklearn.datasets.load_digits()

    data = real_data.dataframe(bundle)  # real_data.thrombin().iloc[:, range(100)]

    print(f'{bundle.data.shape[1]} features')
    selected_features, stats = select_features(data, strategy=strategy, num_features_to_select=10,
                                               iterations=iterations,
                                               plot_step=iterations)

    return score_selected_features(data, selected_features) # choose_random(list(data.columns.values)[1:], N=1))


def score_selected_features(data, selected_features):
    print(f'{selected_features} features selected')

    data_selected = pandas.DataFrame(data, columns=['target'] + selected_features)
    list(selected_features)

    return np.average(real_data.classify(real_data.bunch(data_selected)))


if __name__ == '__main__':
    strategies = [exploitation_strategy(), gaussian_strategy()]

    scores = dict([(s.name, real(s)) for s in strategies])

    print(f'f1 scores {scores}')
