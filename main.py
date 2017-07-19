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
from util import timestamp, timestamp_directory

plt.rcParams["figure.figsize"] = (19.20 / 2, 10.80 / 2)

import random

def choose_random(X, N):
    return list(map(lambda _: random.choice(X), range(N)))


def plot_summary(relevance_by_x_by_run_by_strategy: Dict[str, np.ndarray], name: str = '', xlabel='iteration', xvalues=None):
    plt.ylabel('share of relevant features selected')
    plt.xlabel(xlabel)
    items = list(relevance_by_x_by_run_by_strategy.items())
    colors = plt.cm.rainbow(np.linspace(0, 1, len(items)))
    for (strategy, mutual_information_by_x_by_run), color in zip(items, colors):
        average = np.average(mutual_information_by_x_by_run, axis=0)
        deviation = np.std(mutual_information_by_x_by_run, axis=0)

        x = xvalues if xvalues else range(len(average))
        plt.fill_between(x,
                         list(average - deviation),
                         list(average + deviation),
                         color=color, alpha=.15)
        plt.plot(x, average, label=strategy.name, c=color)
    plt.legend(loc=2)

    fig = plt.gcf()
    fig.savefig(str(timestamp_directory / f"{timestamp()}_{name}.pdf"))
    plt.clf()


def run_batch(data_for_runs: List[DataFrame], strategies: List[Strategy], num_features_to_select,
              iterations, true_relevances=None):
    correct_by_run_by_time_by_strategy = dict([(strategy, np.array(
        [select_features(data=data, strategy=strategy, num_features_to_select=num_features_to_select,
                         iterations=iterations,
                         plot_step=iterations, true_relevances=true_relevances)[1] for data in data_for_runs])) for
                                               strategy in strategies])

    plot_summary(correct_by_run_by_time_by_strategy, name=f"{len(true_relevances)}features_{len(data_for_runs)}runs")

    final_relevancy_by_run_by_strategy = dict([(s, a[:, -1]) for s, a in correct_by_run_by_time_by_strategy.items()])

    return final_relevancy_by_run_by_strategy


def run_series():
    # exploit_strategies = [exploitation_strategy(exploitation) for exploitation in (0, .5, 1, 1.5, 2, 2.5, 3)]
    # linearly_relevant_features() + [.2] * 80
    # show_distribution(distribution)

    all_num_features = [20, 50, 100, 150, 200, 300, 400]

    results = []

    strategies = [gaussian_strategy(), exploitation_strategy(0)]

    for num_features in all_num_features:
        distribution = [.6] * 10 + [.585] * 10 + [0] * (num_features - 20)

        data_for_runs = [generate_data(relevance_distribution=distribution) for _ in range(20)]

        results.append(run_batch(data_for_runs, num_features_to_select=10, iterations=200 + 2 * num_features,
                                 true_relevances=distribution,
                                 strategies=strategies))
        # exploitation_strategy(1.5)])

    to_plot = dict([(strategy, np.swapaxes([result[strategy] for result in results], 0, 1))
          for strategy in strategies])

    plt.ylabel('relevant selected feature count')
    plot_summary(to_plot, xlabel='feature count', name=f'feature_counts', xvalues=all_num_features)


def show_distribution(distribution: List[float]):
    plt.plot(distribution)
    plt.show()


def synthetic():
    distribution = [.6] * 5 + [.585] * 10 + [0] * 20
    data_for_runs = [generate_data(relevance_distribution=distribution) for _ in range(10)]
    run_batch(data_for_runs, num_features_to_select=5, iterations=100, true_relevances=distribution,
                  strategies=[gaussian_strategy()])


def real(strategy=exploitation_strategy(), iterations=1000):
    bundle = sklearn.datasets.load_digits()

    data = real_data.dataframe(bundle)  # real_data.thrombin().iloc[:, range(100)]

    print(f'{bundle.data.shape[1]} features')
    selected_features, stats = select_features(data, strategy=strategy, num_features_to_select=10,
                                               iterations=iterations,
                                               plot_step=iterations)

    return score_selected_features(data, selected_features)  # choose_random(list(data.columns.values)[1:], N=1))


def score_selected_features(data, selected_features):
    print(f'{selected_features} features selected')

    data_selected = pandas.DataFrame(data, columns=['target'] + selected_features)
    list(selected_features)

    return np.average(real_data.classify(real_data.bunch(data_selected)))


def f1_score_real():
    strategies = [gaussian_strategy()] * 10
    scores = [real(s) for s in strategies]
    average = np.average(scores)
    print(f'f1 score {average} +- {np.std(scores)} ({scores})')


if __name__ == '__main__':
    synthetic()
