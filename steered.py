from typing import List, Tuple

from hics.contrast_measure import HiCS

from stats import RandomVariableSamples
from strategy import Strategy, Items

def select_features(data, strategy: Strategy, num_features_to_select: int, iterations: int, plot_step: int,
                    true_relevances: List[float] = None, alpha=.001) -> Tuple[List[str], List[float]]:
    features_with_target = data.columns.values  # type:List[str]
    features = list(filter(lambda i: i != 'target', features_with_target))

    hics = HiCS(data, alpha=alpha, iterations=1, categorical_features=features_with_target)

    relevance_by_feature = dict(
        [(feature, RandomVariableSamples()) for feature in features])  # type:Dict[str, RandomVariableSamples]

    if true_relevances is None:
        true_relevances = [0] * len(features)

    true_relevance_by_feature = dict([(feature, true_relevances[int(feature)]) for feature in features])
    selected_relevant_feature_counts = []

    for iteration in range(iterations):
        items = Items(relevance_by_feature=relevance_by_feature, num_features_to_select=num_features_to_select,
                      iteration=iteration, true_relevance_by_feature=true_relevance_by_feature,
                      name=strategy.name)

        selected_relevant_feature_counts.append(items.num_selected_relevant_features)

        feature, value = strategy.choose(items)

        relevance_by_feature[feature].append(hics.calculate_contrast([feature], 'target'))

        if iteration % plot_step == plot_step - 1:
            items.save_plot()

        print(f"Iteration {iteration}, chosen relevant features: {items.num_selected_relevant_features}")

    return items.selected_features, selected_relevant_feature_counts
