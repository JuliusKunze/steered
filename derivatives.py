from collections import OrderedDict

import numpy as np
import tensorflow as tf

from stats import ValuesWithStats
from util import define_scope


class PartialDerivativesRelativeToMutualInformationLoss:
    def __init__(self, averages, variances):
        self.averages = averages
        self.variances = variances

        self.feature_count = self.averages.shape[0]

        self.loss

    def selected_index(self):
        return tf.reduce_max(self.averages)

    @define_scope
    def average_diffs(self):
        tiled = tf.tile(self.averages, self.feature_count)
        return tiled - tf.transpose(tiled)

    @define_scope
    def variance_diffs(self):
        tiled = tf.tile(self.variances, self.feature_count)
        return tiled - tf.transpose(tiled)

    @define_scope
    def inner_terms(self):
        return self.average_diffs / (tf.sqrt(2.0) * self.variance_diffs)

    @define_scope
    def variance_diffs(self):
        return tf.expand_dims(self.averages) * tf.transpose(tf.expand_dims(self.averages))

    @define_scope
    def loss(self):
        selected_average_diffs = tf.gather(self.average_diffs, self.selected_index)

        inner_terms = tf.erfc(self.average_diffs / (tf.sqrt(2) * self.variance_diffs))
        products = tf.reduce_prod(inner_terms, axis=1)
        return .5 * tf.reduce_sum(tf.multiply(selected_average_diffs, products))

    @define_scope
    def average_and_variance_gradients(self):
        return tf.gradients(self.loss, [self.averages, self.variances])

    @staticmethod
    def best_feature(values: OrderedDict[str, ValuesWithStats]) -> str:
        feature_count = len(values)

        averages = tf.placeholder(tf.float32, [feature_count], name='averages')
        variances = tf.placeholder(tf.float32, [feature_count], name='variances')

        model = PartialDerivativesRelativeToMutualInformationLoss(averages=averages, variances=variances)

        session = tf.Session()

        averages_input = [value.mean for value in values.values()]
        variances_input = [value.variance_of_mean for value in values.values()]
        (average_gradients, variance_gradients), loss = session.run(
            [model.average_and_variance_gradients, model.loss], {averages: averages_input, variances: variances_input})

        print(loss)

        feature_index = np.argmax(variance_gradients)[0]

        return values[feature_index]
