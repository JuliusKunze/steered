from math import erf, sqrt, exp, pi

from lazy import lazy

sqrt2 = sqrt(2)
inverse_sqrt2pi = 1 / sqrt(2 * pi)


def standard_gaussian(x: float) -> float:
    return inverse_sqrt2pi * exp(- .5 * x * x)


def phi(x: float) -> float:
    return .5 * (1 + erf(x / sqrt2))


class GaussianRandomVariable:
    def __init__(self, mean: float, variance: float):
        self.variance = variance
        self.mean = mean

    @lazy
    def standard_deviation(self):
        return sqrt(self.variance)

    def __sub__(self, other: 'GaussianRandomVariable') -> 'GaussianRandomVariable':
        return GaussianRandomVariable(mean=self.mean - other.mean, variance=self.variance + other.variance)

    @lazy
    def p_is_positive(self):
        return phi(self.mean / self.standard_deviation)

    @property
    def expected_value_if_truncated_of_negative_mapped_to_0(self):
        return self.mean * self.p_is_positive + \
               self.standard_deviation * standard_gaussian(-self.mean / self.standard_deviation)

    @property
    def expected_value_given_positive(self):
        return self.expected_value_if_truncated_of_negative_mapped_to_0 / self.p_is_positive


class ValuesWithStats:
    def __init__(self):
        self.values = []
        self.sum = 0.0
        self.sum_of_squared_deviations = 0

    def append(self, value: float):
        old_mean = self.mean

        self.values.append(value)
        self.sum += value

        self.sum_of_squared_deviations += (value - old_mean) * (value - self.mean)

    @property
    def count(self):
        return len(self.values)

    @property
    def mean(self):
        if self.count == 0:
            return 0

        return self.sum / self.count

    @property
    def variance_of_input(self) -> float:
        return self.sum_of_squared_deviations / (self.count - 1)

    @property
    def variance_of_mean(self) -> float:
        if self.count <= 1:
            return float('inf')

        return self.variance_of_input / self.count

    @property
    def mean_as_gaussian(self) -> GaussianRandomVariable:
        return GaussianRandomVariable(self.mean, variance=self.variance_of_mean)
