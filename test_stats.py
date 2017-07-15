from unittest import TestCase

from stats import RandomVariableSamples


class TestRandomVariableSamples(TestCase):
    def test(self):
        v = RandomVariableSamples()
        for e in [1, 2, 2, 3]:
            v.append(e)

        self.assertEqual(v.mean, 2)
        self.assertEqual(v.sum_of_squared_deviations, 2)
        self.assertEqual(v.variance_of_input, .5 * 4 / 3)
        self.assertEqual(v.variance_of_mean, .5 / 3)
