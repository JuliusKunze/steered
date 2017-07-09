from unittest import TestCase

from stats import ValuesWithStats


class TestValuesWithStats(TestCase):
    def test(self):
        v = ValuesWithStats()
        for e in [1, 2, 2, 3]:
            v.append(e)

        self.assertEqual(v.average, 2)
        self.assertEqual(v.sum_of_squared_deviations, 2)
        self.assertEqual(v.variance_of_input, .5 * 4 / 3)
        self.assertEqual(v.variance_of_average, .5 / 3)
