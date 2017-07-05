import unittest

from main import ValuesWithAverage


class TestValuesWithAverage(unittest.TestCase):
    def test_average(self):
        v = ValuesWithAverage()
        for e in [1, 2, 2, 3]:
            v.append(e)

        self.assertEqual(v.average, 2)
        self.assertEqual(v.sum_of_squared_deviations, 2)
        self.assertEqual(v.variance_of_input, .5 * 4 / 3)
        self.assertEqual(v.variance_of_average, .5 / 3)

unittest.main()