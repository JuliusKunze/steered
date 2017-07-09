class ValuesWithStats:
    def __init__(self):
        self.values = []
        self.sum = 0.0
        self.sum_of_squared_deviations = 0

    def append(self, value: float):
        old_average = self.average

        self.values.append(value)
        self.sum += value

        self.sum_of_squared_deviations += (value - old_average) * (value - self.average)

    @property
    def count(self):
        return len(self.values)

    @property
    def average(self):
        if self.count == 0:
            return 0

        return self.sum / self.count

    @property
    def variance_of_input(self) -> float:
        return self.sum_of_squared_deviations / (self.count - 1)

    @property
    def variance_of_average(self) -> float:
        return self.variance_of_input / self.count