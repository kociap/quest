# from sklearn.linear_model import LinearRegression
import math
from collections import deque
from IAverage import IAverage

class ExponentialMovingAverage(IAverage):
    def __init__(self, alpha):
        self.alpha = alpha
        self.reset()

    def add_value(self, value):
        if self.avg is None:
            self.avg = value
            self.avg_sq = value ** 2
        else:
            self.avg = self.alpha * value + (1 - self.alpha) * self.avg
            self.avg_sq = self.alpha * (value ** 2) + (1 - self.alpha) * self.avg_sq

    def get_average(self):
        return self.avg

    def get_variance(self):
        return self.avg_sq - (self.avg ** 2)

    def reset(self):
        self.avg = None
        self.avg_sq = None

class WeightedMovingAverage(IAverage):
    def __init__(self, N):
        self.values = []
        self.N = N

    def add_value(self, value):
        if len(self.values) >= self.N:
            self.values = self.values[1:]
        self.values.append(value)
        if self.counter == 0:
            self.avg = value
            self.avg_sq = value ** 2
            self.var = 0.0
            self.counter += 1
        else:
            self.counter += 1
            self.sum += self.counter * value
            self.sum_sq += self.counter * value**2

    def get_average(self):
        summed = sum([weight * value for weight, value in enumerate(self.values)])
        count = len(self.values)
        return 2 * summed / (count * (count + 1))

    def get_variance(self):
        mean = self.get_average()
        squares = sum([weight * value * value for weight, value in enumerate(self.values)])
        count = len(self.values)
        return 2 * squares / (count * (count + 1)) - squares * squares

    def reset(self):
        self.values = []

class SimpleMovingAverage:
    def __init__(self, N):
        self.N = N
        self.reset()

    def add_value(self, value):
        if len(self.values) == self.N:
            removed = self.values.popleft()
            self.sum -= removed
            self.sum_sq -= removed**2

        self.values.append(value)
        self.sum += value
        self.sum_sq += value**2

    def get_average(self):
        if not self.values:
            return 0.0
        return self.sum / len(self.values)

    def get_variance(self):
        n = len(self.values)
        if n < 2:
            return 0.0
        mean = self.get_average()
        # Bessel correction
        return (self.sum_sq / n - mean**2) * n / (n - 1)

    def reset(self):
        self.values = deque()
        self.sum = 0.0
        self.sum_sq = 0.0

class SimpleAverage(IAverage):
    def __init__(self):
        self.reset()

    def add_value(self, value):
        self.counter += 1
        delta = value - self.avg
        self.avg += delta / self.counter
        delta2 = value - self.avg
        self.avg_sq += delta * delta2

    def get_average(self):
        return self.avg if self.counter > 0 else 0.0

    def get_variance(self):
        if self.counter < 2:
            return 0.0
        return self.avg_sq / (self.counter - 1) # Bessel correction

    def reset(self):
        self.counter = 0
        self.avg = 0.0
        self.avg_sq = 0.0

def anomaly_clearing(data: list, avg: IAverage, inequality, confidence):
    result = [[d,False] for d in data]
    while True:
        anomaly = False
        for idx, x in enumerate(result):
            if x[1]:
                continue
            avg.reset()
            i = max(len(result)-idx,idx)
            while i>0:
                if (idx-i>=0) and not result[idx-i][1]:
                    avg.add_value(result[idx-i][0])
                if (idx+i<len(result)) and not result[idx+i][1]:
                    avg.add_value(result[idx+i][0])
                i -= 1
            if inequality(x[0], avg.get_average(), avg.get_variance()) <= confidence:
                x[1]=True
                anomaly = True
        if not anomaly:
            break
    return result
