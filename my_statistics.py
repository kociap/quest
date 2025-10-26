# from sklearn.linear_model import LinearRegression
import math
import copy
from collections import deque
from IAverage import IAverage
from inequalities import *

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
        self.reset()
        self.N = N

    def add_value(self, value):
        if len(self.values) >= self.N:
            self.values = self.values[1:]
        self.values.append(value)

    def get_average(self):
        count = len(self.values)
        if count == 0:
            return 0.0
        summed = sum([weight * value for weight, value in enumerate(self.values, start=1)])
        return 2 * summed / (count * (count + 1))

    def get_variance(self):
        count = len(self.values)
        if count <= 1:
            return 0.0
        mean = self.get_average()
        squares = sum([weight * value ** 2 for weight, value in enumerate(self.values, start=1)])
        return 2 * squares / (count * (count + 1)) - (mean ** 2)

    def reset(self):
        self.values = []
        self.sum = 0.0
        self.sum_sq = 0.0

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

def f1(result, anomalies):
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for (r,a) in zip(result,anomalies):
        if r == True and a == 1:
            tp += 1
        if r == True and a == 0:
            fp += 1
        if r == False and a == 1:
            fn += 1
        if r == False and a == 0:
            tn += 1
    if (tp + fp) == 0 or (tp + fn) == 0:
        return (0,0,0)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1 = 2 / (1/recall + 1/precision)
    return (precision,recall,f1)

def inequality_prediction(train_data, test_data, avg: IAverage, inequality, confidence):
    clean_train_data_by_price = anomaly_clearing(train_data['price'].values.tolist(),avg,inequality,confidence)
    clean_train_data_by_count = anomaly_clearing(train_data['count'].values.tolist(),avg,inequality,confidence)
    clean_train_data = [[data_1[0],data_2[0]] for data_1,data_2 in zip(clean_train_data_by_price,clean_train_data_by_count) if not data_1[1] and not data_2[1]]
    result = []
    avg.reset()
    avg_count = copy.deepcopy(avg)
    for data in clean_train_data:
        avg.add_value(data[0])
        avg_count.add_value(data[1])
    for idx, data in test_data.iterrows():
        if inequality(data['price'],avg.get_average(),avg.get_variance()) <= confidence:
            result.append(True)
        elif inequality(data['count'],avg_count.get_average(),avg_count.get_variance()) <= confidence:
            result.append(True)
        else:
            result.append(False)
    # print(result)
    return f1(result,test_data['flag'])

