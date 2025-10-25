# from sklearn.linear_model import LinearRegression
import math
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
    def __init__(self):
        self.reset()

    def add_value(self, value):
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
        return self.sum/(((self.counter+1)*self.counter)/2)
    
    def get_variance(self):
        return self.sum_sq/(((self.counter+1)*self.counter)/2) - (self.sum/(((self.counter+1)*self.counter)/2))**2
    
    def reset(self):
        self.sum = 0
        self.sum_sq = 0
        self.counter = 0

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
        return self.avg_sq / (self.counter - 1)  # korekta Bessela 

    def reset(self):
        self.counter = 0
        self.avg = 0.0
        self.avg_sq = 0.0

def anomaly_clearing(data: list, avg: IAverage):
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
            if abs(x[0]-avg.get_average())>=math.sqrt(10*avg.get_variance()):
                # print("\n\n\n\n\n\n")
                # print(x[0],avg.get_average(),avg.get_variance())
                x[1]=True
                anomaly = True
        # print(anomaly, result)
        if not anomaly:
            break
    return result
