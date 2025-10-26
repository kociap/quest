from my_statistics import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from my_isolation_forest import AnomalyDetector

data = pd.read_csv("example.csv")

data['draw_price'] = data['price'].clip(upper = 130)
# data['draw_price'] = data['price']
pd.set_option('display.max_rows', None)
detector = AnomalyDetector(contamination=0.1)
clean_data = detector.fit_and_flag(data['price'])
avg_simple = SimpleAverage()
avg_weighted = WeightedMovingAverage(50)
avg_exponential = ExponentialMovingAverage(0.1)
avg_simple_moving = SimpleMovingAverage(50)
results = []
results.append(anomaly_clearing(list(data['price']),avg_simple))
results.append(anomaly_clearing(list(data['price']),avg_weighted))
results.append(anomaly_clearing(list(data['price']),avg_exponential))
results.append(anomaly_clearing(list(data['price']),avg_simple_moving))
results.append(clean_data.values.tolist())
anomalies = data['flag']
for idx in range(len(anomalies)):
    miss_flag = False
    for result in results:
        if result[idx][1] != anomalies[idx]:
            miss_flag = True
    if miss_flag:
        print(f"{results[0][idx]} , {results[1][idx]} , {results[2][idx]} , {results[3][idx]} , {results[4][idx]} , {anomalies[idx]} , {idx}")
