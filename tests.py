from my_statistics import *
from inequalities import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from my_isolation_forest import AnomalyDetector

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
    # print(tp,fp,fn,tn)
    if (tp + fp) == 0 or (tp + fn) == 0:
        return (0,0,0)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1 = 2 / (1/recall + 1/precision)
    return (precision,recall,f1)
    

data = pd.read_csv("large.csv")

pd.set_option('display.max_rows', None)
detector = AnomalyDetector(contamination=0.02)
clean_data = detector.fit_and_flag(data['price'])

avg_simple = SimpleAverage()
avg_weighted = WeightedMovingAverage(50)
avg_exponential = ExponentialMovingAverage(0.1)
avg_simple_moving = SimpleMovingAverage(50)

avgs = [avg_simple, avg_weighted, avg_exponential, avg_simple_moving]
ineqs = [confidence_czebyszew, confidence_normal_chernoff, confidence_normal]
conf = [0.005 * i for i in range(20)]

anomalies = list(data['flag'])

for avg in avgs:
    print(avg)
    for ineq in ineqs:
        print(ineq)
        plot_data = []
        for c in conf:
            result = anomaly_clearing(list(data['price']), avg, ineq, c)
            result = [x[1] for x in result]
            (p,r,f) = f1(result,anomalies)
            print(f"f1: {(p,r,f)}")
            plot_data.append(f)
        plt.plot(conf, plot_data, label=f"{avg}{ineq}")

plt.legend()
plt.show()
        

# results = []
# results.append(anomaly_clearing(list(data['price']),avg_simple,confidence_czebyszew,0.1))
# results.append(anomaly_clearing(list(data['price']),avg_weighted,confidence_czebyszew,0.1))
# results.append(anomaly_clearing(list(data['price']),avg_exponential,confidence_czebyszew,0.1))
# results.append(anomaly_clearing(list(data['price']),avg_simple_moving,confidence_czebyszew,0.1))
# results.append(clean_data.values.tolist())
# anomalies = data['flag']
#  for idx in range(len(anomalies)):
#      miss_flag = False
#      for result in results:
#          if result[idx][1] != anomalies[idx]:
#              miss_flag = True
#      if miss_flag:
#          print(f"{results[0][idx]} , {results[1][idx]} , {results[2][idx]} , {results[3][idx]} , {results[4][idx]} , {anomalies[idx]} , {idx}")]

 
