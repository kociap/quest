from statistics import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("example.csv")

data['draw_price'] = data['price'].clip(upper = 130)
# data['draw_price'] = data['price']

anomalies = data[data['flag'] > 0]
notanomalies = data[data['flag'] == 0]

plt.scatter(anomalies['order_id'], anomalies['draw_price'], s=5, color="red")
plt.scatter(notanomalies['order_id'], notanomalies['draw_price'], s=5)
plt.plot(data['order_id'], data['base_price'], color="red")
plt.plot(data['order_id'], data['base_price'] + np.sqrt(data['price_range']), color="orange", ls="--")
plt.plot(data['order_id'], data['base_price'] - np.sqrt(data['price_range']), color="orange", ls="--")

moving_avg_alg = ExponentialMovingAverage(0.5)
double_moving_avg_alg = ExponentialMovingAverage(0.25)
triple_moving_avg_alg = ExponentialMovingAverage(0.1)
moving_avg_data = []

for (price,flag) in zip(data['price'],data['flag']):
    if flag == 0:
        moving_avg_alg.add_value(price)
        double_moving_avg_alg.add_value(moving_avg_alg.avg)
        triple_moving_avg_alg.add_value(double_moving_avg_alg.avg)
    moving_avg_data.append(triple_moving_avg_alg.avg)

# i = 0
# for flag in data['flag']:
#     if flag == 0:
#         i += 1

#     avg = weighted_moving_average(list(reversed(list(notanomalies['price'][:i]))))
#     # moving_avg_alg.add_value(price)
#     moving_avg_data.append(avg)

plt.plot(data['order_id'], moving_avg_data, color="cyan")

plt.show()

