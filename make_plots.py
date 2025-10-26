from my_statistics import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

plt.rcParams['font.size'] = 12

figure = plt.figure()
axes = figure.add_subplot()

up_data = pd.read_csv("up.csv")
# down_data = pd.read_csv("down.csv")
# down_data['order_id'] = down_data['order_id'] + len(up_data['order_id'])
# data = pd.concat([up_data, down_data])
up2_data = pd.read_csv("up2.csv")
up2_data['order_id'] = up2_data['order_id'] + len(up_data['order_id'])
data = pd.concat([up_data, up2_data])

data['draw_price'] = data['price'].clip(upper = 180)
data['draw_count'] = data['count'].clip(upper = 130)
# data['draw_price'] = data['price']

anomalies = data[data['flag'] > 0]
notanomalies = data[data['flag'] == 0]

axes.scatter(anomalies['order_id'], anomalies['draw_price'], s=5, color="red")
axes.scatter(notanomalies['order_id'], notanomalies['draw_price'], s=5)
axes.plot(data['order_id'], data['base_price'], color="red")
axes.plot(data['order_id'], data['base_price'] + data['price_range'], color="orange", ls="--")
axes.plot(data['order_id'], data['base_price'] - data['price_range'], color="orange", ls="--")

moving_avg_alg = SimpleMovingAverage(50)
moving_avg_data = []
moving_var_data = []

for (price,flag) in zip(data['price'],data['flag']):
    if flag == 0:
        moving_avg_alg.add_value(price)
    moving_avg_data.append(moving_avg_alg.get_average())
    moving_var_data.append(moving_avg_alg.get_variance())

axes.plot(data['order_id'], moving_avg_data, color="cyan", label="B")
axes.plot(data['order_id'], np.array(moving_avg_data) + np.sqrt(moving_var_data), color="magenta", ls="--")
axes.plot(data['order_id'], np.array(moving_avg_data) - np.sqrt(moving_var_data), color="magenta", ls="--")

axes.legend(loc = "best", bbox_to_anchor = (1, 1), fancybox = False, shadow = False, framealpha = 1.0)
axes.set_axisbelow(True)
axes.yaxis.grid(color = "#606060", linestyle = "--")
axes.yaxis.grid(which = "minor", color = "#808080", linestyle = ":")
axes.yaxis.minorticks_on()
axes.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
axes.ticklabel_format(axis = "y", scilimits = [-5, 3])
# axes.set_ylim(bottom = 1.0)
axes.set_ylabel("Average")
axes.set_title(f"A")

plt.show()

figure = plt.figure()
axes = figure.add_subplot()

axes.scatter(anomalies['order_id'], anomalies['draw_count'], s=5, color="red")
axes.scatter(notanomalies['order_id'], notanomalies['draw_count'], s=5)
axes.plot(data['order_id'], data['base_count'], color="red")
axes.plot(data['order_id'], data['base_count'] + data['count_range'], color="orange", ls="--")
axes.plot(data['order_id'], data['base_count'] - data['count_range'], color="orange", ls="--")

moving_avg_alg = SimpleAverage()
moving_avg_data = []
moving_var_data = []

for (count,flag) in zip(data['count'],data['flag']):
    if flag == 0:
        moving_avg_alg.add_value(count)
    moving_avg_data.append(moving_avg_alg.get_average())
    moving_var_data.append(moving_avg_alg.get_variance())

axes.plot(data['order_id'], moving_avg_data, color="cyan", label="B")
axes.plot(data['order_id'], np.array(moving_avg_data) + np.sqrt(moving_var_data), color="magenta", ls="--")
axes.plot(data['order_id'], np.array(moving_avg_data) - np.sqrt(moving_var_data), color="magenta", ls="--")

axes.legend(loc = "best", bbox_to_anchor = (1, 1), fancybox = False, shadow = False, framealpha = 1.0)
axes.set_axisbelow(True)
axes.yaxis.grid(color = "#606060", linestyle = "--")
axes.yaxis.grid(which = "minor", color = "#808080", linestyle = ":")
axes.yaxis.minorticks_on()
axes.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
axes.ticklabel_format(axis = "y", scilimits = [-5, 3])
# axes.set_ylim(bottom = 1.0)
axes.set_ylabel("Average")
axes.set_title(f"A")

plt.show()

