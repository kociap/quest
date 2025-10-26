from my_statistics import *
from inequalities import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("example.csv")

avg = SimpleAverage()

for row in data[:10].itertuples():
    avg.add_value(row.price)

for row in data.itertuples():
    if avg.get_variance() == 0.0:
        print(f"{row.price}, {row.flag}, 0")
    else:
        print(f"{row.price}, {row.flag}, {confidence_normal(row.price, avg.get_average(), avg.get_variance())}")

    if avg.get_variance() != 0.0 and confidence_normal(row.price, avg.get_average(), avg.get_variance()) >= 0.01:
        avg.add_value(row.price)
   
