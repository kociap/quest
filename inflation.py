
from sklearn.linear_model import LinearRegression
import numpy as np

def linear_regression(data: list):
    data = data[::-1]
    X = np.arange(len(data)).reshape(-1,1)
    model = LinearRegression().fit(X, data)
    pred = model.predict([[len(data)]])[0]
    return pred

def weighted_moving_average(data: list):
    sum = 0
    for id,val in enumerate(data):
        sum += (len(data)-id)*val
    return sum/((len(data)+1)*len(data)/2)

def exponential_moving_average(data: list):
    sum = 0
    coeff = 1
    denominator = 0
    alpha = 2/(len(data)+1)
    for id,val in enumerate(data):
        sum += coeff*val
        denominator += coeff
        coeff *= alpha
    return sum/denominator
