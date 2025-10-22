import argparse
import random
import sys
from enum import Enum

parser = argparse.ArgumentParser(
  description = """""",
  formatter_class = argparse.RawDescriptionHelpFormatter,
)
parser.add_argument("orders", type = int,
  help = "the total number of orders to generate.")
parser.add_argument("--disable-errors", type = bool, default = False,
  help = "disable generation of random errors.")
parser.add_argument("--error-frequency", type = int, default = 1000,
  help = """the frequency of data anomalies. The probability of an anomaly
            occurring is 1/frequency.""")
parser.add_argument("--price", type = float, default = 1.0,
  help = "the base price of the product.")
parser.add_argument("--inflation", type = float, default = 1.0,
  help = "the inflation rate in %% per annum.")
parser.add_argument("--count-mean", type = int, default = 1,
  help = "the average size of the order.")
parser.add_argument("--count-range", type = int, default = 1,
  help = "the deviation from the average size of the order.")
parser.add_argument("--order-frequency", type = int, default = 1,
  help = "the number of days between consecutive orders")

args = parser.parse_args()

class Anomaly(Enum):
  PRICE_SWAP = 1,
  PRICE_PERIOD = 2,
  COUNT_SWAP = 3,

def random_anomalies():
  anomalies = (Anomaly.PRICE_SWAP, Anomaly.PRICE_PERIOD, Anomaly.COUNT_SWAP)
  k = random.randint(1, len(anomalies))
  return random.sample(anomalies, k)

def swap_last_2_integral_digits(number):
  if number < 10:
    return number
  string = str(number)
  period = string.find(".")
  if period == -1:
    string = string[:-2] + string[-1] + string[-2]
    return int(string)
  else:
    integral = string[:period]
    string = integral[:-2] + integral[-1] + integral[-2] + string[period:]
    return float(string)

def calculate_order(count, price, anomalies):
  for anomaly in anomalies:
    if anomaly == Anomaly.PRICE_SWAP:
      price = swap_last_2_integral_digits(price)
    elif anomaly == Anomaly.PRICE_PERIOD:
      price = round(price, 2) * 100
    elif anomaly == Anomaly.COUNT_SWAP:
      count = swap_last_2_integral_digits(count)
  return count, price

order = 1
price = args.price
inflation_frequency = 30
# periodic_inflation = (1 + annual_inflation) ^ (1 / periods_per_year) - 1
inflation = pow(1 + args.inflation / 100, 1 / (365 / inflation_frequency)) - 1
print(inflation, file = sys.stderr)
for day in range(0, args.order_frequency * args.orders):
  # Inflate prices once per period.
  if day % inflation_frequency == inflation_frequency - 1:
    price += price * inflation

  count = int(random.normalvariate(args.count_mean, args.count_range))

  anomaly = 1 - random.random() <= 1 / args.error_frequency
  anomalies = random_anomalies() if anomaly and not args.disable_errors else ()
  order_count, order_price = calculate_order(count, price, anomalies)
  print(f"{order},0,{order_count},{order_price:.2f},{1 if anomaly else 0}")

  order += 1

