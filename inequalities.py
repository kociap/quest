import math
from scipy.stats import norm

# Nierówność Czebyszewa Pr(|X - E[X]| >= k) <= Var[X] / k^2
# Inaczej: Pr(|X - E[X]| / sqrt(Var[X]) >= k) <= 1 / k^2
# Zwracamy właśnie to 1/k^2
# "Jakie maksymalnie jest prawdopodobieństwo, że osiądnięto co najmniej takie odchylenie?"
def confidence_czebyszew(value, avg, var):
    k = abs(value - avg) / math.sqrt(var)
    if k == 0:
        return 1
    return 1 / (k*k)

# Nierówność Chernoffa
# Pr(X >= a) <= min(t > 0) E[e^(Xt)] / e^(ta)
# Pr(X <= a) <= min(t < 0) E[e^(Xt)] / e^(ta)
# Czentrując:
# Pr(X - E[X] >= a) <= min(t > 0) E[e^((X-E[X])t)] / e^(ta)
# Pr(X - E[X] <= a) <= min(t < 0) E[e^((X-E[X])t)] / e^(ta)
# Dla rozkładu normalnego można wyliczyć, że prawe strony wynoszą e^(-a^2/(2Var[X]))
# A przynajmniej według [https://en.wikipedia.org/wiki/Chernoff_bound]
# Dostajemy coś w stylu
# Pr(|X - E[X]| >= a) <= exp(-a^2/(2Var[X]))
def confidence_normal_chernoff(value, avg, var):
    a = abs(value - avg)
    return math.exp(-a*a / (2 * var))
    
# Metoda 3 sigm (Z score)
# To stosujemy kiedy znamy dokładnie (albo możemy wyestymować) dystrybuantę
# Pr(-a <= X <= a) = F(a) - F(-a)
# Pr(|X| >= a) = 1 - F(a) + F(-a)
# Pr(|X - E[X]| / sqrt(Var[X]) >= a) = 1 - N(a) + N(-a)
def confidence_normal(value, avg, var):
    a = abs(value - avg) / math.sqrt(var)
    return 1 - norm.cdf(a,0,1) + norm.cdf(-a,0,1)
