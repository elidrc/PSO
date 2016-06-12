import math

def sphere(solution):
    d = len(solution)
    sumatory = 0
    for i in range(0, d):
        sumatory += solution[i] ** 2
    return sumatory

def rosenbrock():
    pass

def ackley(solution, a=20, b=0.2, c=2*math.pi):
    d = len(solution)
    sum1 = 0
    sum2 = 0

    for i in range(0, d):
        sum1 += solution[i] ** 2
        sum2 += math.cos(c * solution[i])
    term1 = -a * math.exp(-b * math.sqrt(sum1 / d))
    term2 = -math.exp(sum2 / d)

    return term1 + term2 + a + math.exp(1)
