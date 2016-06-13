from math import *


def sphere(solution):
    dimensions = len(solution)
    sumatory = 0
    for i in range(0, dimensions):
        sumatory += solution[i] ** 2

    return sumatory


def rosenbrock(solution):
    dimensions = len(solution)
    sumatory = 0

    for i in range(0, dimensions):
        first_term = (solution[i + 1] - solution[i] ** 2) ** 2
        second_term = (solution[i] - 1) ** 2
        sumatory += 100 * first_term + second_term

    return sumatory


def ackley(solution, a=20, b=0.2, c=2*pi):
    dimensions = len(solution)
    first_sumatory = 0
    second_sumatory = 0

    for i in range(0, dimensions):
        first_sumatory += solution[i] ** 2
        second_sumatory += cos(c * solution[i])
    first_term = -a * exp(-b * sqrt(first_sumatory / dimensions))
    second_term = -exp(second_sumatory / dimensions)

    return first_term + second_term + a + exp(1)

def griewank(solution):
    dimensions = len(solution)
    sumatory = 0
    product = 1

    for i in range(0, dimensions):
        sumatory += solution[i] ** 2 / 4000.0
        product *= cos(solution[i] / sqrt(i))

    return sumatory - product + 1

def rastrigin(solution):
    dimensions = len(solution)
    sumatory = 0

    for i in range(0, dimensions):
        sumatory += (solution[i] ** 2 - 10 * cos(2 * pi * solution[i]))

    return 10 * dimensions + sumatory
