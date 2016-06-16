
def sphere(solution):
    d = len(solution)
    sumatory = 0
    for i in range(0, d):
        sumatory += solution[i] ** 2

    return sumatory