from benchmark_functions import *
from pso import *

import matplotlib.pyplot as plt

iterations = 100
particles = 500
dimensions = 2
search_space = [[-5.12] * dimensions, [5.12] * dimensions]
# print init_pso(iterations, particles, search_space)
velocity, fitness, local_best, local_position, global_best, global_position = init_pso(iterations, particles,
                                                                                       search_space)
# print create_swarm(particles, search_space)
swarm = create_swarm(particles, search_space)

iteration = 0
while iteration < iterations:
    fitness = [sphere(solution) for solution in swarm]

    local_best, local_position = update_local_position(swarm, fitness, local_best, local_position)

    global_best, global_position = update_global_position(swarm, local_best, global_best, global_position, iteration)

    swarm, velocity = update_swarm(swarm, velocity, local_position, global_position, iteration)

    swarm = check_swarm(swarm, search_space)

    iteration += 1

plt.plot(global_best, '.-', label='%f' % min(global_best))
plt.xlim(-1, iteration)
# plt.ylim(min(global_best), max(global_best)+0.01)
plt.legend()
plt.show()
