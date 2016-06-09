import matplotlib.pyplot as plt

from pso import *
from benchmark_functions import sphere

iterations = 100
particles = 100
search_space = [[0] * 5, [5.12] * 5]

# print init_pso(iterations, particles, search_space)
velocity, fitness, local_best, local_position, global_best, global_position = init_pso(iterations, particles,
                                                                                       search_space)
# print create_swarm(particles, search_space)
swarm = create_swarm(particles, search_space)

iteration = 0
while iteration < iterations:
    fitness = array([abs(0 - sphere(solution)) for solution in swarm]).T
    local_best, local_position = update_local_position(swarm, fitness, local_best, local_position)
    global_best, global_position = update_global_position(swarm, local_best, global_best, global_position,
                                                          iteration)
    swarm, velocity = update_swarm(swarm, velocity, local_position, global_position, 2, 2, 1, 1)

    iteration += 1

plt.plot(global_best, 'o-', label='%f' % global_best[iteration - 1])
plt.xlim(-1, iteration)
plt.ylim(min(global_best), max(global_best) + 0.01)
plt.legend()
plt.show()
