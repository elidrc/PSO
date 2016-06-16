from numpy import *



def init_pso(iterations, particles, search_space):
    dimensions = size(search_space, 1)

    velocity = zeros((particles, dimensions))
    fitness = ones((particles, 1)) * inf
    local_best = ones((particles, 1)) * inf
    local_position = ones((particles, dimensions)) * inf
    global_best = ones((iterations, 1)) * inf
    global_position = ones((iterations, dimensions)) * -inf

    return velocity, fitness, local_best, local_position, global_best, global_position


def create_swarm(particles, search_space):
    dimensions = size(search_space, 1)

    range_search_space = ones((particles, dimensions)) * search_space[1] - ones((particles, dimensions)) * search_space[0]
    lower = ones((particles, dimensions)) * search_space[0]

    return random.rand(particles, dimensions) * range_search_space + lower


def update_local_position(swarm, fitness, local_best, local_position):
    if ~isinf(local_best[0]):
        best_indices = less(fitness, local_best)
        local_best = local_best * (~best_indices).astype(int) + (fitness * best_indices.astype(int))
        local_position[nonzero(best_indices), :] = swarm[nonzero(best_indices), :]
    else:
        local_best = fitness
        local_position = swarm

    return local_best, local_position


def update_global_position(swarm, local_best, global_best, global_position, iteration):
    position = argmin(local_best)
    best = local_best[position]

    if iteration > 0:
        if best < global_best[iteration - 1]:
            global_best[iteration] = best
            global_position[iteration, :] = swarm[position, :]
        else:
            global_best[iteration] = global_best[iteration - 1]
            global_position[iteration, :] = global_position[iteration - 1, :]

    else:
        global_best[iteration, :] = best
        global_position[iteration, :] = swarm[position, :]

    return global_best, global_position


def update_swarm(swarm, velocity, local_position, global_position, iteration, c1=2, c2=2, inertia_weight=1,
                 constriction=1):
    dimensions = size(swarm, 1)
    particles = size(swarm, 0)

    r1 = random.rand(particles, dimensions)
    r2 = random.rand(particles, dimensions)

    # print local_position, swarm
    cognitive_knowledge = c1 * r1 * (local_position - swarm)
    social_knowledge = c2 * r2 * (ones((particles, dimensions)) * global_position[iteration, :] - swarm)

    velocity = constriction * ((inertia_weight * velocity) + cognitive_knowledge + social_knowledge)
    swarm = swarm + velocity

    return swarm, velocity


def check_swarm(swarm, search_space):
    particles = size(swarm, 0)
    dimensions = size(swarm, 1)

    lower_limit = ones((particles, dimensions)) * search_space[0]
    upper_limit = ones((particles, dimensions)) * search_space[1]

    outside_lower_limit = (swarm < lower_limit)
    outside_upper_limit = (swarm > upper_limit)

    range_space_search = upper_limit - lower_limit
    swarm = swarm * (~(outside_lower_limit + outside_upper_limit)).astype(int) + \
            ((lower_limit + random.rand(particles, dimensions) * range_space_search) * \
             (outside_lower_limit + outside_upper_limit).astype(int))

    return swarm


def pso(function, particles, search_space, iterations):
    velocity, fitness, local_best, local_position, global_best, global_position = init_pso(iterations, particles,
                                                                                           search_space)
    # print create_swarm(particles, search_space)
    swarm = create_swarm(particles, search_space)

    iteration = 0
    while iteration < iterations:
        fitness = array([abs(0 - function(solution)) for solution in swarm]).T
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
