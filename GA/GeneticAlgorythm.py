import numpy


# В этом проекте используется десятичное представление хромосомы
def calculate_population_fitness(equation_inputs, pop):
    # Вычисление значения пригодности каждого решения в текущей популяции.
    # Фитнес-функция вычисляет сумму произведений между каждым входом и его соответствующим весом.
    fitness = numpy.sum(pop * equation_inputs, axis=1)
    return fitness


def select_mating_pool(pop, fitness, num_parents):
    # Выбор лучших особей текущего поколения в качестве родителей для производства потомства следующего поколения.
    parents = numpy.empty((num_parents, pop.shape[1]))

    for parent_num in range(num_parents):
        max_fitness_idx = numpy.where(fitness == numpy.max(fitness))
        max_fitness_idx = max_fitness_idx[0][0]
        parents[parent_num, :] = pop[max_fitness_idx, :]
        fitness[max_fitness_idx] = -99999999999

    return parents


def crossover(parents, offspring_size):
    offspring = numpy.empty(offspring_size)
    # Точка, в которой происходит кроссовер между двумя родителями. Обычно она находится в центре
    crossover_point = numpy.uint8(offspring_size[1] / 2)

    for k in range(offspring_size[0]):
        # Индекс первого сопряженного родителя.
        parent1_idx = k % parents.shape[0]
        # Индекс второго родителя для спаривания.
        parent2_idx = (k + 1) % parents.shape[0]
        # У нового потомства первая половина генов будет взята от первого родителя.
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        # У нового потомства будет вторая половина генов, взятых у второго родителя.
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]

    return offspring


def mutation(offspring_crossover, num_mutations=1):
    mutations_counter = numpy.uint8(offspring_crossover.shape[1] / num_mutations)

    # Мутация изменяет количество генов, как определено аргументом num_mutations. Изменения случайны.
    for idx in range(offspring_crossover.shape[0]):
        gene_idx = mutations_counter - 1
        for mutation_num in range(num_mutations):
            # Случайное значение, добавляемое к гену.
            random_value = numpy.random.uniform(-1.0, 1.0, 1)
            offspring_crossover[idx, gene_idx] = offspring_crossover[idx, gene_idx] + random_value
            gene_idx = gene_idx + mutations_counter

    return offspring_crossover
