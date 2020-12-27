import numpy
import ga
import matplotlib.pyplot as pyplot


def show_plot(best_outputs):
    pyplot.plot(best_outputs)
    pyplot.xlabel("Итерации")
    pyplot.ylabel("Пригодность")
    pyplot.show()


if __name__ == '__main__':

    # Входные данные уравнения.
    equation_inputs = [4, -2, 3.5, 5, -11, -4.7]

    # Количество весов, которые мы хотим оптимизировать.
    num_weights = len(equation_inputs)
    sol_per_pop = 12
    num_parents_mating = 8

    # Определение численности населения.
    population_size = (sol_per_pop,
                       num_weights)  # У популяции будет хромосома sol_per_pop, где каждая хромосома имеет num_weights генов.

    # Создание начальной популяции.
    new_population = numpy.random.uniform(low=-4.0, high=4.0, size=population_size)
    print(new_population)

    best_outputs = []
    num_generations = 2000

    for generation in range(num_generations):
        print("Generation : ", generation)

        # Измерение приспособленности каждой хромосомы в популяции.
        fitness = ga.calculate_population_fitness(equation_inputs, new_population)
        print("Fitness")
        print(fitness)

        best_outputs.append(numpy.max(numpy.sum(new_population * equation_inputs, axis=1)))
        # Лучший результат в текущей итерации.
        print("Best result : ", numpy.max(numpy.sum(new_population * equation_inputs, axis=1)))

        # Выбор лучших родителей для скрещивания.
        parents = ga.select_mating_pool(new_population, fitness,
                                        num_parents_mating)
        print("Parents")
        print(parents)

        # Создание следующего поколения с использованием кроссовера.
        offspring_crossover = ga.crossover(parents,
                                           offspring_size=(population_size[0] - parents.shape[0], num_weights))
        print("Crossover")
        print(offspring_crossover)

        # Добавление некоторых вариаций потомству с помощью мутаций.
        offspring_mutation = ga.mutation(offspring_crossover, num_mutations=2)
        print("Mutation")
        print(offspring_mutation)

        # Создание новой популяции на основе родителей и потомков.
        new_population[0:parents.shape[0], :] = parents
        new_population[parents.shape[0]:, :] = offspring_mutation

    # Получение лучшего решения после итерации, завершающей все поколения.
    # Сначала рассчитывается пригодность для каждого решения в последнем поколении.
    fitness = ga.calculate_population_fitness(equation_inputs, new_population)

    # Возвращаем индекс решения, соответствующий лучшей пригодности.
    best_match_idx = numpy.where(fitness == numpy.max(fitness))

    print("Лучшее решение: ", new_population[best_match_idx, :])
    print("Лучший показатель приходности решения : ", fitness[best_match_idx])

    show_plot(best_outputs)
