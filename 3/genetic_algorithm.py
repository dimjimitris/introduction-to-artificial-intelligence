import random
import numpy as np

from functions import rastrigin_2d


def set_seed(seed: int) -> None:
    # Set fixed random seed to make the results reproducible
    random.seed(seed)
    np.random.seed(seed)


class GeneticAlgorithm:
    def __init__(
        self,
        population_size: int,
        mutation_rate: float,
        mutation_strength: float,
        crossover_rate: float,
        num_generations: int,
    ):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.crossover_rate = crossover_rate
        self.num_generations = num_generations


    def initialize_population(self) -> np.ndarray:
        # Initialize the population with random values within the range [-5, 5]
        lower_bound, upper_bound = -5 , 5
        population = np.random.uniform(low = lower_bound, high = upper_bound, size = (self.population_size, 2))
        return population


    def evaluate_population(self, population) -> np.ndarray:
        # Calculate the fitness for each individual in the population
        fitness_values = np.array([rastrigin_2d(individual[0], individual[1]) for individual in population])
        return fitness_values


    def selection(self, population, fitness_values) -> np.ndarray:
        # Rank the population based on fitness in ascending order
        # (with Rastrigin function, lower value means better fitness value)
        ranked_indices = np.argsort(fitness_values)
        ranked_population = population[ranked_indices]

        # Generate selection probabilities inversely proportional to rank
        selection_probabilities = np.linspace(start = 1, stop = 0, num = self.population_size)
        selection_probabilities /= selection_probabilities.sum() # Normalize to sum to 1

        # Select parents of the new generation based on their probability
        selected_indices = np.random.choice(self.population_size, size = self.population_size, replace = True, p = selection_probabilities)
        selected_parents = ranked_population[selected_indices]

        return selected_parents


    def crossover(self, parents) -> np.ndarray:
        # We assume the new generation will be of the same size of the old one
        offspring = np.empty((self.population_size, 2))

        # Go through the parents in pairs and create one offspring
        for i in range(0, self.population_size, 2):
            parent1 = parents[i]
            parent2 = parents[i + 1] if (i + 1) < len(parents) else parents[0]

            if np.random.rand() < self.crossover_rate: # Crossover occurs with a probability of 'crossover_rate'
                alpha = np.random.rand() # Random crossover point

                offspring[i] = alpha * parent1 + (1 - alpha) * parent2
                if (i + 1) < self.population_size:
                    offspring[i + 1] = (1 - alpha) * parent1 + alpha * parent2
            else:
                offspring[i] = parent1
                if (i + 1) < self.population_size:
                    offspring[i + 1] = parent2

        return offspring


    def mutate(self, individuals) -> np.ndarray:
        # Iterate over all individuals and their genes to apply mutation
        for individual in individuals:
            for gene in range(len(individual)):
                if np.random.rand() < self.mutation_rate:
                    # Apply Gaussian mutation
                    mutation = np.random.normal(0, self.mutation_strength)
                    individual[gene] += mutation
        return individuals


    def evolve(self, seed: int) -> ...:
        # Run the genetic algorithm and return the lists that contain the best solution for each generation,
        #   the best fitness for each generation and average fitness for each generation
        set_seed(seed)

        population = self.initialize_population()

        best_solutions = []
        best_fitness_values = []
        average_fitness_values = []

        for generation in range(self.num_generations):
            # Evaluate the fitness of the population
            fitness_values = self.evaluate_population(population)
            best_solution = population[np.argmin(fitness_values)]
            best_fitness = np.min(fitness_values) # The lowest is the best in our case
            average_fitness = np.mean(fitness_values)

            best_solutions.append(best_solution)
            best_fitness_values.append(best_fitness)
            average_fitness_values.append(average_fitness)

            # Rank selection
            parents_for_reproduction = self.selection(population, fitness_values)

            # Crossover
            offspring = self.crossover(parents_for_reproduction)

            # Mutation
            population = self.mutate(offspring)

        return best_solutions, best_fitness_values, average_fitness_values
