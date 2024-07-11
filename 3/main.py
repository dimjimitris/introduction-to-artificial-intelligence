from genetic_algorithm import GeneticAlgorithm
import matplotlib.pyplot as plt
import numpy as np

def calculate_average_of_lists(lst):
    return np.mean(lst, axis=0)

# Define which experiment to run
experiment = 1

if __name__ == "__main__":


    # Experiment 1: Finding genetic algorithm parameters
    if experiment == 1:

        print(f"\nExperiment 1: Finding genetic algorithm parameters\n")
        print("- Running algorithm with different parameters...\n\n")

        # Define the default configuration for the GA
        default_config = {
            'population_size': 100,
            'mutation_rate': 0.01,
            'mutation_strength': 0.2,
            'crossover_rate': 0.9,
            'num_generations': 100
        }

        # Initialize a seed for reproducibility
        seed = 27

        # Lists of parameters values to iterate over, excluding the default
        parameters = {
            'population_size': [200, 500],
            'mutation_rate': [0.02, 0.04],
            'mutation_strength': [0.1, 0.5],
            'crossover_rate': [0.8, 0.7],
            'num_generations': [200, 500]
        }

        configurations = [default_config] # List of configurations to run

        # Create configurations that vary one parameter at a time
        for param, values in parameters.items():
            for value in values:
                new_config = default_config.copy()
                new_config[param] = value
                configurations.append(new_config)

        results = [] # This will hold the results of each configuration

        # Run the GA for each configuration and store results
        for config in configurations:
            ga = GeneticAlgorithm(**config)
            best_solutions, best_fitness_values, average_fitness_values = ga.evolve(seed)
            generations = range(1, len(best_fitness_values) + 1)
            print(f"(PS, MR, MS, CR, G) = {list(config.values())}")
            print(f"Best solution found: {best_solutions[-1]} with fitness: {best_fitness_values[-1]}\n")
            results.append((best_solutions[-1], best_fitness_values, average_fitness_values))

        # -- Plotting setup --
        # Determine how many figures are needed
        max_subplots_per_fig = 6
        num_figs = len(results) // max_subplots_per_fig + (len(results) % max_subplots_per_fig > 0)

        # Plot each result in separate subplots
        for fig_idx in range(num_figs):

            # Indexes for calculating which subset of the 'results' list should be plotted in this figure
            start_idx = fig_idx * max_subplots_per_fig
            end_idx = min(start_idx + max_subplots_per_fig, len(results))
            
            fig, axs = plt.subplots(2, 3, figsize=(14, 7))
            axs = axs.flatten()

            for plot_idx, result_idx in enumerate(range(start_idx, end_idx)):
                best_solutions, best_fitness_values, average_fitness_values = results[result_idx]
                generations = range(1, len(best_fitness_values) + 1)

                axs[plot_idx].semilogy(generations, best_fitness_values, label=f'Best Fitness')
                axs[plot_idx].semilogy(generations, average_fitness_values, label=f'Average Fitness')
                axs[plot_idx].set_title(f'(PS, MR, MS, CR, G) = {list(configurations[result_idx].values())}')
                axs[plot_idx].set_xlabel('Generation')
                axs[plot_idx].set_ylabel('Fitness (log scale)')
                axs[plot_idx].legend()

            plt.tight_layout()
        plt.show()

    # Experiment 2: Randomness in genetic algorithm
    if experiment == 2:

        print(f"\nExperiment 2: Randomness in genetic algorithm\n")
        print("- Running algorithm with 5 more seeds...\n\n")
        
        # List of seeds to test the algorithm with
        seeds = [27, 18, 42, 80, 33, 58]

        # Initialize GA with the best parameters configuration found in Experiment 1
        ga = GeneticAlgorithm(
                population_size=500,
                mutation_rate=0.01,
                mutation_strength=0.2,
                crossover_rate=0.9,
                num_generations=100,
            )

        # Lists to store results for each seed
        local_best_fitness_values = np.zeros((len(seeds), ga.num_generations))
        local_avg_fitness_values = np.zeros((len(seeds), ga.num_generations))

        # Run the GA for each seed and store results
        for idx, seed in enumerate(seeds):
            best_solutions, best_fitness_values, average_fitness_values = ga.evolve(seed)
            local_best_fitness_values[idx] = best_fitness_values
            local_avg_fitness_values[idx] = average_fitness_values
            print(f"Best solution found for Seed {seed}: {best_solutions[-1]} with fitness: {average_fitness_values[-1]}\n")
        
        # Calculate the mean fitness values across all seeds to evaluate stability
        mean_best_fitness_values = calculate_average_of_lists(local_best_fitness_values)
        mean_avg_fitness_values = calculate_average_of_lists(local_avg_fitness_values)
        std_dev_best_fitness_value = np.std(local_best_fitness_values) # Standard deviation of best fitness values across all seeds
        
        print(f"Averaged Best Fitness value: {mean_best_fitness_values[-1]} with Standard Deviation: {std_dev_best_fitness_value}\n")

        generations = range(1, len(best_fitness_values) + 1)

        _, axs = plt.subplots(2, len(seeds) // 2, figsize=(14, 7))
        axs = axs.flatten()

        # Plotting GA's results for each seed
        for i, seed in enumerate(seeds):
            axs[i].semilogy(generations, local_best_fitness_values[i], label=f'Best Fitness')
            axs[i].semilogy(generations, local_avg_fitness_values[i], label=f'Average Fitness')
            axs[i].set_title(f'Seed = {seed}')
            axs[i].set_xlabel('Generation')
            axs[i].set_ylabel('Fitness (log scale)')
            axs[i].legend()

        plt.tight_layout()

        # Plotting the Averaged Fitness across all seeds
        plt.figure(figsize = (12, 6))
        plt.semilogy(generations, mean_best_fitness_values, label = 'Best Fitness')
        plt.semilogy(generations, mean_avg_fitness_values, label = 'Average Fitness')
        plt.title('Averaged Fitness across all seeds')
        plt.xlabel('Generation')
        plt.ylabel('Fitness (log scale)')
        plt.legend()


        print("\n- Running algorithm again with decreasing population sizes, showing results averaged across all seeds...\n\n")
        population_sizes = [500, 250, 125, 50]

        # Dictionaries to store the aggregated fitness values and standard deviation
        # for each population size, across all seeds
        final_best_fitness_values = {}
        final_avg_fitness_values = {}
        std_dev_best_fitness_value = {}

        #Run the GA for each population size and store results
        for pop_s in population_sizes:
            # Initialize arrays to store results for each seed with the current population size
            local_best_fitness_values = np.zeros((len(seeds), ga.num_generations))
            local_avg_fitness_values = np.zeros((len(seeds), ga.num_generations))
            
            # Iterate over each seed and store results
            for idx, seed in enumerate(seeds):
                ga.population_size = pop_s
                best_solutions, best_fitness_values, average_fitness_values = ga.evolve(seed)
                local_best_fitness_values[idx] = best_fitness_values
                local_avg_fitness_values[idx] = average_fitness_values
            
            # Aggregate results
            final_best_fitness_values[pop_s] = calculate_average_of_lists(local_best_fitness_values)
            final_avg_fitness_values[pop_s] = calculate_average_of_lists(local_avg_fitness_values)
            std_dev_best_fitness_value[pop_s] = np.std(local_best_fitness_values) # Calculare standard deviation across all seeds

            print(f"Population Size = {pop_s}:")
            print(f"Best Fitness = {final_best_fitness_values[pop_s][-1]}")
            print(f"Average Fitness = {final_avg_fitness_values[pop_s][-1]}")
            print(f"Standard Deviation of Best Fitness: {std_dev_best_fitness_value[pop_s]}\n")

        # Plotting setup
        generations = range(1, len(best_fitness_values) + 1)

        _, axs = plt.subplots(2, len(population_sizes) // 2, figsize=(13, 7))
        axs = axs.flatten()

        for i, pop_s in enumerate(population_sizes):
            axs[i].semilogy(generations, local_best_fitness_values[i], label=f'Best Fitness')
            axs[i].semilogy(generations, local_avg_fitness_values[i], label=f'Average Fitness')
            axs[i].set_title(f'Population Size = {pop_s}')
            axs[i].set_xlabel('Generation')
            axs[i].set_ylabel('Fitness (log scale)')
            axs[i].legend()

        plt.tight_layout()
        plt.show()

    # Experiment 3: Crossover impact
    if experiment == 3:
        print(f"\nExperiment 3: Crossover impact\n")
        print("- Running algorithm with different crossover rates, showing results averaged across all seeds...\n\n")

        # List of crossover rates to test
        crossover_rates = [0.3, 0.5, 0.75, 0.9, 1.0]
        seeds = [27, 18, 42, 80, 33, 58]

        # dictionaries to store the aggregated fitness values for each crossover rate
        final_best_fitness_values = {}
        final_avg_fitness_values = {}
        for crossover_rate in crossover_rates:
            local_best_fitness_values = np.zeros((len(seeds), 100))
            local_avg_fitness_values = np.zeros((len(seeds), 100))
            for idx, seed in enumerate(seeds):
                ga = GeneticAlgorithm(
                    population_size=500,
                    mutation_rate=0.01,
                    mutation_strength=0.2,
                    crossover_rate=crossover_rate,
                    num_generations=100,
                )

                best_solutions, best_fitness_values, average_fitness_values = ga.evolve(seed)

                # take average across all seeds
                local_best_fitness_values[idx] = best_fitness_values
                local_avg_fitness_values[idx] = average_fitness_values
            
            # perform the aggregate function: take the average of the fitness values across all seeds
            final_best_fitness_values[crossover_rate] = calculate_average_of_lists(local_best_fitness_values)
            final_avg_fitness_values[crossover_rate] = calculate_average_of_lists(local_avg_fitness_values)

            print(f"Crossover Rate = {crossover_rate}:")
            print(f"Final Best Fitness: {final_best_fitness_values[crossover_rate][-1]}")
            print(f"Final Average Fitness: {final_avg_fitness_values[crossover_rate][-1]}\n")

        generations = range(1, len(best_fitness_values) + 1)

        _, axs = plt.subplots(2, (len(crossover_rates) + 1) // 2, figsize=(14, 7))
        axs = axs.flatten()

        for i, crossover_rate in enumerate(final_best_fitness_values):
            axs[i].semilogy(generations, final_best_fitness_values[crossover_rate], label=f'Best Fitness')
            axs[i].semilogy(generations, final_avg_fitness_values[crossover_rate], label=f'Average Fitness')
            axs[i].set_title(f'Crossover Rate = {crossover_rate}')
            axs[i].set_xlabel('Generation')
            axs[i].set_ylabel('Average Fitness Across All Seeds(log scale)')
            axs[i].legend()

        plt.tight_layout()
        plt.show()

    # Experiment 4: Mutation and Convergence
    if experiment == 4:
        print(f"\nExperiment 4: Mutation and Convergence\n")
        print("- Running algorithm increasing mutation rate and mutation strength, showing results averaged across all seeds...\n\n")

        # parameters used for testing
        # base mutation rate is 0.01 and base mutation strength is 0.2. We test by increasing those values
        mutation_rates = [0.01, 0.02, 0.04]
        mutation_strengths = [0.2, 0.4]
        seeds = [27, 18, 42, 80, 33, 58]

        # dictionaries to store the aggregated fitness values for each mutation rate and mutation strength
        final_best_fitness_values = {}
        final_avg_fitness_values = {}
        for mutation_rate in mutation_rates:
            # initialize dictionaries for each mutation rate
            final_best_fitness_values[mutation_rate] = {}
            final_avg_fitness_values[mutation_rate] = {}
            for mutation_strength in mutation_strengths:
                local_best_fitness_values = np.zeros((len(seeds), 100))
                local_avg_fitness_values = np.zeros((len(seeds), 100))
                for idx, seed in enumerate(seeds):
                    ga = GeneticAlgorithm(
                        population_size=500,
                        mutation_rate=mutation_rate,
                        mutation_strength=mutation_strength,
                        crossover_rate=0.9,
                        num_generations=100,
                    )

                    best_solutions, best_fitness_values, average_fitness_values = ga.evolve(seed)

                    local_best_fitness_values[idx] = best_fitness_values
                    local_avg_fitness_values[idx] = average_fitness_values
                
                # perform the aggregate function: take the average of the fitness values across all seeds
                final_best_fitness_values[mutation_rate][mutation_strength] = calculate_average_of_lists(local_best_fitness_values)
                final_avg_fitness_values[mutation_rate][mutation_strength] = calculate_average_of_lists(local_avg_fitness_values)

                print(f"MR = {mutation_rate}, MS = {mutation_strength}:")
                print(f"Final Best Fitness = {final_best_fitness_values[mutation_rate][mutation_strength][-1]}")
                print(f"Final Average Fitness = {final_avg_fitness_values[mutation_rate][mutation_strength][-1]}\n")

        generations = range(1, len(best_fitness_values) + 1)

        _, axs = plt.subplots(2, 3, figsize=(14, 7))
        axs = axs.flatten()

        i = 0
        for mutation_rate in mutation_rates:
            for mutation_strength in mutation_strengths:
                if i % 2 == 0:
                    idx = i // 2
                else:
                    idx = 3 + i // 2

                axs[idx].semilogy(generations, final_best_fitness_values[mutation_rate][mutation_strength], label=f'Best Fitness')
                axs[idx].semilogy(generations, final_avg_fitness_values[mutation_rate][mutation_strength], label=f'Average Fitness')
                axs[idx].set_title(f'Mutation Rate = {mutation_rate}, Mutation Strength = {mutation_strength}')
                axs[idx].set_xlabel('Generation')
                axs[idx].set_ylabel('Average Fitness Across All Seeds(log scale)')
                axs[idx].legend()
                i += 1

        plt.tight_layout()
        plt.show()