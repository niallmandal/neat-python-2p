from __future__ import print_function
import neat

from neat.reporting import ReporterSet
from neat.math_util import mean
from neat.six_util import iteritems, itervalues

class return_population(object):
    def __init__(self, config, initial_state=None):
        self.reporters = ReporterSet()
        self.config = config
        stagnation = config.stagnation_type(config.stagnation_config, self.reporters)
        self.reproduction = config.reproduction_type(config.reproduction_config,
                                                     self.reporters,
                                                     stagnation)
        if config.fitness_criterion == 'max':
            self.fitness_criterion = max
        elif config.fitness_criterion == 'min':
            self.fitness_criterion = min
        elif config.fitness_criterion == 'mean':
            self.fitness_criterion = mean
        elif not config.no_fitness_termination:
            raise RuntimeError(
                "Unexpected fitness_criterion: {0!r}".format(config.fitness_criterion))

        if initial_state is None:
            # Create a population from scratch, then partition into species.
            self.population = self.reproduction.create_new(config.genome_type,
                                                           config.genome_config,
                                                           config.pop_size)
            self.species = config.species_set_type(config.species_set_config, self.reporters)
            self.generation = 0
            self.species.speciate(config, self.population, self.generation)
        else:
            self.population, self.species, self.generation = initial_state

        self.best_genome = None

    def run(self):
        return self

class pop(neat.Population):
    def run(self, fitness_function, self2, n=None):
        if self.config.no_fitness_termination and (n is None):
            raise RuntimeError("Cannot have no generational limit with no fitness termination")
        if self2.config.no_fitness_termination and (n is None):
            raise RuntimeError("Cannot have no generational limit with no fitness termination")

        k = 0
        while n is None or k < n:
            k += 1

            self.reporters.start_generation(self.generation)
            self2.reporters.start_generation(self2.generation)

            # Evaluate all genomes using the user-provided function.
            fitness_function(list(iteritems(self.population)),list(iteritems(self2.population)), self.config)

            # Gather and report statistics.
            best1 = None
            for g in itervalues(self.population):
                if best1 is None or g.fitness > best1.fitness:
                    best1 = g
            self.reporters.post_evaluate(self.config, self.population, self.species, best1)

            best2 = None
            for g in itervalues(self2.population):
                if best2 is None or g.fitness > best2.fitness:
                    best2 = g
            self2.reporters.post_evaluate(self2.config, self2.population, self2.species, best2)

            # Track the best genome ever seen.
            if self.best_genome is None or best1.fitness > self.best_genome.fitness:
                self.best_genome = best1

            if not self.config.no_fitness_termination:
                # End if the fitness threshold is reached.
                fv = self.fitness_criterion(g.fitness for g in itervalues(self.population))
                if fv >= self.config.fitness_threshold:
                    self.reporters.found_solution(self.config, self.generation, best1)
                    break

            #Do the same with self2 <--- balances out config version
            if self2.best_genome is None or best2.fitness > self2.best_genome.fitness:
                self2.best_genome = best2

            if not self2.config.no_fitness_termination:
                # End if the fitness threshold is reached.
                fv = self2.fitness_criterion(g.fitness for g in itervalues(self2.population))
                if fv >= self2.config.fitness_threshold:
                    self2.reporters.found_solution(self2.config, self2.generation, best2)
                    break

            # Create the next generation from the current generation.
            self.population = self.reproduction.reproduce(self.config, self.species, self.config.pop_size, self.generation)
            self2.population = self2.reproduction.reproduce(self2.config, self2.species, self2.config.pop_size, self2.generation)


            # Check for complete extinction.
            if not self.species.species:
                self.reporters.complete_extinction()

                # If requested by the user, create a completely new population,
                # otherwise raise an exception.
                if self.config.reset_on_extinction:
                    self.population = self.reproduction.create_new(self.config.genome_type,
                                                                   self.config.genome_config,
                                                                   self.config.pop_size)
                else:
                    raise CompleteExtinctionException()

            if not self2.species.species:
                self2.reporters.complete_extinction()

                # If requested by the user, create a completely new population,
                # otherwise raise an exception.
                if self2.config.reset_on_extinction:
                    self2.population = self2.reproduction.create_new(self2.config.genome_type,
                                                                   self2.config.genome_config,
                                                                   self2.config.pop_size)
                else:
                    raise CompleteExtinctionException()

            # Divide the new population into species.
            self.species.speciate(self.config, self.population, self.generation)
            self2.species.speciate(self2.config, self2.population, self2.generation)

            self.reporters.end_generation(self.config, self.population, self.species)
            self2.reporters.end_generation(self2.config, self2.population, self2.species)

            self.generation += 1
            self2.generation += 1

        if self.config.no_fitness_termination:
            self.reporters.found_solution(self.config, self.generation, self.best_genome)
        if self2.config.no_fitness_termination:
            self2.reporters.found_solution(self2.config, self2.generation, self2.best_genome)

        return (self.best_genome,self2.best_genome)
