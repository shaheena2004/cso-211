import numpy as np
import os
import random
from src.predict import predict
from src.edit import make_store_matrix
from src.edit import weighted_edit_distance

from src.dataset_names import DATASET_NAME
from datetime import datetime
from src.exhaustive_search import exhaustive_search



class Chromosome:
    def __init__(self, limit=1, size=3, insert=None, delete=None, replace=None):
        self.LIMIT = limit
        self.size = size

        if insert is None:
            insert = random.uniform(0, self.LIMIT)
        if delete is None:
            delete = random.uniform(0, self.LIMIT)
        if replace is None:
            replace = random.uniform(0, self.LIMIT)

        self.genes = np.array([insert, delete, replace])
        self.fitness = .0


class Genetic:
    def __init__(self,  P_Mutate=0.1, P_Tournament=0.75,
                 R_Mutate=0.25, sizeOfGroup=30, maxGeneration=20):
        self.P_Mutate = P_Mutate
        self.P_Tournament = P_Tournament
        self.R_Mutate = R_Mutate
        self.SIZE_OF_GROUP = sizeOfGroup
        self.MAX_GENERATION = maxGeneration
        self.chromosomeList = []
    def initChromosome(self):
        for _ in range(self.SIZE_OF_GROUP):
            self.chromosomeList.append(Chromosome())
            self.evaluate_chromosome(self.chromosomeList[-1])

    def evaluate_chromosome(self, chromosome):

        make_store_matrix(chromosome.genes[0], chromosome.genes[1], chromosome.genes[2])

        auc = predict()

        chromosome.fitness = auc

    def selectParent(self):
        p1 = random.randrange(self.SIZE_OF_GROUP)
        p2 = random.randrange(self.SIZE_OF_GROUP)
        while p1 == p2:
            p2 = random.randrange(self.SIZE_OF_GROUP)

        if self.chromosomeList[p1].fitness < self.chromosomeList[p2].fitness:
            p1, p2 = p2, p1

        if self.P_Tournament > random.random():
            return p1
        else:
            return p2

    def mutation(self, chromosome):
        for i in range(chromosome.size):
            if random.random() < self.P_Mutate:
                chromosome.genes[i] += random.uniform(-1, 1) * self.R_Mutate
                chromosome.genes[i] = max(chromosome.genes[i], 0)
                chromosome.genes[i] = min(chromosome.genes[i], chromosome.LIMIT)

        return chromosome

    def crossOver(self, p1, p2):
        mother, father = self.chromosomeList[p1], self.chromosomeList[p2]
        offspring = Chromosome()

        offspring.genes = (mother.genes + father.genes) / 2.
        return offspring

    def replace(self, p1, p2, offspring):
        i = p2 if self.chromosomeList[p1].fitness > self.chromosomeList[p2].fitness else p1
        self.chromosomeList[i].genes = offspring.genes
        self.chromosomeList[i].fitness = offspring.fitness

    def print_population_statistics(self, epoch):
        fSum, fMax = .0, -1e9

        for chromosome in self.chromosomeList:
            fSum += chromosome.fitness
            fMax = max(fMax, chromosome.fitness)

        print(
            f'{epoch}/{self.MAX_GENERATION} average fitness: {round(fSum / self.SIZE_OF_GROUP, 4)}, max fitness: {round(fMax, 4)}')

    def getBestFitness(self):
        fMax, iMax = -1e9, -1

        for i, chromosome in enumerate(self.chromosomeList):
            if chromosome.fitness > fMax:
                fMax = chromosome.fitness
                iMax = i
        return iMax

    def run(self):
        self.initChromosome()
        epoch = 0
        while epoch < self.MAX_GENERATION:
            p1 = self.selectParent()
            p2 = self.selectParent()
            if p1 == p2:
                continue

            offspring = self.crossOver(p1, p2)
            self.mutation(offspring)
            self.evaluate_chromosome(offspring)
            self.replace(p1, p2, offspring)
            self.print_population_statistics(epoch)

            epoch += 1

        weights_vs_auc = np.genfromtxt(f"./output/weight_vs_auc_{DATASET_NAME}.csv", delimiter=",", dtype=np.float32)
        for chromosome in self.chromosomeList:
            if not any(np.array_equal(row, np.array((chromosome.genes[0], chromosome.genes[1], chromosome.genes[2], chromosome.fitness))) for row in weights_vs_auc):
                weights_vs_auc = np.vstack((weights_vs_auc, np.array((chromosome.genes[0], chromosome.genes[1], chromosome.genes[2], chromosome.fitness))))
        np.savetxt(f"./output/weight_vs_auc_{DATASET_NAME}.csv", weights_vs_auc , delimiter=",")
            
        iMax = self.getBestFitness()
        bestChromosome = self.chromosomeList[iMax]
        return bestChromosome.genes , bestChromosome.fitness

    generations = int(input("Enter no.of generations we need to chech for:"))
    st_time = datetime.now()

    print("\nRUNNING THE GENETIC ALGORITHM:")
    genetic = Genetic(maxGeneration=generations)
    optimal_weights, max_auc = genetic.run()
    en_time = datetime.now()
    print(f"The Genetic Algorithm ran for {generations} generations : optimal icost {optimal_weights[0]}, dcost {optimal_weights[1]}, rcost {optimal_weights[2]}, max_auc {round(max_auc, 5)}")
    print(f'Time taken: {round((en_time-st_time).total_seconds())} s\n')