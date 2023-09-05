#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author : anthony 2022 09 26 GA (基因演算法) fit text
"""

import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys


TARGET_PARASE = 'You get it!' #target DNA
POP_SIZE = 300                #population size
CROSSOVER_RATE = 0.6              #mating probability (DNA crossover)
MUTATION_RATE = 0.01          #mutation probability
N_GENERATIONS = 1000

DNA_SIZE = len(TARGET_PARASE)
TARGET_ASCII = np.fromstring(TARGET_PARASE ,dtype= np.uint8) #convert string to number
ASCII_BOUND = [32,126]

class GA_TEXT(object):
    def __init__(self,DNA_SIZE,DNA_BOUND,cross_rate,mutation_rate,pop_size):
        self.DNA_SIZE = DNA_SIZE
        self.DNA_BOUND = DNA_BOUND
        self.DNA_BOUND[1] += 1
        self.cross_rate = cross_rate
        self.mutation_rate = mutation_rate
        self.pop_size = pop_size

        #int8 for convert to ASCII
        self.pop = np.random.randint(*DNA_BOUND,size=(self.pop_size,self.DNA_SIZE)).astype(np.int8)

    def transleDNA(self,DNA): #convert to readable string
        return DNA.tostring().decode('ascii')

    #0 沿著列去相加,1 沿著行去相加
    def getfitness(self):
        match_count = (self.pop == TARGET_ASCII).sum(axis=1)
        return match_count 

    def select(self):
        fitness = self.getfitness() + 1e-3 #avoid np choice p = 0
        idx = np.random.choice(np.arange(self.pop_size),size=self.pop_size,replace=True,p = fitness/np.sum(fitness))
        return self.pop[idx]

    def crossover(self,parent,pop):
        if np.random.rand() < self.cross_rate:
            i_ = np.random.randint(0,self.pop_size,size=1) #choose another to cross over
            cross_points = np.random.randint(0,2,size=self.DNA_SIZE).astype(np.bool) # choice crossover points
            parent[cross_points] = pop[i_,cross_points] 

        return parent 

    def mutate(self,child):
        for point in range(self.DNA_SIZE):
            if np.random.rand() < self.mutation_rate:
                child[point] = np.random.randint(*self.DNA_BOUND) #choose ascii index
            
        return child
    
    #將生下的下一代人口覆蓋到上一代
    def evolve(self):
        pop  = self.select()
        pop_copy = self.pop.copy()
        for parent in pop:  #for every parent
            child = self.crossover(parent,pop_copy)
            child = self.mutate(child)
            parent[:] = child
        
        self.pop = pop
    
if __name__ == "__main__":
    ga = GA_TEXT(DNA_SIZE,ASCII_BOUND,CROSSOVER_RATE,MUTATION_RATE,POP_SIZE)

    for genation in range(N_GENERATIONS):
        fitnesss = ga.getfitness()
        best_DNA = ga.pop[np.argmax(fitnesss)]
        best_prase = ga.transleDNA(best_DNA)

        print(f"_______{genation+1}_________")
        print("Best Parse :", best_prase)
        if best_prase == TARGET_PARASE:
            break 
        ga.evolve()
