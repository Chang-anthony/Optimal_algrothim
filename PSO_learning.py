#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author : anthony 2022 09 24 粒子群演算法
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

#target funtion
def compute_objective_value(array):
    val = 0
    for ele in array:
        val += ele **2
    return val


class PSO(object):
    def __init__(self,pop_size,dimension,upper_bounds,lower_bounds,compute_objective_value,
    cognition_factor = 0.5, social_factor = 0.5) -> None:
    
        self.pop_size = pop_size # 母體數量
        self.dimension = dimension #這個問題的變數數量
        self.upper_bound = upper_bounds #最大值限制
        self.lower_bound = lower_bounds #最小值限制

        self.solutions = [] #current solution
        self.indivdual_best_sloution = [] # indivdual_best_sloution
        self.indivdual_best_objective_value = [] #indivdual_best_value

        self.gobal_best_solution = [] #gobal best solution
        self.glbal_best_objective_value = sys.float_info.max
        self.cognition_factor = cognition_factor #particle follows its own search experience
        self.social_factor = social_factor #particle movement follows the swarm search experience
        self.compute_objective_value = compute_objective_value
    
    def initialize(self):
        min_index = 0
        min_value = sys.float_info.max

        #初始化所有粒子的值
        for i in range(self.pop_size):
            solution = []
            for d in range(self.dimension):
                rand_pos = self.lower_bound[d] + np.random.random() * (self.upper_bound[d]-self.lower_bound[d])
                solution.append(rand_pos)

            self.solutions.append(solution)

            #update indivual solution
            self.indivdual_best_sloution.append(solution)
            objective = self.compute_objective_value(solution)
            self.indivdual_best_objective_value.append(objective)

            #record the smallest objective val
            if (objective < min_value):
                min_index = i
                min_value = objective
        
        #update so far the best solution
        self.gobal_best_solution = self.solutions[min_index].copy()
        self.glbal_best_objective_value = min_value

    def move_to_new_positions(self):
        for i,solution in enumerate(self.solutions):
            alpha = self.cognition_factor * np.random.random()
            beta = self.social_factor * np.random.random()
            for d in range(self.dimension):
                v = alpha * (self.indivdual_best_sloution[i][d]- self.solutions[i][d]) +\
                    beta*(self.gobal_best_solution[d] - self.solutions[i][d])
                

                self.solutions[i][d] += v
                self.solutions[i][d] = min(self.solutions[i][d],self.upper_bound[d])
                self.solutions[i][d] = max(self.solutions[i][d],self.lower_bound[d])

    def update_best_solutions(self):
        for i,solution in enumerate(self.solutions):
            obj_val = self.compute_objective_value(solution)

            #update indiviual solution
            if(obj_val < self.indivdual_best_objective_value[i]):
                self.indivdual_best_sloution[i] = solution
                self.indivdual_best_objective_value[i] = obj_val

                if(obj_val < self.glbal_best_objective_value):
                    self.gobal_best_solution = solution
                    self.glbal_best_objective_value = obj_val


    #target funtion
    def compute_objective_value(array):
        val = 0
        for ele in array:
            val += ele **2
        return val



if __name__ == "__main__":
    pop_size = 5
    solver = PSO(pop_size,2,[100,100],[-100,-100],compute_objective_value)
    solver.initialize()

    #target funtion
    def compute_objective_value(array):
        val = 0
        for ele in array:
            val += ele **2
        return val


    for iteration in range(20):
        solver.move_to_new_positions()
        solver.update_best_solutions()

        #print
        print(f"==============iteration{iteration+1}============")
        for i ,solution in enumerate(solver.solutions):
            print(f"solution {i+1}")
            print(f"{solution}:{compute_objective_value(solution)}")
        print("golbal best solution:")
        print(f"{solver.gobal_best_solution}:{solver.glbal_best_objective_value}")

