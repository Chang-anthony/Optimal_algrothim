#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author : anthony 2022 09 25 爬山演算法
"""

import random
import numpy as np
import matplotlib.pyplot as plt
import math
import sys

#target funtion
def fitness(x):
    return math.sin(x*x) + 2.0 * math.cos(2.0 * x)

class ClimbM():
    def __init__(self,pop_size,limit,vlimit,dimension = 1 ,max_iter = 1000):
        self.pop_size = pop_size #同時幾個人搜索
        self.limit = limit #搜索範圍
        self.vlimit = vlimit #搜所速度
        self.dimension = dimension #幾維問題 
        self.max_iter = max_iter #最大迭代數

        self.x = None
        self.y = None

        self.best_solution = None #待尋找的最佳解
        self.best_value = sys.float_info.max #找到的最佳解的函式值

    # def initial(self):    

    def update(self):

        self.x = self.limit[0] + (self.limit[1] - self.limit[0]) * np.random.rand(self.pop_size,self.dimension)
        while True:
            self.max_iter -= 1
            if self.best_value <= 0.001 or self.max_iter == 0:
                break

            delta = self.vlimit[0] + (self.vlimit[1] - self.vlimit[0]) * np.random.rand(self.pop_size,self.dimension)

            self.x = self.x + delta
            self.x[self.x >= self.limit[1]] = self.limit[1]
            self.x[self.x <= self.limit[0]] = self.limit[0]


            value = []
            for i in range(self.pop_size):
                value.append(fitness(self.x[i]))
        
            self.y = np.array(value)

            index = np.argmin(self.y)
            if self.y[index] < self.best_value:
                self.best_solution = self.x[index]
                self.best_value = self.y[index]

        return self.best_solution,self.best_value  

if __name__ == "__main__":
    pop_size = 1
    solver = ClimbM(pop_size,[0,10],[-0.01,0.01],1)
    # solver.initial()

    best_solution,best_value = solver.update()

    # for iteration in range():
    #     solver.move_to_next_position()
    #     solver.update_best_solutions()

    #     #print
    #     print(f"==============iteration{iteration+1}============")
    #     # for i ,solution in enumerate(solver.current_solutions):
    #     #     print(f"solution {i+1}")
    #     #     print(f"{solution}:{fitness(solution)}")
    print(solver.max_iter)
    print("golbal best solution:")
    print(f"{solver.best_solution}:{solver.best_value}")



