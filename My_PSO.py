#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author : anthony 2022 09 24 粒子群演算法(PSO)
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
# np.random.seed(1)
print("hello")

#target function
def fitness(x):
    '''
        此待優化問題可改變,這邊都假設為解決二維問題
    '''
    #return np.sum( x**2 ) 
    return 20 + x[0] * x[0] + x[1] * x[1] - 10 * np.cos(2*np.pi * x[0]) - 10 * np.cos(2 * np.pi * x[1])

class PSO(object):
    def __init__(self,n_size,dimension,x_limit,v_limit,weight = 0.8,indiviual_factor = 0.5,social_factor = 0.5):
        self.n_size = n_size #粒子數量
        self.dimension = dimension #優化問題的維數
        self.x_limit = x_limit #搜索世界的位置限制
        self.v_limit = v_limit #粒子搜索的速度限制
        
        self.weight = weight #慣性權重 (更新速度的公式用)
        self.indiviual_factor = indiviual_factor #個體粒子的學習率 (更新速度的公式用)
        self.social_factor = social_factor #群體的學習率 (更新速度的公式用)

        self.current_solutions = None #現在的解

        self.indiviual_best_solutions = None #個體粒子的最優解
        self.indiviual_best_value = None #個體粒子的最優值

        self.gobal_best_solutions = None #群體的最優解
        self.gobol_best_value = sys.float_info.max  #群體的最優值,初始化都為最大值
        
    def initial(self):
        #都假設待解問題為 二維問題
        #initial position
        x = self.x_limit[0] + (self.x_limit[1] - self.x_limit[0]) * np.random.rand(self.n_size,self.dimension)
        #initial vel
        self.now_v = np.random.rand(self.n_size,self.dimension) 

        self.current_solutions = x.copy()

        #初始覆蓋每個種群的最佳位置
        self.indiviual_best_solutions = x.copy()
        values = []
        #種群的個人最佳解
        for i in range(self.n_size):
            values.append(fitness(self.current_solutions[i]))
        
        self.indiviual_best_value = np.array(values)

        index = np.argmin(self.indiviual_best_value) #find small value index
        #初始化世界最佳解與最佳值
        if (self.indiviual_best_value[index] < self.gobol_best_value):
            self.gobal_best_solutions = self.indiviual_best_solutions[index].copy()
            self.gobol_best_value = self.indiviual_best_value[index].copy()

    def move_to_next_position(self):
        vi_ = self.now_v.copy()#get i-1 vel
        alpha = self.indiviual_factor * np.random.random() #indiviual factor
        beta = self.social_factor  * np.random.random() # social factor 

        #update v
        self.now_v = self.weight * vi_ + alpha * (self.indiviual_best_solutions - self.current_solutions) + beta *(self.gobal_best_solutions - self.current_solutions) 
        #check v_limit
        self.vcilp()

        #update current colution 
        self.current_solutions = self.current_solutions + self.now_v
        #check x_limit
        self.xcilp()

    def update_best_solutions(self):
        values = []
        #與種群的個人最佳解
        for i in range(self.n_size):
            values.append(fitness(self.current_solutions[i]))
        fx = np.array(values)

        for i in range(self.n_size):
            if fx[i] < self.indiviual_best_value[i]:
                self.indiviual_best_value[i] = fx[i].copy()
                self.indiviual_best_solutions[i,:] = self.current_solutions[i,:].copy()
        
        index = np.argmin(self.indiviual_best_value) #find small value index
        #初始化世界最佳解與最佳值
        if (self.indiviual_best_value[index] < self.gobol_best_value):
            self.gobal_best_solutions = self.indiviual_best_solutions[index].copy()
            self.gobol_best_value = self.indiviual_best_value[index].copy()

    def vcilp(self):
        '''
            check v_limit
        '''
        #check v_limit
        self.now_v[self.now_v > self.v_limit[1]] = self.v_limit[1]
        self.now_v[self.now_v < self.v_limit[0]] = self.v_limit[0]

    def xcilp(self):
        '''
            check x_limit
        '''
        #check x_limit
        self.current_solutions[self.current_solutions > self.x_limit[1]] = self.x_limit[1]
        self.current_solutions[self.current_solutions < self.x_limit[0]] = self.x_limit[0]

if __name__ == "__main__":
    # test = np.array([1,2])
    
    # x = np.array([[1,2],
    #      [3,4]   ])

    # x2 = np.array([[4,5],
    #                 [6,7]   ])

    # test = fitness(np.array([x.T[:,0],x.T[:,1]]))
    # print(x2 - x)
    # print(x.T[:,0])
    # print(x.T[:,1])

    # print(test)
    iter = 100

    pop_size = 50
    solver = PSO(pop_size,2,[-100,100],[-10,10])
    solver.initial()

    for iteration in range(iter):
        solver.move_to_next_position()
        solver.update_best_solutions()

        #print
        print(f"==============iteration{iteration+1}============")
        # for i ,solution in enumerate(solver.current_solutions):
        #     print(f"solution {i+1}")
        #     print(f"{solution}:{fitness(solution)}")
        print("golbal best solution:")
        print(f"{solver.gobal_best_solutions}:{solver.gobol_best_value}")

