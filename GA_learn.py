#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author : anthony 2022 09 25 GA (基因演算法) fit function
"""

import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

DNA_SIZE = 10 #DNA length 每個基因的長度
POP_SIZE = 100 #population size 人口基數
CROSSOVER_RATE = 0.8 #mating probability 交配率(DNA crossover)
MUTATE_RATE = 0.003 #mutation probability 基因變異率
N_GENERATIONS = 200 #總共生幾代 ,迭代幾次
X_BOUND = [0,5] #範圍限制

#target funtion  to find maximum or minimum  in this case is maximum
def fitness(x):
    return np.sin(10*x)*x + np.cos(2*x)*x  

#find non-zero fitness for selection
def get_fitness(pred):
    #計算適應度 ,因為為概率的表示 ,所以不會有負值
    #find this populations size minimum value 
    #找到裡面的最小值 並且減去 這樣就不會有負值產生
    #而加那個 極小值,是防止之後計算選擇概率時會剛好除到零的情況
    return pred + 1e-3 - np.min(pred)

#translate binary DNA to decimal and normlize it to range(0,5) 
def transleDNA(pop):
    #DNA_size 10
    # if [1,1,1,1,1,1,1,1,1,1] 
    # np.arange(DNA_SIZE)) [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] 
    # np.arange(DNA_SIZE) ** 2 [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    # dot to Dec will 1023 
    # and need to normlize to x_limit upper for below
    # 1023 / float(2 ** DNA_SIZE - 1) because in computer is 0 ~ 1023 is 2 ** 10
    # so 1023 / float(2 ** DNA_SIZE - 1) = 1 is high bound
    # and need to * x_limit high 
    # in this case  x_limit high = 5
    # so result will = 5 
    return pop.dot(2 ** np.arange(DNA_SIZE)) / float(2 ** DNA_SIZE -1) * X_BOUND[1]


#nature selection pos's fitness
def select(pop,fitness):
    #用途：从a(一维数据)中随机抽取数字，返回指定大小(size)的数组
    #replace:True表示可以取相同数字，False表示不可以取相同数字
    #数组p：与数组a相对应，表示取数组a中每个元素的概率，默认为选取每个元素的概率相同。
    idx = np.random.choice(np.arange(POP_SIZE),size=POP_SIZE,replace=True,p = fitness/fitness.sum())
    return pop[idx]


#交配過程 (mating process)
def crossover(parent,pop):
    if np.random.rand() < CROSSOVER_RATE :
        i_ = np.random.randint(0,POP_SIZE,size=1) #select another individual from pop
        cross_points = np.random.randint(0,2,size=DNA_SIZE).astype(np.bool)  #choice crossover points
        parent[cross_points] = pop[i_,cross_points]
    
    return parent

#變異在孩子得DNA片段裡,隨機挑選出來變異
def mutate(child):
    for point in range(DNA_SIZE):
        if np.random.rand() < MUTATE_RATE:
            child[point] = 1 if child[point] == 0 else 0
    return child

if __name__ == "__main__":

    pop = np.random.randint(2, size=(POP_SIZE,DNA_SIZE))

    plt.ion() # something about ploting
    x = np.linspace(*X_BOUND,200)
    plt.plot(x,fitness(x))

    for i_ in range(N_GENERATIONS):
        F_value = fitness(transleDNA(pop))

        #somthing about plot
        if 'sca' in globals(): sca.remove()
        sca = plt.scatter(transleDNA(pop),F_value,s=200,lw=0,c='red',alpha=0.5)
        plt.pause(0.05)

        #GA part (evolution)
        fitnesss = get_fitness(F_value)
        #取最適應的種群
        print("Most fitted DNA: ",pop[np.argmax(fitnesss),:])
        pop = select(pop,fitnesss)
        pop_copy = pop.copy()
        for parent in pop:
            child = crossover(parent,pop_copy)
            child = mutate(child)
            parent[:] = child # parent replace by it child

    plt.ioff()
    plt.show()