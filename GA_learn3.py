#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author : anthony 2022 09 26 GA (基因演算法) fit TSP(旅行商人)
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys


N_CITES = 20 #DNA SIZE
CROSS_RATE = 0.5
MUTATE_RATE = 0.02
POP_SIZE = 500
N_GENERATIONS = 500

class GA_TSP(object):
    def __init__(self,DNA_SIZE,cross_rate,mutate_rate,pop_size):
        self.DNA_size = DNA_SIZE
        self.cross_rate = cross_rate
        self.mutate_rate = mutate_rate
        self.pop_size = pop_size

        self.pop = np.vstack([np.random.permutation(DNA_SIZE) for i in range(pop_size)])
    

    def translateDNA(self,DNA,city_position):#get citie's crood in order
        line_x = np.empty_like(DNA,dtype=np.float64)
        line_y = np.empty_like(DNA,dtype=np.float64)
        for i,d in enumerate(DNA):
            city_crood = city_position[d]
            line_x[i,:] = city_crood[:,0]
            line_y[i,:] = city_crood[:,1]
        
        return line_x,line_y
    

    def get_fitness(self,line_x,line_y):
        total_dis = np.empty_like(line_x.shape[0],shape = line_x.shape[0],dtype=np.float64)
        for i ,(xs,ys) in enumerate(zip(line_x,line_y)):
            #sqrt((x1-x0)**2 + (y1-y0)**2)
            total_dis[i] = np.sum(np.sqrt(np.diff(xs) ** 2 + np.diff(ys)**2))
        
        #假設路線A走 總共走5m ,路線B 總共走 4.3m ,沒用exp 他們的差距就是0.7
        #用 exp() 可以更好的放大 這兩個路線之間的差距,在對選擇最優的基因有好的影響 
        fitness = np.exp(self.DNA_size * 2 / total_dis)
        return fitness , total_dis

    def select(self,fitness):
        idx = np.random.choice(np.arange(self.pop_size),size=self.pop_size,replace=True,p = fitness/np.sum(fitness))

        return self.pop[idx]
    
    def crossover(self,parent,pop):
        if np.random.rand() < self.cross_rate:
            i_ = np.random.randint(0,self.pop_size,size=1)
            cross_point = np.random.randint(0,2,size= self.DNA_size).astype(np.bool)
            keep_city = parent[~cross_point]
            swap_city = pop[i_,np.isin(pop[i_].ravel(),keep_city,invert=True)]
            parent[:] = np.concatenate((keep_city,swap_city))
        #ravel() = reshape(-1) 多維數組拉平成一維
        return parent

    def mutate(self,child):
        for point in range(self.DNA_size):
            if np.random.rand() < self.mutate_rate:
                swap_point = np.random.randint(0,self.DNA_size)
                swap_a , swap_b = child[point],child[swap_point]
                child[point],child[swap_point] = swap_b,swap_a

        return child

    def evolve(self,fitness):
        pop = self.select(fitness)
        pop_copy = pop.copy()
        for parent in pop:
            child = self.crossover(parent,pop_copy)
            child = self.mutate(child)
            parent[:] = child
        
        self.pop = pop

class TravelSalePerson(object):
    def __init__(self,n_citys):
        self.city_posiotion = np.random.rand(n_citys,2)
        plt.ion()
    
    def plot(self,lx,ly,total_d):
        plt.cla()
        plt.scatter(self.city_posiotion[:,0].T,self.city_posiotion[:,1].T,s=100,c='k')
        plt.plot(lx.T,ly.T,'r-')
        plt.text(-0.05,0.05,"Total distance=%.2f" % total_d,fontdict={'size':20,'color':'red'})
        plt.xlim(-0.1,1.1)
        plt.ylim(-0.1,1.1)
        plt.pause(0.01)

if __name__=="__main__":
    ga = GA_TSP(N_CITES,CROSS_RATE,MUTATE_RATE,POP_SIZE)

    env = TravelSalePerson(N_CITES)

    for generation in range(N_GENERATIONS):
        lx,ly = ga.translateDNA(ga.pop,env.city_posiotion)
        fitness,total_dis = ga.get_fitness(lx,ly)
        ga.evolve(fitness)
        best_idx = np.argmax(fitness)
        print(f"___________{generation+1}__________")
        print('Gen:',generation,'|best fit: %.2f' % fitness[best_idx],)
        env.plot(lx[best_idx],ly[best_idx],total_dis[best_idx])
    
plt.ioff()
plt.show()

            
