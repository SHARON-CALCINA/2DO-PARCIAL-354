# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 16:41:52 2020

@author: SHARON
"""
import numpy as np
import pandas as pd
rutas= pd.read_csv('recorridos.csv')
print(rutas)
city= rutas.iloc[:, 0].values.tolist()
print(city)
rutas=rutas.drop(rutas.columns[rutas.columns.str.contains('unnamed',case = False)],axis = 1)
distan= rutas.iloc[:].values.tolist()
print(distan)
distances = distan

print(distances)
print(distances[1][0])

totalCities = len(city)
popSize = 2000;
population = [];
import random
def setup():
  for i in range(0, popSize):
    x=random.sample(range(totalCities), totalCities)
    if(x not in population):
        population.append(x)
  print(population)
setup()
population=sorted(population)
d=[]
def calcDistance():
  for i in range(0, len(population)):
      summation = 0
      x=population[i]
      #print(x)
      for j in range(1, len(x)):
        summation += distances[x[j-1]][x[j]]
        #print(distances[x[j-1]][x[j]])
      init=distances[x[0]][x[len(x)-1]]
      d.append(summation+init)
      
calcDistance()
print(d)
bestEver=0
fitness=[]
currentBest=[]
def calculateFitness():
  
  recordDistance=1000
  for  i in range(0,len(d)):
      if d[i] < recordDistance:
          recordDistance = d[i]
          bestEver = d[i];
          currentBest=population[i]
      fitness.append( 1 / d[i])
      #fitness.append( 1 / (pow(d[i], 8) + 1))
calculateFitness()
print(fitness)


import numpy as np, random, operator, pandas as pd, matplotlib.pyplot as plt

#Create a fitness function

class Fitness:
    distances = [[0, 10, 8, 25],
             [10, 0, 4, 29],
             [8, 4, 0, 27],
             [25, 29, 27, 0]]
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness= 0.0
    
    def routeDistance(self):
         if self.distance ==0:
            pathDistance = 0
            for i in range(0, len(self.route)):
                fromCity = self.route[i]
                toCity = None
                if i + 1 < len(self.route):
                    toCity = self.route[i + 1]
                else:
                    toCity = self.route[0]
                pathDistance += distances[fromCity][toCity]
            self.distance = pathDistance
         return self.distance
    
    def routeFitness(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.routeDistance())
        return self.fitness
#Creacion de la poblacion inicial
# se randomiza el orden de las ciudades, es decir se cr4ea un individuo random
def createRoute(cityList):
    route = random.sample(cityList, len(cityList))
    return route

#Create first "population" (list of routes)
#se crea una poblacion random de un tamaño especificio
def initialPopulation(popSize, cityList):
    population = []
    for i in range(0, popSize):
        population.append(createRoute(cityList))
    return population

#Alortimo genetico
#Rango de individuos
#Esta funcion toma una poblacioon y la ordenea descendentemente usando el fitness de cada individuo
def rankRoutes(population):
    fitnessResults = {}
    for i in range(0,len(population)):
        fitnessResults[i] = Fitness(population[i]).routeFitness()
    sorted_results=sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = True)
    return sorted_results
#funcion de seleccion que sera usada para hacer la lista de rutas padre
def selection(popRanked, eliteSize):
    selectionResults = []
    df = pd.DataFrame(np.array(popRanked), columns=["Index","Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100*df.cum_sum/df.Fitness.sum()
    print(df)
    for i in range(0, eliteSize):
        selectionResults.append(popRanked[i][0])
    for i in range(0, len(popRanked) - eliteSize):
        pick = 100*random.random()
        for i in range(0, len(popRanked)):
            if pick <= df.iat[i,3]:
                print(df.iat[i,3])
                selectionResults.append(popRanked[i][0])
                break
    print('---------- SELECTION--------------')
    print(selectionResults)
    return selectionResults

# mating pool
def matingPool(population, selectionResults):
    matingpool = []
    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        matingpool.append(population[index])
    return matingpool
#funcion de crossover function para dos padres para la creacion de un hijo
def breed(parent1, parent2):
    child = []
    childP1 = []
    childP2 = []
    
    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))
    
    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    for i in range(startGene, endGene):
        childP1.append(parent1[i])
        

    childP2 = [item for item in parent2 if item not in childP1]
    print('(random) Gen inicial: ',startGene, '        (random) Gen Final: ', endGene)

    print('Padre1: ',parent1)
    print('Padre2: ',parent2)

    print('Hijo del Padre1: ',childP1)
    print('Hijo del Padre2: ',childP2)
    child = childP1 + childP2

    print('hIJO DE AMBOS PADRES: ',child)
    return child

#función para ejecutar crossover sobre todo el mating pool
def breedPopulation(matingpool, eliteSize):
    children = []
    length = len(matingpool) - eliteSize
    pool = random.sample(matingpool, len(matingpool))

    for i in range(0,eliteSize):
        children.append(matingpool[i])
    
    for i in range(0, length):
        child = breed(pool[i], pool[len(matingpool)-i-1])
        children.append(child)
    return children

#funcion para mutar una sola ruta
def mutate(individual, mutationRate):
    for swapped in range(len(individual)):
        if(random.random() < mutationRate):
            swapWith = int(random.random() * len(individual))
            
            city1 = individual[swapped]
            city2 = individual[swapWith]
            
            individual[swapped] = city2
            individual[swapWith] = city1
    return individual

#función para ejecutar la mutación en toda la población
def mutatePopulation(population, mutationRate):
    mutatedPop = []
    
    for ind in range(0, len(population)):
        mutatedInd = mutate(population[ind], mutationRate)
        mutatedPop.append(mutatedInd)
    return mutatedPop



#Cracion de la siguiente generacion
def nextGeneration(currentGen, eliteSize, mutationRate):
    popRanked = rankRoutes(currentGen)
    selectionResults = selection(popRanked, eliteSize)
    matingpool = matingPool(currentGen, selectionResults)
    children = breedPopulation(matingpool, eliteSize)
    nextGeneration = mutatePopulation(children, mutationRate)
    return nextGeneration

#ALGORITMO GENETICO
def geneticAlgorithm(population, popSize, eliteSize, mutationRate, generations):
    pop = initialPopulation(popSize, population)
    
    progress = [1 / rankRoutes(pop)[0][1]]
    
    print("Distancia Inicial: " + str(progress[0]))
    
    for i in range(1, generations+1):
        
        pop = nextGeneration(pop, eliteSize, mutationRate)
        progress.append(1 / rankRoutes(pop)[0][1])
        if i%50==0:
          print('Generacion '+str(i),"Distancia: ",progress[i])
        
    bestRouteIndex = rankRoutes(pop)[0][0]
    bestRoute = pop[bestRouteIndex]
    plt.plot(progress)
    plt.ylabel('Distancia')
    plt.xlabel('Generacion')
    plt.title('Mejor Fitness vs Generacion')
    plt.tight_layout()
    plt.show()
    return bestRoute

cityList = []
for i in range(0,4):
    cityList.append(i)
best_route=geneticAlgorithm(population=cityList, popSize=40, eliteSize=20, mutationRate=0.01, generations=1)
print('***RUTA OPTIMA***')
print(best_route)
