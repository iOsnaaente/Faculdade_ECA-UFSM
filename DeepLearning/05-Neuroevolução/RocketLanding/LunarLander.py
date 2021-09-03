# -*- coding: utf-8 -*-

import numpy as np
import gym
from matplotlib import pyplot as plt

env = wrap_env(gym.make('LunarLander-v2'))

R = 0.0
obs = env.reset()
while True:
  action = np.random.randint(0,4)
  obs, reward, done, info = env.step(action)
  env.render()
  R += reward
  if done:
    break

env.close()
show_video()

"""Criação do código"""

class Net():
  def __init__(self, chromosome):
    # 8 => 10 => 4
    self.w1 = chromosome[:80].reshape((10,8))
    self.b1 = chromosome[80:90].reshape((10,))
    self.w2 = chromosome[90:130].reshape((4,10))
    self.b2 = chromosome[130:].reshape((4,))
  def forward(self, x):
    s1 = np.dot(self.w1, x) + self.b1
    z1 = s1 * (s1 > 0.0)
    s2 = np.dot(self.w2, z1) + self.b2
    z2 = s2 * (s2 > 0.0)
    return z2.argmax()

def create_individual():
  # np.random.Normal usa => ( média, distribuição e tamanho)  
  N = 8*10+10+10*4+4
  chromossome = np.random.normal( 0, 10, size=(N,) )
  return chromossome, None

def create_population(size):
  population = list()
  for _ in range(size):
    population.append( create_individual() )
  return population

def get_phenotype( chromossome ):
  n = Net(chromossome )
  return n

def fitness(n):
  R = 0.0
  trials = 6
  env = gym.make('LunarLander-v2')
  for _ in range(trials):
    obs = env.reset()
    while True:
      action = n.forward(obs)
      obs, reward, done, info = env.step(action)
      R += reward
      if done:
        break
  env.close()
  return (10000.0 + R/trials) / 10000.0

def crossover2point(chr1, chr2):
  N = len(chr1)
  idx1 = np.random.randint(0,N)
  idx2 = np.random.randint(0,N)
  if idx1 > idx2:
    idx1, idx2 = idx2, idx1
  new_chr1 = np.concatenate((chr1[0:idx1], chr2[idx1:idx2], chr1[idx2:]))
  new_chr2 = np.concatenate((chr2[0:idx1], chr1[idx1:idx2], chr2[idx2:]))
  return new_chr1, new_chr2

def mutation(chr, p):
  N = len(chr)
  total = int(np.random.normal(p*N, p*N))
  if total < 0:
    total = 0
  elif total > N:
    total = N
  for _ in range(total):
    idx = np.random.randint(0, N)
    if np.random.rand() > 0.5:
      chr[idx] += np.random.normal(0, 1)
    else:
      chr[idx] = np.random.normal(0, 10)
  return chr

def compute_fitness(population):
  scored = list()
  for chromosome, score in population:
    #if score is None:
    n = get_phenotype(chromosome)
    score = fitness(n)
    scored.append((chromosome, score))
  scored.sort(key=lambda x:x[1], reverse=True)
  return scored

def roulette(population):
  fitnesses = np.array([np.exp(fitness) for _, fitness in population] )
  total = np.sum(fitnesses)
  fitnesses /= total
  choice = np.random.rand()
  subtotal = 0
  idx = 0
  for f in fitnesses:
    subtotal += f
    if subtotal >= choice or idx == len(fitnesses)-1:
      break
    idx += 1
  return idx

def new_generation(population, size, mutation_rate):
  new_population = list()
  while len(new_population) < size:
    idx1 = roulette(population)
    idx2 = roulette(population)
    if idx1 == idx2:
      continue
    chr1, _ = population[idx1]
    chr2, _ = population[idx2]
    new_chr1, new_chr2 = crossover2point(chr1, chr2)
    new_chr1 = mutation(new_chr1, mutation_rate)
    new_chr2 = mutation(new_chr2, mutation_rate)
    new_population.append((new_chr1, None))
    new_population.append((new_chr2, None))
  return new_population[:size]

def genalg(pop_size, elite_size, epochs, mutation_rate):
  x = list()
  y = list()

  population = create_population(pop_size)

  for generation in range(epochs+1):

    population = compute_fitness(population)

    chr, fit = population[0]
    _, worst = population[-1]
    n = get_phenotype(chr)
    y.append(fit)
    x.append(generation)

    R = 0.0
    env2 = wrap_env(gym.make('LunarLander-v2'))
    obs = env2.reset()
    while True:
      action = n.forward(obs)
      obs, reward, done, info = env2.step(action)
      env2.render()
      R += reward
      if done:
        break
    env2.close()

    clear_output()
    print('Geração', generation, ', melhor:', fit, ', pior', worst)
    show_video()
      
    plt.plot(x, y)
    plt.show()

    elite = population[:elite_size]
    population = new_generation(population, pop_size - elite_size, mutation_rate) + elite
  return n, x, y

n, x, y = genalg(50, 3, 1000, 0.03)

with open("DadosPesosBias.txt", 'w') as f:
  for data in n.w1 : 
    f.write( str(data) )
    print(data)