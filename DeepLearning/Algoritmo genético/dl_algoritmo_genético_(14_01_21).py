# -*- coding: utf-8 -*-

from matplotlib.image import imsave, imread
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
import numpy as np

from Threads_.Threads_ import Trehdis
from time import time 

img_orig = Image.open('D:\Desktop\git\DeepLearning\Algoritmo genético/meandlaula.png', 'r')
img_orig.thumbnail((250, 250))

y_hat = np.asarray(img_orig, dtype=int)[ :, :, :3 ]


BACKGROUND_COLOR = tuple( [ int(x) for x in np.mean( np.mean(y_hat, axis=0), axis=0) ] )

W, H, C = y_hat.shape

img_new = Image.new('RGB', (H,W) )

drw = ImageDraw.Draw(img_new, "RGBA")
drw.polygon( [(20,10),(150,150), (100,20)], fill=(255, 0, 0, 100))

y = np.asarray( img_new, dtype=int)

err = np.sum( ( ( y - y_hat )**2 ) / (W*H*C) )

def create_individual( w, h, triangles = 100 ):
  chromosome = list()
  for _ in range(triangles):               # Criação dos triangulos
    for _ in range(3):                    # Cada vértice em cada triangulo
      x = int ( np.random.rand() * w )    # Width aleatório de inicio 
      y = int ( np.random.rand() * h )    # Heigth aleatório de inicio 
      chromosome += [x, y]
    r, g, b, a = [ int(np.random.rand()*255) for _ in range(4)  ]
    a = int((a/255)*64)
    chromosome += [r, g, b, a]
  return chromosome, None

create_individual(10,20,1)

def create_population(size, w, h):
  population = list()
  for _ in range(size):
    population.append( create_individual(w,h) )
  return population


def get_phenotype( chromosome, triangles = 100):
  chromosome = chromosome.copy()
  img = Image.new('RGB', (H,W), BACKGROUND_COLOR )
  drw = ImageDraw.Draw(img, 'RGBA')

  for _ in range( triangles ):
    triangle = list()
    for _ in range(3):
      x = chromosome.pop(0)
      y = chromosome.pop(0)
      triangle.append((y,x))
    r = chromosome.pop(0)
    g = chromosome.pop(0)
    b = chromosome.pop(0)
    a = chromosome.pop(0)
    drw.polygon(triangle, fill=(r,g,b,a) )
  return img


def fitness(y, y_hat):
  diff = (y - y_hat)**2 
  err = np.sum(diff) / (y.shape[0]*y.shape[1]*y.shape[2])
  return err


def crossover(chr1, chr2):
  idx = np.random.randint(0, len(chr1))
  new_chr1 = chr1[ :idx] + chr2[idx: ]
  new_chr2 = chr2[ :idx] + chr1[idx: ]
  
  return new_chr1, new_chr2


def crossover2point(chr1, chr2):
  idx1 = np.random.randint(0, len(chr1))
  idx2 = np.random.randint(0, len(chr1))
  if idx1 > idx2:
    idx1, idx2 = idx2, idx1 
  
  new_chr1 = chr1[:idx1] + chr2[idx1:idx2] + chr1[idx2:]
  new_chr2 = chr2[:idx1] + chr1[idx1:idx2] + chr2[idx2:]

  return new_chr1, new_chr2


def mutation(chr, n, w, h):
  l = len(chr)
  p = l*n 
  tot = int( np.random.normal(p, p/2) )
  if tot < 0:
    tot = 0 
  elif tot > l:
    tot = l 
  
  for _ in range(tot):
    idx = np.random.randint(0, l)
    if (idx % 10) < 6: 
      if idx % 2 :
        chr[idx] = int( np.random.rand() * w )   
      else: 
        chr[idx] = int( np.random.rand() * h )
    elif (idx % 10) < 9 :
        chr[idx] = int ( np.random.rand() * 255 ) 
    else : 
        chr[idx] = int ( np.random.rand() * 64 )
  
  return chr


def compute_fitness(population, y_hat, triangle = 100):

  scored = list()
  for chr, score in population:
    if score is None:
      img = get_phenotype(chr, triangle)
      y = np.asarray(img, dtype=int)
      score = fitness(y, y_hat)
    scored.append((chr, score))
  scored.sort(key=lambda x:x[1])
  return scored

def roulette(population):
  fitnesses = [1.0 / fitness for _, fitness in population]
  total = np.sum(fitnesses)
  choice = np.random.rand()*total
  subtotal = 0
  idx = 0
  for fitness in fitnesses:
    subtotal += fitness
    if subtotal >= choice:
      break
    idx += 1
  return idx


def new_generation(population, size, mutation_rate, w, h):
  new_population = list()
  while len(new_population) < size:
    chr1, _ = population[roulette(population)]
    chr2, _ = population[roulette(population)]

    new_chr1, new_chr2 = crossover2point(chr1, chr2)

    new_chr1 = mutation(new_chr1, mutation_rate, w, h)
    new_chr2 = mutation(new_chr2, mutation_rate, w, h)

    new_population.append((new_chr1, None))
    new_population.append((new_chr2, None))
  
  return new_population[:size]

# Precisamos da mesma quantidade de threads que pop_size
from threading import Thread, Lock

def compute_fitness_individual( individual, y_hat , triangle = 100):
  chr, score = individual
  if score is None:
    img = get_phenotype( chr, triangle )
    y = np.asarray(img, dtype=int)
    score = fitness(y, y_hat)
    new_individual = (chr, score)

  return new_individual


class Thredis ( Thread ):
  result = 0
  img = 0
  err = 0 
  def __init__(self, ind, hat):
    Thread.__init__(self)
    self.individual = ind
    self.hat   = hat 

  def get_fitness(self, y, y_hat):   
    diff = (y - y_hat)**2 
    self.err = np.sum(diff) / (y.shape[0]*y.shape[1]*y.shape[2])
    
  def get_pheno(self, triangles = 100 ):
    chromosome = self.individual[0].copy()
    self.img = Image.new('RGB', (H,W), BACKGROUND_COLOR )
    drw = ImageDraw.Draw(self.img, 'RGBA')

    for _ in range( triangles ):
      triangle = list()
      for _ in range(3):
        x = chromosome.pop(0)
        y = chromosome.pop(0)
        triangle.append((y,x))
      r = chromosome.pop(0)
      g = chromosome.pop(0)
      b = chromosome.pop(0)
      a = chromosome.pop(0)
      drw.polygon(triangle, fill=(r,g,b,a) )

  def compute_fitness( self, y_hat, triangle = 100 ):
    chr, score = self.individual
    if score is None:
      img = self.get_pheno( triangle )
      y = np.asarray(img, dtype=int)
      score = self.get_fitness(y, y_hat)
      new_individual = (chr, score)
    return new_individual

  def run (self):
    self.result = self.compute_fitness( self.individual, self.hat )
    if self.result == 0 :
      self.result = (0,0)


def genalg(w, h, y_hat, pop_size, elite_size, epochs, mutation_rate):
  count = 0 
  imgs = list()
  
  x, y = list(), list()
  
  threads = []

  population = create_population(pop_size, w, h)
  fig = plt.figure()

  for generation in range( epochs + 1 ):

    time1 = time()

    for ind in population:
      t = Thredis( ind, y_hat )
      t.start()
      threads.append( t )

    for i in threads:
      i.join()

    population = [] 
    for t in threads:
      if t.result == 0:
        population.append( ([i for i in range(pop_size*4)], 9999999) )
      else:
        population.append( t.result )

    population.sort( key = lambda x : x[1])
    threads = []

    if generation % 100 == 0:
      chr, fit = population[0]
      img_gen = get_phenotype(chr)
      imgs.append(img_gen)
      y.append(fit)
      x.append(generation)
      
      count += 1 
      
      print("Geração: ", generation, ", melhor: ", fit)

      plt.imshow(img_gen)
      plt.savefig('D:\Desktop\git\DeepLearning\Algoritmo genético/Generation%sFit%2.10f.png'%(generation, fit), dpi =600)

    elite = population[:elite_size]
    population = new_generation( population, pop_size - elite_size, mutation_rate, w, h) + elite 
    time2 = time()
    print( time2 - time1 )

  return imgs, x, y

images, x, y = genalg( W, H, y_hat, 100, 1, 100, 0.0025)

plt.plot(x, y)
#plt.imshow(images[-1])
plt.show()