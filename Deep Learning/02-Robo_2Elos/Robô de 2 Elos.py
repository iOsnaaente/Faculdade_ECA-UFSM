# -*- coding: utf-8 -*-

# Módulo que simula o robô de dois elos
from robot2link import Robot
import numpy as np

class MyRobot(Robot):

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
  
  def pos(self):
    px = np.cos(self.theta1)*self.l1 + np.cos(self.theta1+self.theta2)*self.l2 
    py = np.sin(self.theta1)*self.l1 + np.sin(self.theta1+self.theta2)*self.l2

    return np.array([px, py]) 


  def cost(self):
    px = self.l1*np.cos(self.theta1) + self.l2*np.cos(self.theta1 + self.theta2)
    py = self.l1*np.sin(self.theta1) + self.l2*np.sin(self.theta1 + self.theta2)

    d2 = ((self.mx - px)**2 + (self.my - py)**2)

    return d2


  def grad(self):
    h = 0.0000001

    px = lambda t1, t2 : self.l1*np.cos(t1) + self.l2*np.cos(t1 + t2)
    py = lambda t1, t2 : self.l1*np.sin(t1) + self.l2*np.sin(t1 + t2)

    f = ((self.mx - px(self.theta1, self.theta2))**2 + (self.my - py(self.theta1, self.theta2))**2)

    dx = (((self.mx - px(self.theta1+h, self.theta2))**2 + (self.my - py(self.theta1+h, self.theta2))**2) - f)/h
    dy = (((self.mx - px(self.theta1, self.theta2+h))**2 + (self.my - py(self.theta1, self.theta2+h))**2) - f)/h

    return np.array([dx, dy])
    
  def optimize(self, step):
    
    gx, gy = self.grad()

    self.theta1 = self.theta1 - step*gx 
    self.theta2 = self.theta2 - step*gy 

    self.move(self.theta1, self.theta2)


# Exemplo de uso
r = MyRobot()
r.set_goal(110,-30)

t0s = np.linspace(0,np.pi,100)
t1s = np.linspace(0,np.pi/2,100)

for t0, t1 in zip(t0s, t1s):
  r.move(t0,t1)
  
r.animation()


r = MyRobot()
mx, my = np.random.rand(2)*300.0 - 150.0
r.set_goal(mx, my)

for _ in range(200):
  r.optimize(0.00001)
r.animation()