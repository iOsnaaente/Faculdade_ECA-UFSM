from Snake import Snake
import pygame
import random
import os

current_path = os.path.dirname(__file__) # Where your .py file is located
image_path = os.path.join(current_path, 'img') # The resource folder path

img = lambda path : os.path.join(image_path, path)

cakePos   = [0,0]

snakesPos = []
wallPos   = []

#IMAGENS DO JOGO
wall      = pygame.image.load( img("wall.png") ) 
headSnake = pygame.image.load( img("face.png") )
bodySnake = pygame.image.load( img("body.png") )
cake      = pygame.image.load( img("cake.png") )

# Offset 
offset = 16 

# Velocidade 
velocidade = 1

cobra1 = Snake(velocidade)

FIM_DE_JOGO = False

# plota as paredas da arena
wallPos = []

fileWalls = open(current_path + "/stage1.txt", 'r')
walls = fileWalls.readlines()

for y, wall in enumerate( walls ):
	wall = wall[:-1].split(',')
	for x, i in enumerate( wall ):
		if i:
			wallPos.append([x, y])

pygame.init()

dimensoes = pygame.display.Info()

dimensoes_tela = [dimensoes.current_w, dimensoes.current_h]

screen = pygame.display.set_mode(dimensoes_tela)

pygame.display.set_caption("My sad Snake")
#pygame.display.set_icon()

clk = pygame.time.Clock()


# gera os bolos
def cake_Generator():
	cakePos = [ ( ( random.randint( 16 ,dimensoes_tela[0] - 6 ) ) // 16) * 16,
				( ( random.randint( 16 ,dimensoes_tela[1] - 16 ) ) // 16) * 16 ] 
	if cakePos not in wallPos:
		if cakePos not in snakesPos:
			return cakePos
		else:
			cake_Generator()	
	else:
		cake_Generator() 


def plot_Walls():
	for i in wallPos:
		i = [ n*offset for n in range(len(i))]
		screen.blit(wall, i)

def plot_Cake():
	screen.blit(cake, cakePos)


# plota as cobras no cenário com o offset definido
def plot_Snake(snakes):
	snakesPos = []
	for snake in snakes: 
		for i in snake.snakePos:
			snakesPos.append(i)
			if i is not snake.snakePos[0]:
				screen.blit(bodySnake, [pos*offset for pos in i])
			else: 
				screen.blit(headSnake, [pos*offset for pos in snake.snakePos[0] ] )

cakePos = cake_Generator()

snakeDir = 'stopped'

while True:
	for event in pygame.event.get():
		
		if event.type == pygame.QUIT:
			pygame.quit()
		
		if event.type == pygame.KEYDOWN:
			if event.key == pygame.QUIT:	
				pygame.quit()

			if event.key == pygame.K_UP:
				if snakeDir is not 'down':
					snakeDir = "up" 

			if event.key == pygame.K_DOWN:
				if snakeDir is not 'up':
					snakeDir = "down" 

			if event.key == pygame.K_LEFT:
				if snakeDir is not 'right':
					snakeDir = "left" 

			if event.key == pygame.K_RIGHT:
				if snakeDir is not 'left':
					snakeDir = "right" 

	cobra1.move(snakeDir)	

	screen.fill([100,200,100])
	try:
		plot_Walls()
		plot_Cake()

		plot_Snake( list(cobra1) )

		if cobra1.snakePos[0] in wallPos:
			snakeDir = 'stopped'
			FIM_DE_JOGO = True

		if cobra1.snakePos[0] == cakePos:
			cobra1.eat()
			cakePos = cake_Generator()

		if cobra1.snakePos[0] in snakesPos:
			snakeDir = 'stopped'
			FIM_DE_JOGO = True

	except:
		print(cakePos)
		cakePos = cake_Generator()


	pygame.display.update()

	clk.tick(60)