#################################################################
#						IMPORTAÇÕES								#
#################################################################
import pygame
import random


#################################################################
#						VARIAVEIS								#
#################################################################

dimensoes_tela = [16*45, 16*30]

cakePos = [0,0]
wallPos = []

snake_Dir = {
	
	"Left"   : [-16,0],
	"Right"  : [ 16,0],
	"Up"     : [0,-16],
	"Down"   : [0, 16],
	"Stopped": [0, 0 ]
}

#IMAGENS DO JOGO

wall      = pygame.image.load("./wall.png")
headSnake = pygame.image.load("./face.png")
bodySnake = pygame.image.load("./body.png")
cake      = pygame.image.load("./cake.png")


velocidade = 5

snakePos = [[160,160],[172,160],[188, 160]]
snakeDir = snake_Dir["Stopped"]
FIM_DE_JOGO = False


#################################################################
#						FUNÇÕES									#
#################################################################

def somaCoordenadas(mat1=[0,0], mat2=[0,0]):
	return [mat1[0]+mat2[0], mat1[1]+mat2[1]] 


def move_Snake(snakePos=[], snakeDir=[0,0]):
	if snakeDir[0] == snakeDir[1]:
		pass
	else:
		for i in range(len(snakePos)-1,0,-1):
			snakePos[i][0] = snakePos[i-1][0]
			snakePos[i][1] = snakePos[i-1][1]
	snakePos[0] = somaCoordenadas(snakePos[0], snakeDir)


def cake_Generator():
	cakePos = [
	((random.randint(16,dimensoes_tela[0]-6))//16)*16,
	((random.randint(16,dimensoes_tela[1]-16))//16)*16
	] 
	
	if cakePos not in wallPos:
		if cakePos not in snakePos:
			return cakePos
		else:
			cake_Generator()	
	else:
		cake_Generator() 


def plot_Walls():
	for i in wallPos:
		screen.blit(wall, i)


def plot_Cake():
	screen.blit(cake, cakePos)


def plot_Snake():
	for i in snakePos:
		if i is not snakePos[0]:
			screen.blit(bodySnake, i)
		else: 
			screen.blit(headSnake, snakePos[0])




#################################################################
#						CÓDIGO	AVULSO							#
#################################################################

#DELIMITANDO AS BORDAS
for x in range(0,dimensoes_tela[0],16):
	wallPos.append([x,0])
	wallPos.append([x,dimensoes_tela[1]-16])

for y in range(0,dimensoes_tela[1],16):
	wallPos.append([0,y])
	wallPos.append([dimensoes_tela[0]-16,y])


cakePos = cake_Generator()


#################################################################
#						INICIANDO PYGAME						#
#################################################################

pygame.init()

screen = pygame.display.set_mode(dimensoes_tela)

pygame.display.set_caption("My_Poor_Snake")

clock = pygame.time.Clock()


while True:
	for event in pygame.event.get():
		
		if event.type == pygame.QUIT:
			pygame.quit()
		
		if event.type == pygame.KEYDOWN:
			if event.key == 27:	#ESC
				pygame.quit()	#To quit

			if event.key == 273:
				if snakeDir is not snake_Dir["Down"]:
					snakeDir = snake_Dir["Up"] 
			if event.key == 274:
				if snakeDir is not snake_Dir["Up"]:
					snakeDir = snake_Dir["Down"]
			if event.key == 275:
				if snakeDir is not snake_Dir["Left"]:
					snakeDir = snake_Dir["Right"]
			if event.key == 276:
				if snakeDir is not snake_Dir["Right"]:
					snakeDir = snake_Dir["Left"]
			


	screen.fill([100,200,100])
	try:
		plot_Walls()
		plot_Cake()
		plot_Snake()

		move_Snake(snakePos, snakeDir)

		if snakePos[0] in wallPos:
			snakeDir = snake_Dir["Stopped"]
			FIM_DE_JOGO = True

		if snakePos[0] == cakePos:
			snakePos.append([0,0])
			cakePos = cake_Generator()

		if snakePos[0] in snakePos[2:]:
			snakeDir = snake_Dir["Stopped"]
			FIM_DE_JOGO = True
	except:
		print(cakePos)
		cakePos = cake_Generator()


	pygame.display.update()

	clock.tick(velocidade*5)