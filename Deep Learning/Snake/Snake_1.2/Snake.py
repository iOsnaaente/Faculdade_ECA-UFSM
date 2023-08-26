class Snake ():

    orientations = {
                    "left"   : [-1, 0],
                    "right"  : [ 1, 0],
                    "up"     : [ 0,-1],
                    "down"   : [ 0, 1],
                    "stopped": [ 0, 0] }

    orientation = 'stopped'
    speed  = 5.0 
    lenght = 2.0
    points = 0.0

    snakePos = []
    
    cor = [ [0,0,0],[0,0,0],[0,0,0] ]

    def __init__(self, speed = 1):
        self.speed = 1.0
        self.snakePos = [ [0,0],[1,0],[2,0] ]
        
        self.orientations = {
                    "left"   : [-1*self.speed, 0           ],
                    "right"  : [ 1*self.speed, 0           ],
                    "up"     : [ 0           ,-1*self.speed],
                    "down"   : [ 0           , 1*self.speed],
                    "stopped": [ 0           , 0           ]  }


    def move(self, orientation ) :
        if self.orientation == orientation:
            pass
        else: 
            for pos in range(len(self.snakePos)-1,0,-1):
                self.snakePos[pos] =  self.snakePos[pos-1] 

            self.orientation = orientation
            self.snakePos[0] = [self.snakePos[0][i]+self.orientations[self.orientation][i] for i in range(2) ]

    def die(self):
        print('Fim de jogo')
        self.orientation = 'stopped'


    def eat(self):
        self.snakePos.append([0,0])
        self.points = self.points + 1000


    def gain(self, points):
        self.points = self.points + points