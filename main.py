import numpy as np
import pygame
import sys
import pickle




UP = 0
DOWN = 1
FORWARD = 2
BACKWARD = 3

TILE_SIZE = 40
HORIZONTAL_TILES = VERTICAL_TILES = 15

useSaved = True # change this value to use saved
loadedFile = "better_model.pickle" # used modul
savedFileName = loadedFile # "modul.pickle"


Q = {}
if useSaved:
	with open(loadedFile, 'rb') as handle:
	    Q = pickle.load(handle)
else:
	for x1 in range(0,HORIZONTAL_TILES):
		for y1 in range(0,VERTICAL_TILES):
			for x2 in range(0,VERTICAL_TILES):
				for y2 in range(0,VERTICAL_TILES):
					Q[(x1 - x2,y1 - y2)] = [float("{0:.2f}".format(np.random.uniform(-1,0))) for i in range(4)]



pygame.init()
clock = pygame.time.Clock()
screen = pygame.display.set_mode((HORIZONTAL_TILES * TILE_SIZE,VERTICAL_TILES * TILE_SIZE))
fps = 5 # switch fps by clicking "F"
show = True # switch show state by clicking "S"




reward = 0
lr = 0.1
gamma = 0.9
epsilon = 0.2
epsilon_decay_rate = 0.07
decay_epsilon_every = 30
games = 0


def draw():
	for x in range(HORIZONTAL_TILES):
		for y in range(VERTICAL_TILES):
			pygame.draw.rect(screen,(255,255,255),pygame.Rect(x * TILE_SIZE,y* TILE_SIZE,TILE_SIZE,TILE_SIZE),width=1)
	food.draw(screen,(0,255,0))
	player.draw(screen,(255,0,0))
def quit():
	with open(savedFileName,"wb") as handler:
		pickle.dump(Q, handler,protocol=pickle.HIGHEST_PROTOCOL)
	sys.exit()
class Blob:
	def __init__(self):
		self.depos()
	def __str__(self):
		print(f"(x: {self.x} , y : {self.y})")

	def __sub__(self,other):
		return (self.x - other.x , self.y - other.y)
	def __eq__(self,other):
		return self.x == other.y and self.y == other.y

	def depos(self):
		self.x = np.random.randint(0,HORIZONTAL_TILES)
		self.y = np.random.randint(0,VERTICAL_TILES)

	def draw(self,screen,color):
		pygame.draw.circle(screen,color,(self.x  * TILE_SIZE + TILE_SIZE / 2  , self.y * TILE_SIZE + TILE_SIZE  / 2),15)

	def action(self,action):
		if(action == UP and self.y > 0): 
			self.y -= 1
		elif(action == DOWN and self.y < VERTICAL_TILES- 1):
			self.y += 1

		if(action == BACKWARD and self.x > 0): 
			self.x -= 1
		elif(action == FORWARD and self.x < HORIZONTAL_TILES - 1):
			self.x += 1

player = Blob()
food = Blob()

while True:
	clock.tick(fps)
	screen.fill((0,0,0))
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			quit()

		if event.type == pygame.KEYDOWN:
			if event.key == pygame.K_f:
				if fps == 5:
					fps = 50000
				else:
					fps = 5
				print("FPS: ",fps)

			if event.key == pygame.K_s:
				show = not show
				print(f"show: " ,show)

			if event.key == pygame.K_ESCAPE:
				quit()
	
	obs = player - food

	if np.random.randint(0, 1) < epsilon:
		action = np.random.randint(0,4)
	else:
		action = np.argmax(Q[obs])

	player.action(action)

	new_obs = player - food
	max_future_q = np.max(Q[new_obs])
	curr_q = Q[obs][action]


	reward -= 20
	if player.x == food.x and player.y == food.y:
		reward += 300

		food.depos()

		games += 1
		
		if games % decay_epsilon_every == 0:
			epsilon -= epsilon_decay_rate

		print(f"games: {games}, epsilon: {epsilon}")
		
	new_q = (1 - lr) * curr_q + lr * (reward + gamma * max_future_q)
	Q[obs][action] = new_q

	reward = 0


	if  show:
		draw()
		pygame.display.update()



