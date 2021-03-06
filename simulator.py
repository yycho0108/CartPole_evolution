import random
import pygame
from pygame.locals import *

class Simulator(object):

    def __init__(self, width, height, rate=100):
        pygame.init()
        self.screen = pygame.display.set_mode((width,height))
        pygame.display.set_caption('CartPole Evolution')
        self.objects = {}
        self.rate = rate

        # up and down keys
        self.k_u = False
        self.k_d = False
        self.w = width
        self.h = height

    def run_once(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    self.k_u = True
                if event.key == pygame.K_DOWN:
                    self.k_d = True
                if event.key == pygame.K_ESCAPE:
                    return False
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_UP:
                    self.k_u = False
                if event.key == pygame.K_DOWN:
                    self.k_d = False 

        self.screen.fill((255,255,255))

        for o in self.objects.values():
            o.draw(self.screen, self.w, self.h)

        pygame.display.flip()
        pygame.time.wait(1000/self.rate)
        return True

    def add(self, name, obj):
        # passed by ref
        self.objects[name] = obj

    def remove(self, name):
        del self.objects[name]

    def get_keys(self):
        return self.k_u, self.k_d

    def run(self, cb):
        while self.run_once():
            if cb():
                break

def callback():
    print 'callback'

class Obj1(object):
    def __init__(self):
        self.position = (50,50)
        pass

    def draw(self, screen, w, h):
        pygame.draw.circle(screen, (255,0,0), (30,30), 10)

def main():
    simulator = Simulator(640,480)
    obj1 = Obj1()
    simulator.add('obj1',obj1)
    simulator.run(callback)
        
if __name__ == "__main__":
    main()
