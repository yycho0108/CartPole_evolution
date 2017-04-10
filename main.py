#!/usr/bin/python

from pid import PID
from net import Net
from simulator import Simulator

from abc import ABCMeta, abstractmethod
import numpy as np
import math
from matplotlib import pyplot as plt
from multiprocessing import Pool

import pygame

## PARAMETERS
population_size = 1000 # size of gene pool
selection_size = 400 # how many among the selection get preserved
num_repeat = 1 # number of repeated runs for each test, currently not used
num_epoch = 200 # how many generations to run for
sim_duration = 1000 # how long each generation will be tested for
mutate_mag = 1e-2 # how much the mutation will modify the chromosomes
tau = 0.02 # time per step

### CONFIG FOR PID
#param_size = 8

### CONFIG FOR NETWORK
network_topology = [4,8,1]
param_size = np.dot(network_topology[:-1],network_topology[1:])

class Environment(object):
    __metaclass__ = ABCMeta
    def __init__(self):
        self.state = x
    @abstractmethod
    def step(self, ctrl):
        pass
    @abstractmethod
    def reset(self):
        pass

class CartPoleEnv(Environment):
    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5 # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0

        # Angle at which to fail the episode
        self.theta_threshold_radians = np.deg2rad(16)
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max])

        self.reset()

    def step(self, ctrl, tau=tau):
        force = self.force_mag * ctrl
        state = self.state
        x, x_dot, theta, theta_dot = state
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta* temp) / (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc  = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        x  = x + tau * x_dot
        x_dot = x_dot + tau * xacc
        theta = (theta + tau * theta_dot) % (2*np.pi)
        theta_dot = theta_dot + tau * thetaacc
        self.state = (x,x_dot,theta,theta_dot)

        up = (-self.theta_threshold_radians < theta and \
                theta < self.theta_threshold_radians)
        fail = (x < -self.x_threshold or x > self.x_threshold)

        #done =  x < -self.x_threshold \
        #        or x > self.x_threshold \
        #        or theta < -self.theta_threshold_radians \
        #        or theta > self.theta_threshold_radians
        #done = theta < -self.theta_threshold_radians \
        #        or theta > self.theta_threshold_radians
        #done = bool(done)

        return np.array(self.state), fail, up

    def reset(self):
        #self.state = np.array([0,0,np.pi/2,0]) # swing-up
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        return np.array(self.state)

def mutate(params, mag=mutate_mag):
    params += np.random.normal(size=params.shape, scale=mag)

def crossover(params,output_size=None):
    n = len(params)
    if output_size is None:
        output_size = n
    return params[np.random.randint(n,size=(output_size,param_size)),range(param_size)]

class Controller(object):
    def __init__(self, p, t='pid'):
        self.t = t
        if t is 'pid':
            t_w, t_k_p, t_k_i, t_k_d, x_w, x_k_p, x_k_i, x_k_d = p
            self.t_w = t_w
            self.x_w = x_w
            self.t_pid = PID(t_k_p, t_k_i, t_k_d)
            self.x_pid = PID(x_k_p, x_k_i, x_k_d)
        elif t is 'net':
            w = []
            end = 0
            for a,b in zip (network_topology[:-1], network_topology[1:]):
                w.append(np.reshape(p[end:end+a*b],(a,b)))
                end = end + a*b
            self.c_net = Net(w)
    def compute(self, s):
        t = self.t

        if t is 'pid':
            t_w, x_w = self.t_w, self.x_w
            x,_,t,_ = s
            return t_w * self.t_pid.compute(t,tau) + \
                    x_w * self.x_pid.compute(x,tau)
        elif t is 'net':
            return self.c_net.compute(s)

def fitness(param, duration = sim_duration, repeat = num_repeat):
    env = CartPoleEnv()
    controller = Controller(param,'net')
    #controller = Controller(param,'pid')

    def run_once():
        #x,_,t,_ = env.reset()
        s = env.reset()
        score = 0.0

        for i in range(duration):
            ctrl = controller.compute(s)
            s, fail, up = env.step(ctrl)
            if fail:
                break
            if up:
                score += 1

        score /= duration
        return score

    score = sum([run_once() for _ in range(repeat)])/repeat

    return score

def train(num_epoch):
    params = np.random.normal(size=(population_size,param_size))

    best_scores = []
    best_params = []

    pool = Pool(processes=4)

    for i in range(num_epoch): # evolution over 100 epochs
        scores = np.array(pool.map(fitness, params))
        best_indices = np.argsort(scores)[-selection_size:]
        parents = params[best_indices]
        params = crossover(parents,output_size=(population_size - selection_size))
        params = np.vstack((parents,params))

        std = np.std(params, axis=0) # measure deviation before mutation

        mutate(params)

        best_idx = best_indices[0]
        best_score = scores[best_idx]
        best_param = params[best_idx]

        print '[{}] : {}; std : {}'.format(i, best_score, np.mean(std))

        best_scores.append(best_score)
        best_params.append(best_param)

    best_idx = np.argmax(best_scores)
    best_score = best_scores[best_idx]
    best_param = best_params[best_idx]

    #plt.plot(best_scores)
    #plt.show()

    with open('data.txt',"a+") as f:
        param_txt = ', '.join(['%.3f' % e for e in best_param])
        f.write('[{}] : {}\n'.format(best_score, param_txt))

    np.savetxt('params.txt', params)

    return best_param

class CartPoleGraphics(object):
    def __init__(self):
        self.x = self.t = 0
        self.pole_img = None

    def draw(self, screen, w, h):
        if self.pole_img is None:
            self.pole_img = pygame.Surface((w/5, w/5))

        x,t = self.x, self.t + np.pi/2

        cx = w/2
        cy = h/2

        c_w = (w * 80 / 640) # cart width
        c_h = (h * 30 / 480) # cart height

        b_d = (w * 10 / 640) # ball diameter 
        r_l = (w * 120 / 640) # rod length

        ## Draw Rail
        pygame.draw.line(screen,(150,150,150),(0,cy),(w,cy),2)

        ## Draw Cart
        rect = pygame.Rect(0,0,c_w,c_h)
        c_x = cx + x*w/4.8
        c_y = cy
        rect.center = (c_x, c_y)
        pygame.draw.rect(screen,(128,228,255),rect,0)

        ## Draw Pole
        b_x = c_x + r_l * np.cos(t)
        b_y = c_y - r_l * np.sin(t)
        pygame.draw.line(screen,(30,30,30),(c_x,c_y),(b_x,b_y),5)

        ## Draw Ball
        cir = pygame.Rect(0,0,2*b_d,2*b_d)
        cir.center = (b_x, b_y)
        pygame.draw.ellipse(screen,(64,196,255),cir,0)

        #x = self.position[0]
        #y = self.position[1]
        #w = 2 * self.r
        #h = self.h 

        #rect = (x-w/2,y-h/2,w,h)
        #pygame.draw.rect(screen,(128,228,255),rect,0)
        #pygame.draw.rect(screen,(64,196,255),rect,5)

        #r_img = pygame.transform.rotate(self.img, r2d(t))

        #rect = r_img.get_rect()
        #rect.center = (x,y)

        #if len(self.trajectory) > 1:
        #    pygame.draw.lines(screen, self.t_col, False,self.trajectory)
        #screen.blit(r_img, rect)

    def update(self, x, t):
        self.x, self.t = x,t

def test(param):
    sim = Simulator(640,480, rate=int(1/tau))

    env = CartPoleEnv()
    controller = Controller(param,'net')
    #controller = Controller(param,'pid')

    s = env.reset()

    g = CartPoleGraphics()

    sim.add('cartpole', g)

    data = {'s': s}

    def step():
        s = data['s']
        ctrl = controller.compute(s)
        s, fail, up = env.step(ctrl)
        data['s'] = s
        x,_,t,_ = s
        g.update(x,t)
        return fail

    sim.run(step)



if __name__ == "__main__":
    #param = np.array([0.015, 0.884, -0.425, -1.317, -0.941, 0.623, 0.882, 0.755, -0.313, 0.748, 0.469, -0.008, -0.486, -0.488, 0.310, -0.863, 0.351, -0.452, 1.111, -0.216, 0.213, -0.579, 2.225, 2.558, 0.646, 0.333, -0.412, 2.776, -0.207, 0.748, -0.027, -0.152, -0.115, 0.609, -0.040, 2.503, -0.466, -0.032, 1.819, 1.097])
    param = train(num_epoch)
    test(param)
