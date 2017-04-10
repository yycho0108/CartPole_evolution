### ATTEMPT to apply genetic algorithm for inverse kinematics

import tf
import numpy as np

joint_lengths = [0.1, 0.1, 0.1] # 3-DOF robot arm

def fk(joint_angles):
    th = joint_angles
    p1 = np.array([0,0,joint_lengths[0]])
    p2 = p1 + joint_lengths[1] * np.array([np.cos(th[1])*np.cos(th[0]), np.cos(th[1])*np.sin(th[0]), np.sin(th[1])])
    p3 = p2 + joint_lengths[2] * np.array([np.cos(th[2])*np.cos(th[0]), np.cos(th[2])*np.sin(th[0]), np.sin(th[2])])
    return p3

desired_angles = np.random.uniform(low=0.0,high=2*np.pi,size=3)
desired_position = fk(desired_angles)

def fitness(param):
    global desired_position
    return -np.linalg.norm(fk(param) - np.array(desired_position))**2

def main():
    print fk([0,0,0])
    print fitness([0,0,0])

if __name__ == "__main__":
    main()
