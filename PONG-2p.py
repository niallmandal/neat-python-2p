# Import modules
import gym
import retro
import time
import numpy as np

import neat
import pickle
import pandas as pd
from scipy.spatial import distance

#import custom code for 2p
from twoplayer import two_genes

import random

#create OpenAI Retro environment
env = retro.make(game='Pong-Atari2600', players=2)

#These are the x values for the orange and green paddles
orange_x = 68
green_x = 188

def eval_genomes(genomes1, genomes2, config):
    #Sometimes, each population can have a mismatched amount of individuals
    #This can be a result of extended stagnation (a genome will not improve over "n" generations, so it is removed)
    #To fix this, the population with a greater amount of individuals than the other will play against a random individual
    if len(genomes1) > len(genomes2):
        larger_population = genomes1
    else:
        larger_population = genomes2

    for idx,genome1 in enumerate(larger_population):
        try:
            genome1 = genomes1[idx]
        except:
            genome1 = random.choice(genomes1)

        try:
            genome2 = genomes2[idx]
        except:
            genome2 = random.choice(genomes2)

        #g1 represents the orange paddle as a general rule
        #g2 represents the green paddle as a general rule
        g1_id, g1_genome = genome1
        g2_id, g2_genome = genome2

        #an initial action is needed to be able to derive the observations from Pong
        env.reset()
        frame, rew, done, info = env.step([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1])

        #initializing each network from each population's respective genome
        net1 = neat.nn.recurrent.RecurrentNetwork.create(g1_genome, config)
        net2 = neat.nn.recurrent.RecurrentNetwork.create(g2_genome, config)

        #setting the current fitness to zero
        #fitness is based on score
        fc1 = 0
        fc2 = 0

        done = False
        while not done:
            orange_rew = rew[1]
            green_rew = rew[0]

            #if a point is lost, then the reward is set as -1 for that frame
            #reward is rounded to 0 if -1
            if orange_rew < 0:
                orange_rew = 0
            if green_rew < 0:
                green_rew = 0

            #Set all variables from Pong Data
            score_orange = info["score1"]
            score_green = info["score2"]
            ball_x = info["ball_x"]
            ball_y = info["ball_y"]
            orange_y = info["orange_y"]
            green_y = info["green_y"]

            #POLAR COORDINATE SYSTEM (r,θ)

            #Euclidean distance of ball to paddle
            orange_magnitude = distance.euclidean((orange_x,orange_y),(ball_x,ball_y))
            green_magnitude = distance.euclidean((green_x,green_y),(ball_x,ball_y))

            #Angle of ball to paddle in radians
            orange_theta = np.arctan2((ball_y-orange_y),(ball_x-orange_x))
            green_theta = np.arctan2((green_y-ball_y),(green_x-ball_x))

            #aggregate all of the variables into a 1-D array
            ob_orange = [ball_x,ball_y,orange_y,orange_magnitude,orange_theta]
            ob_green = [ball_x,ball_y,green_y,green_magnitude,green_theta]

            #activate each neural network to provide an input for the actual game
            nnOutput1 = net1.activate(ob_orange)
            nnOutput2 = net2.activate(ob_green)

            #set values for up or down per paddle
            p1_up = nnOutput1[0]
            p1_down = nnOutput1[1]
            p2_up = nnOutput2[0]
            p2_down = nnOutput2[1]

            #input[4]: green paddle up
            #input[5]: green paddle down
            #input[6]:orange paddle up
            #input[7]: orange paddle down
            input = [1,0,0,0,p2_up,p2_down,p1_up,p1_down,0,0,0,0,0,0,0,1]

            #pass in the data from each nn to the actual game of PONG
            frame, rew, is_done, info = env.step(input)
            env.render()

            #form total fitness of each genome (basically just the score of the game)
            fc1 += orange_rew
            fc2 += green_rew
            difference = abs(fc1 - fc2)

            total = score_orange + score_green

            if done or difference >= 21:
                print("Orange Genome: ", g1_id, ", Fitness Achieved: ", fc1)
                print("Green Genome: ", g1_id, ", Fitness Achieved: ", fc2)
                done = True

            #set fitness for each to be passed to neat-python
            g1_genome.fitness = fc1
            g2_genome.fitness = fc2
#import configuration file of NEAT parameters
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, 'config-feedforward.txt')

#instantiation of the first population
p = two_genes.pop(config)

#instantiation of the second population
p2 = two_genes.return_population(config).run()

#winner is when a maximum score is achieved (user must set this in config)
winner = p.run(eval_genomes,p2)
