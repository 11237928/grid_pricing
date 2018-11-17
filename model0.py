#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Hakan Hekimgil, Jafar Chaab
"""

# this is the action/output
def retailprice(t):
    return None


# MODEL PARAMETERS
# number of service providers
nproviders = 1
# number of customers
ncustomers = 3
# number of time slots
ntimeslots = 24
# price bounds (p.225 last paragraph)
k1 = 1.5
k2 = 1.5
# weighting factor (p.225 last paragraph)
rho = 0.9
# wholesale price from grid operator
wholepricedata =[
        1.90, 1.80, 1.75, 1.70, 1.70, 2.00, 
        1.90, 1.80, 1.75, 1.70, 1.70, 2.00, 
        1.90, 1.80, 1.75, 1.70, 1.70, 2.00, 
        1.90, 1.80, 1.75, 1.70, 1.70, 2.00]
def wholeprice(t):
    return wholepricedata[t-1]

# CUSTOMER PARAMETERS
# customers' dissatisfaction related parameters (Table 2)
alpha = [0.8, 0.5, 0.3]
beta = [0.1, 0.1, 0.1]
# customer demand data (currentle from Fig. 5)
edemandcritdata =[
        [14.50, 14.25, 14.50, 14.25, 14.10, 13.70, 
         13.70, 14.10, 14.90, 16.00, 16.70, 17.20, 
         18.00, 18.10, 18.10, 18.20, 18.00, 17.50, 
         17.00, 17.10, 16.50, 16.00, 15.00, 14.50],
        [14.50, 14.25, 14.50, 14.25, 14.10, 13.70, 
         13.70, 14.10, 14.90, 16.00, 16.70, 17.20, 
         18.00, 18.10, 18.10, 18.20, 18.00, 17.50, 
         17.00, 17.10, 16.50, 16.00, 15.00, 14.50],
        [14.50, 14.25, 14.50, 14.25, 14.10, 13.70, 
         13.70, 14.10, 14.90, 16.00, 16.70, 17.20, 
         18.00, 18.10, 18.10, 18.20, 18.00, 17.50, 
         17.00, 17.10, 16.50, 16.00, 15.00, 14.50]]
edemandcurtdata =[
        [14.50, 14.25, 14.50, 14.25, 14.10, 13.70, 
         13.70, 14.10, 14.90, 16.00, 16.70, 17.20, 
         18.00, 18.10, 18.10, 18.20, 18.00, 17.50, 
         17.00, 17.10, 16.50, 16.00, 15.00, 14.50],
        [14.50, 14.25, 14.50, 14.25, 14.10, 13.70, 
         13.70, 14.10, 14.90, 16.00, 16.70, 17.20, 
         18.00, 18.10, 18.10, 18.20, 18.00, 17.50, 
         17.00, 17.10, 16.50, 16.00, 15.00, 14.50],
        [14.50, 14.25, 14.50, 14.25, 14.10, 13.70, 
         13.70, 14.10, 14.90, 16.00, 16.70, 17.20, 
         18.00, 18.10, 18.10, 18.20, 18.00, 17.50, 
         17.00, 17.10, 16.50, 16.00, 15.00, 14.50]]
def edemandcrit(t,n):
    return edemandcritdata[n-1,t-1]
def edemandcurt(t,n):
    return edemandcurtdata[n-1,t-1]
def edemand(t,n):
    return edemandcritdata[n-1,t-1] + edemandcurtdata[n-1,t-1]
# ranges of demand reduction
dmincoef = 0.1
dmaxcoef = 0.5
def dmin(t, n):
    return dmincoef * edemandcurt(t,n)
def dmax(t, n):
    return dmaxcoef * edemandcurt(t,n)
# price elasticities
elasticity_off_peak = -0.3
elasticity_mid_peak = -0.5
elasticity_on_peak = -0.7
def elasticity(t):
    assert t>=1 and t<=24
    if t <= 12:
        return elasticity_off_peak
    if t >= 17 and t <= 21:
        return elasticity_on_peak
    return elasticity_mid_peak
# energy consumption
def econscrit(t,n):
    return edemandcrit(t,n)
def econscurt(t,n):
    return edemandcurt(t,n) * (1 + elasticity(t) * ((retailprice(t,n) - wholeprice(t)) / wholeprice(t)))
def econs(t,n):
    return econscrit(t,n) + econscurt(t,n)
# dissatisfaction cost
def phi(t,n):
    return (beta[n-1] + (alpha[n-1] / 2) * (edemand(t,n) - econs(t,n)))


# OBJECTIVE FUNCTIONS
def cuobj(t,n):
    return retailprice(t,n) * econs(t,n) + phi(t,n)
def cuobjfn(n):
    # minimize
    return sum([cuobj(t+1,n) for t in range(ntimeslots)])
def cuobjf():
    # minimize
    return sum([cuobjfn(n+1) for n in range(ncustomers)])
def spobj(t,n):
    return (retailprice(t,n) - wholeprice(t)) * econs(t,n)
def spobjf():
    # maximize
    return sum([[spobj(t+1,n+1) for t in range(ntimeslots)] for n in range(ncustomers)])
def objf():
    # maximize
    return rho * spobjf() - (1 - rho) * cuobjf()


#Q-Learning

def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.
    
    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.
    
    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    
    """
import gym
import itertools
import matplotlib
import numpy as np
import pandas as pd
import sys


if "../" not in sys.path:
  sys.path.append("../") 

from collections import defaultdict
from lib.envs.cliff_walking import CliffWalkingEnv
from lib import plotting

matplotlib.style.use('ggplot')
env = CliffWalkingEnv()
def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.
    
    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.
    
    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    
    """
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn
def q_learning(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
    """
    Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
    while following an epsilon-greedy policy
    
    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.
    
    Returns:
        A tuple (Q, episode_lengths).
        Q is the optimal action-value function, a dictionary mapping state -> action values.
        stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """
    
    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))    
    
    # The policy we're following
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
    
    for i_episode in range(num_episodes):
        # Print out which episode we're on, useful for debugging.
        if (i_episode + 1) % 100 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()
        
        # Reset the environment and pick the first action
        state = env.reset()
        
        # One step in the environment
        # total_reward = 0.0
        for t in itertools.count():
            
            # Take a step
            action_probs = policy(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = env.step(action)

            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t
            
            # TD Update
            best_next_action = np.argmax(Q[next_state])    
            td_target = reward + discount_factor * Q[next_state][best_next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta
                
            if done:
                break
                
            state = next_state
    
    return Q, stats
