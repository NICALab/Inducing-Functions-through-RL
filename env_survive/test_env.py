import time
import numpy as np
import matplotlib.pyplot as plt
from env_survive.env_survive import EnvSurvive

raw_vision = True
true_hidden = False
different_lr = True
path = './env_survive/mnist'
env = EnvSurvive(path='./env_survive/mnist', seed=0, raw_vision=True)

for i in range(1):
    obs = env.reset()
    obs_preprocess = obs

obs = env.reset()
env.render()

while True:
    action = env.action_space.sample()
    obs, rew, done, info = env.step(action)
    env.render(action=action)
    obs_preprocess = obs


    if done:
        print('\n\nreset')
        obs = env.reset()
