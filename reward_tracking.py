import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pickle
from parameters import par
from itertools import product

data = pickle.load(open('./savedir/vanilla_no_penalty_trajectories_v0.pkl', 'rb'))

for d in data:
    iter        = d['iter']
    reward_locs = d['reward_locs']
    agent_locs  = np.array(d['agent_locs']).astype(np.int32)
    actions     = np.argmax(d['actions'], axis=-1)

    reward_values = []
    reward_locations = []
    for i in range(par['batch_size']):
        reward_locations.append(list(reward_locs[i].keys()))
        reward_values.append([x['rew'] for x in reward_locs[i].values()])

    tally = np.zeros([par['batch_size'],len(par['rewards'])])
    for i in range(par['batch_size']):
        reward0 = False
        reward1 = False
        sequential = False
        for t in range(agent_locs.shape[0]):
            if tuple(agent_locs[t,i,:]) in reward_locations[i]:
                if tuple(agent_locs[t,i,:]) == tuple(agent_locs[-1,i,:])
