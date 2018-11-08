import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pickle
from parameters import par


data = pickle.load(open('./savedir/trajectory_iter1800.pkl', 'rb'))
reward_locs = data['reward_locs']
agent_locs = np.array(data['agent_locs']).astype(np.int32)
actions = data['actions']

room = -2*np.ones([par['room_height'], par['room_width']])
for i, r in enumerate(reward_locs):
    room[r[0],r[1]] = par['rewards'][i]

agent = 4

fig, ax = plt.subplots(1, figsize=[8,8])
im = ax.imshow(room, animated=True)
ax.set_title('Agent {} in Room'.format(agent))
ax.set_xticks([])
ax.set_yticks([])

def updatefig(t):
    state = np.copy(room)
    state[agent_locs[t,agent,0],agent_locs[t,agent,1]] = -1
    ax.set_title('Time Step {}'.format(t))
    im.set_array(state)
    return im,

ani = FuncAnimation(fig, updatefig, frames=np.arange(par['num_time_steps']-1), \
    interval=100, blit=False)
ani.save('iter9000_trial4.gif')
plt.show()
