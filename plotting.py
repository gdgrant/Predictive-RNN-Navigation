import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pickle
from parameters import par
from itertools import product

#data = pickle.load(open('./savedir/navigation_trajectories_v1.pkl', 'rb'))
data = pickle.load(open('./savedir/navigation_with_discount_plus_neurons_trajectories_v0.pkl', 'rb'))
data = data[-1]

print('Data from iteration {}.'.format(data['iter']))
reward_locs     = np.array(data['reward_locs'])
agent_locs      = np.array(data['agent_locs']).astype(np.int32)
actions         = np.array(data['actions'])

def animate():

    room = -2*np.ones([par['room_height'], par['room_width']])
    for i, r in enumerate(reward_locs[0]):
        room[r[0],r[1]] = par['rewards'][i]

    fig, ax = plt.subplots(1, figsize=[8,8])
    im = ax.imshow(room, animated=True)
    ax.set_title('')
    ax.set_xticks([])
    ax.set_yticks([])

    def updatefig(ind):
        t     = ind%(par['num_time_steps']-1)
        agent = ind//(par['num_time_steps']-1)

        room = -2*np.ones([par['room_height'], par['room_width']])
        for i, r in enumerate(reward_locs[agent]):
            room[r[0],r[1]] = par['rewards'][i]

        state = np.copy(room)
        state[agent_locs[t,agent,0],agent_locs[t,agent,1]] = -1
        ax.set_title('Trial {}, Time Step {}'.format(agent, t))
        im.set_array(state)
        return im,

    ani = FuncAnimation(fig, updatefig, frames=np.arange((par['num_time_steps']-1)*par['batch_size']), \
        interval=50, blit=False)
    #ani.save('iter9000_trial4.gif')
    plt.show()


def density():

    target = ['loc_density','action_density','greatest_action']
    target = target[2]

    act_dict = {
        0   :   'Right',
        1   :   'Left',
        2   :   'Down',
        3   :   'Up',
        4   :   'Pick'
    }

    room_pos = np.zeros([par['room_height'], par['room_width']])
    room_act = np.zeros([par['num_actions'],par['room_height'], par['room_width']])
    for t, i in product(range(par['num_time_steps']-1), range(par['batch_size'])):
        l = agent_locs[t,i]
        room_pos[l[0],l[1]] += 1

        room_act[np.argmax(actions[t,i]),l[0], l[1]] += 1


    room_act /= room_pos[np.newaxis,:,:]

    if target == 'action_density':
        fig, ax = plt.subplots(1,par['num_actions'],figsize=[14,8])
        for a in range(par['num_actions']):
            ax[a].imshow(room_act[a], aspect='auto', clim=[0,room_act.max()], cmap='plasma')
            ax[a].set_title('Action: {}'.format(act_dict[int(a)]))
        plt.show()

    elif target == 'loc_density':
        room_pos[1,1] = -1
        room_pos[3,2] = -1
        plt.imshow(room_pos/par['batch_size'])
        plt.colorbar()
        plt.title('Trajectory Density')
        plt.show()

    elif target == 'greatest_action':

        # Excludes "picking up" action
        room_act = np.argmax(room_act[:par['num_actions']-1,:,:], axis=0)

        fig, ax = plt.subplots()
        cax = ax.imshow(room_act, cmap='magma')
        cbar = fig.colorbar(cax, ticks=np.arange(par['num_actions']-1))
        cbar.ax.set_yticklabels([act_dict[i] for i in range(par['num_actions']-1)])
        plt.title('Most Likely Action')
        plt.show()


    # else:
        # plt.show(smiley_face)

density()
