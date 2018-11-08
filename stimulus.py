### Authors: Nicolas Y. Masse, Gregory D. Grant
import numpy as np
from parameters import par


# Actions that can be taken
#   Move up, down, left, right
#   Do nothing
#   Pick up reward

print('Using \'Room Stim\' stimulus file.')

class RoomStimulus:

    def __init__(self):

        # Shape configuration
        self.input_shape    = [par['num_time_steps'], par['batch_size'],par['n_input']]
        self.output_shape   = [par['num_time_steps'], par['batch_size'],par['n_output'] ]
        self.stimulus_shape = [par['num_time_steps'], par['batch_size'],par['num_motion_tuned'] ]
        self.response_shape = [par['num_time_steps'], par['batch_size'],par['num_motion_dirs'] ]
        self.fixation_shape = [par['num_time_steps'], par['batch_size'],par['num_fix_tuned'] ]
        self.rule_shape     = [par['num_time_steps'], par['batch_size'],par['num_rule_tuned'] ]
        self.mask_shape     = [par['num_time_steps'], par['batch_size']]

        self.initialize_rooms()
        self.place_agents()

        self.rewards = par['rewards']


    def initialize_rooms(self):

        self.rooms = np.zeros([par['batch_size'], par['room_height'], par['room_width']])
        locs       = np.random.choice(par['room_width']*par['room_height'], size=len(par['rewards']), replace=False)

        self.reward_locations = []
        for i in range(len(par['rewards'])):
            rew_loc = [int(locs[i]//par['room_width']), int(locs[i]%par['room_width'])]
            self.rooms[:,rew_loc[0],rew_loc[1]] = 1.
            self.reward_locations.append(rew_loc)


    def place_agents(self):

        xs = np.random.choice(par['room_width'],size=par['batch_size'])
        ys = np.random.choice(par['room_height'],size=par['batch_size'])
        self.agent_loc = [[int(ys[i]), int(xs[i])] for i in range(par['batch_size'])]


    def identify_reward(self, location):

        try:
            return self.reward_locations.index(location)
        except ValueError:
            return None


    def make_inputs(self):

        # Inputs contain information for batch x (d1, d2, d3, d4, on_stim)
        inputs = np.zeros([par['batch_size'], 5])
        inputs[:,0] = [agent[0] for agent in self.agent_loc]
        inputs[:,1] = [agent[1] for agent in self.agent_loc]
        inputs[:,2] = [par['room_height'] - agent[0] for agent in self.agent_loc]
        inputs[:,3] = [par['room_width'] - agent[1] for agent in self.agent_loc]

        for i in range(par['batch_size']):
            inputs[i,4] = 0 if self.identify_reward(self.agent_loc[i]) is None else 1

        return inputs


    def agent_action(self, action):
        """ Takes in a vector of actions of size [batch_size, n_output] """

        action = np.argmax(action, axis=-1) # to [batch_size]
        reward = np.zeros(par['batch_size'])

        for i, a in enumerate(action):

            if a == 0:
                # Input 0 = Do nothing
                pass
            elif a == 1 and self.agent_loc[i][1] != par['room_height']-1:
                # Input 1 = Move Up (visually right)
                self.agent_loc[i][1] += 1
            elif a == 2 and self.agent_loc[i][1] != 0:
                # Input 2 = Move Down (visually left)
                self.agent_loc[i][1] -= 1
            elif a == 3 and self.agent_loc[i][0] != par['room_width']-1:
                # Input 3 = Move Right (visually down)
                self.agent_loc[i][0] += 1
            elif a == 4 and self.agent_loc[i][0] != 0:
                # Input 4 = Move Left (visually up)
                self.agent_loc[i][0] -= 1
            elif a == 5:
                # Input 5 = Pick Reward
                rewarded = self.identify_reward(self.agent_loc[i])
                if rewarded is not None:
                    reward[i] = self.rewards[rewarded]

        return reward
