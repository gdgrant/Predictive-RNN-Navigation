### Authors: Nicolas Y. Masse, Gregory D. Grant
import numpy as np
from parameters import par
import copy

# Actions that can be taken
#   Move up, down, left, right
#   Do nothing
#   Pick up reward

print('Using \'Room Stim\' stimulus file.')

class RoomStimulus:

    def __init__(self):

        self.initialize_rooms()
        self.place_agents()

        self.rewards = par['rewards']


    def initialize_rooms(self):

        # Two sets of reward locations:  Random and default
        rand_locs = np.random.choice(par['room_width']*par['room_height'], size=len(par['rewards']), replace=False)
        default_locs = [[1,1], [par['room_height']-2,par['room_width']-2], [1,par['room_width']-2], [par['room_height']-2],1]

        # Assign one stimulus location per reward
        self.stim_loc = []
        for i in range(len(par['rewards'])):

            if par['use_default_rew_locs']:
                if i >= len(default_locs):
                    raise Exception('Implement more default reward locations!')
                rew_loc = default_locs[i]
            else:
                rew_loc = [int(rand_locs[i]//par['room_width']), int(rand_locs[i]%par['room_width'])]

            self.stim_loc.append(rew_loc)

        # One locations are assigned, place rewards at those locations
        self.place_rewards()


    def place_rewards(self):

        # Set the reward locations to the allocated stimulus locations, in random order
        self.reward_locations = []
        self.reward_vectors = []
        for _ in range(par['batch_size']):
            trial_set = [self.stim_loc[ind] for ind in np.random.permutation(len(par['rewards']))]
            self.reward_locations.append(trial_set)

        print(np.array(self.reward_locations).shape)
        quit()


    def place_agents(self):

        xs = np.random.choice(par['room_width'],size=par['batch_size'])
        ys = np.random.choice(par['room_height'],size=par['batch_size'])
        self.agent_loc = [[int(ys[i]), int(xs[i])] for i in range(par['batch_size'])]

        self.loc_history = [self.agent_loc]


    def identify_reward(self, location, i):

        try:
            return self.reward_locations[i].index(location)
        except ValueError:
            return None


    def get_reward_vector(self, reward_index):

        if reward_index is not None:
            return self.reward_vectors[reward_index]
        else:
            return 0


    def make_inputs(self):

        # Inputs contain information for batch x (d1, d2, d3, d4, on_stim)
        inputs = np.zeros([par['batch_size'], par['n_input']])
        inputs[:,0] = [agent[0] for agent in self.agent_loc]
        inputs[:,1] = [agent[1] for agent in self.agent_loc]
        inputs[:,2] = [par['room_height'] - agent[0] for agent in self.agent_loc]
        inputs[:,3] = [par['room_width'] - agent[1] for agent in self.agent_loc]

        for i in range(par['batch_size']):

            trial_locs = self.reward_locations[i]
            self.agent_loc[i]

            # Reward index = index of the reward value
            # We want/need to match the reward index to the current location
            # Perhaps a dictionary
            #    Where the keys are tuples indicating locations
            #    And the items are the reward vectors + reward values
            reward_index = '  placeholder strings aren\'t indices  '
            inputs[i,par['num_nav_tuned']:par['num_nav_tuned']+par['num_rew_tuned'] = par['reward_vectors'][reward_index]



            #reward_vector = self.get_reward_vector(self.identify_reward(self.agent_loc[i], i))
            #inputs[i,par['num_nav_tuned']:par['num_nav_tuned']+par['num_rew_tuned']] += reward_vector

            if self.identify_reward(self.agent_loc[i],i) is not None:
                inputs[i,par['num_nav_tuned']:par['num_nav_tuned']+self.identify_reward(self.agent_loc[i], i)] += 1.



        return np.float32(inputs)


    def agent_action(self, action, mask):
        """ Takes in a vector of actions of size [batch_size, n_output] """

        action = np.argmax(action, axis=-1) # to [batch_size]
        reward = np.zeros(par['batch_size'])

        for i, a in enumerate(action):

            # If the network has found a reward for this trial, cease movement
            if mask[i] == 0.:
                continue

            if a == 0 and self.agent_loc[i][1] != par['room_width']-1:
                # Input 0 = Move Up (visually right)
                self.agent_loc[i][1] += 1
            elif a == 1 and self.agent_loc[i][1] != 0:
                # Input 1 = Move Down (visually left)
                self.agent_loc[i][1] -= 1
            elif a == 2 and self.agent_loc[i][0] != par['room_height']-1:
                # Input 2 = Move Right (visually down)
                self.agent_loc[i][0] += 1
            elif a == 3 and self.agent_loc[i][0] != 0:
                # Input 3 = Move Left (visually up)
                self.agent_loc[i][0] -= 1
            elif a == 4:
                # Input 5 = Pick Reward
                rewarded = self.identify_reward(self.agent_loc[i], i)
                if rewarded is not None:
                    reward[i] = self.rewards[rewarded]
                else:
                    reward[i] = -0.1

        self.loc_history.append(copy.deepcopy(self.agent_loc))

        return np.float32(reward)


    def get_agent_locs(self):
        return np.array(self.agent_loc).astype(np.float32)


if __name__ == '__main__':
    r = RoomStimulus()
    r.make_inputs()
