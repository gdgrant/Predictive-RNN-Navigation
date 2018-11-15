### Authors: Nicolas Y. Masse, Gregory D. Grant

# Required packages
import tensorflow as tf
import numpy as np
import os, sys, time
import pickle

# Model modules
from parameters import *
import stimulus
import AdamOpt

# Match GPU IDs to nvidia-smi command
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# Ignore Tensorflow startup warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Make stimulus environment
stimulus_env = stimulus.RoomStimulus()

class Model:

    """ RNN model for supervised and reinforcement learning training """

    def __init__(self, starting_locations, reward_scores, reward_vectors):

        self.time_mask = tf.unstack(tf.ones([par['num_time_steps'],par['batch_size']]), axis=0)
        self.locations = tf.cast(starting_locations, tf.int32)
        self.location_history = [self.locations]
        self.reward_scores = reward_scores
        self.reward_vectors = reward_vectors

        self.define_stimulus()
        self.declare_variables()
        self.rnn_cell_loop()
        self.optimize()


    def define_stimulus(self):
        self.x_transform = tf.constant(par['x_transform'])
        self.y_transform = tf.constant(par['y_transform'])


    def simulate_step(self, actions, mask):

        actions = actions * mask

        delta_x = tf.reduce_sum(actions @ self.x_transform, axis=1)
        delta_y = tf.reduce_sum(actions @ self.y_transform, axis=1)

        locations_x = tf.cast(self.locations[:,0], tf.float32) + delta_x
        locations_y = tf.cast(self.locations[:,1], tf.float32) + delta_y

        locations_x = tf.clip_by_value(locations_x, 0., tf.constant(par['room_width']-1.))
        locations_y = tf.clip_by_value(locations_y, 0., tf.constant(par['room_height']-1.))

        self.locations = tf.cast(tf.stack([locations_x, locations_y], axis=1), tf.int32)
        self.location_history.append(self.locations)

        x_vector = tf.one_hot(self.locations[:,0], par['room_width'])
        y_vector = tf.one_hot(self.locations[:,1], par['room_height'])

        state = tf.reshape(x_vector, [par['batch_size'], par['room_width'], 1]) \
              * tf.reshape(y_vector, [par['batch_size'], 1, par['room_height']])

        reward = actions[:,-1] * tf.reduce_sum(state * self.reward_scores, axis=(1,2))

        return reward


    def make_stimulus(self):

        stim_x0 = self.locations[:,0]
        stim_x1 = (tf.constant(par['room_width'])-1) - tf.cast(self.locations[:,0], tf.int32)
        stim_y0 = self.locations[:,1]
        stim_y1 = (tf.constant(par['room_height'])-1) - tf.cast(self.locations[:,1], tf.int32)

        x_vector = tf.cast(tf.one_hot(self.locations[:,0], par['room_width']), tf.int32)
        y_vector = tf.cast(tf.one_hot(self.locations[:,1], par['room_height']), tf.int32)

        state = tf.reshape(x_vector, [par['batch_size'], par['room_width'], 1, 1]) \
              * tf.reshape(y_vector, [par['batch_size'], 1, par['room_height'], 1])

        reward_stim = tf.reduce_sum(tf.cast(state, tf.float32) * self.reward_vectors, axis=(1,2))

        stim = tf.stack([stim_x0, stim_x1, stim_y0, stim_y1], axis=1)
        stim = tf.concat([tf.cast(stim, tf.float32), reward_stim], axis=1)

        return stim


    def declare_variables(self):
        """ Initialize required variables """

        # All the possible prefixes based on network setup
        lstm_var_prefixes   = ['Wf', 'Wi', 'Wo', 'Wc', 'Uf', 'Ui', 'Uo', 'Uc', 'bf', 'bi', 'bo', 'bc']
        rl_var_prefixes     = ['W_pol_out', 'b_pol_out', 'W_val_out', 'b_val_out']

        # Add relevant prefixes to variable declaration
        prefix_list = lstm_var_prefixes + rl_var_prefixes

        # Use prefix list to declare required variables and place them in a dict
        self.var_dict = {}
        with tf.variable_scope('network'):
            for name in prefix_list:
                self.var_dict[name] = tf.get_variable(name, initializer=par[name+'_init'], trainable=True)

      
    def rnn_cell_loop(self):
        """ Initialize parameters and execute loop through
            time to generate the network outputs """
            
        # Specify training method outputs
        self.input_data = []
        self.pol_out    = []
        self.val_out    = []
        self.action     = []
        self.reward     = []

        # Records and states
        self.h          = []
        self.mask       = []
        h      = tf.zeros_like(par['h_init'])
        c      = tf.zeros_like(par['h_init'])
        mask   = tf.ones([par['batch_size'], 1])
        reward = tf.zeros([par['batch_size'], par['n_val']])

        for t in range(par['num_time_steps']):
            
            # Load input data for this time step
            inputs = tf.stop_gradient(self.make_stimulus())

            # Update the recurrent cell state
            h, c = self.predictive_cell(inputs, h, c)

            # Compute outputs for action
            pol_out         = h @ self.var_dict['W_pol_out'] + self.var_dict['b_pol_out']
            action_index    = tf.multinomial(pol_out, 1)
            action          = tf.one_hot(tf.squeeze(action_index), par['n_pol'])

            # Compute outputs for loss
            pol_out         = tf.nn.softmax(pol_out, 1)
            val_out         = h @ self.var_dict['W_val_out'] + self.var_dict['b_val_out']

            # Check for trial continuation
            continue_trial  = tf.cast(tf.equal(reward, 0.), tf.float32)
            mask           *= continue_trial

            # Calculate new state and reward
            feedback_reward = tf.expand_dims(tf.stop_gradient(self.simulate_step(action, mask)), axis=1)
            reward = feedback_reward*mask*tf.reshape(self.time_mask[t],[par['batch_size'],1])

            # Record RL inputs and outputs
            self.input_data.append(inputs)
            self.pol_out.append(pol_out)
            self.val_out.append(val_out)
            self.action.append(action)
            self.reward.append(reward)
            self.h.append(h)

            # Record mask
            self.mask.append(mask)


    def predictive_cell(self, x, h, c):

        # Compute LSTM state
        # f : forgetting gate, i : input gate,
        # c : cell state, o : output gate
        f   = tf.sigmoid(x @ self.var_dict['Wf'] + h @ self.var_dict['Uf'] + self.var_dict['bf'])
        i   = tf.sigmoid(x @ self.var_dict['Wi'] + h @ self.var_dict['Ui'] + self.var_dict['bi'])
        cn  = tf.tanh(x @ self.var_dict['Wc'] + h @ self.var_dict['Uc'] + self.var_dict['bc'])
        c   = f * c + i * cn
        o   = tf.sigmoid(x @ self.var_dict['Wo'] + h @ self.var_dict['Uo'] + self.var_dict['bo'])

        # Compute hidden state
        h = o * tf.tanh(c)

        return h, c


    def optimize(self):
        """ Calculate losses and apply corrections to model """

        # Set up optimizer and required constants
        epsilon = 1e-7
        adam_optimizer = AdamOpt.AdamOpt(tf.trainable_variables(), learning_rate=par['learning_rate'])

        self.time_mask = tf.reshape(tf.stack(self.time_mask), [par['num_time_steps'], par['batch_size'], 1])
        self.mask      = tf.stack(self.mask)
        self.reward    = tf.stack(self.reward)
        self.action    = tf.stack(self.action)
        self.pol_out   = tf.stack(self.pol_out)

        # Pad the value output of the network by one time step
        val_out = tf.stack(self.val_out)
        val_out = tf.concat([val_out, tf.zeros([1, par['batch_size'], par['n_val']])], axis=0)

        # Determine terminal state of the network
        terminal_state = tf.cast(tf.logical_not(tf.equal(self.reward, tf.constant(0.))), tf.float32)
        
        # Compute predicted value and advantage for finding policy loss
        pred_val = self.reward + par['discount_rate']*val_out[1:,:,:]*(1-terminal_state)
        advantage = pred_val - val_out[:-1,:,:]

        # Stop gradients back through action, advantage, predicted value, mask
        action_static    = tf.stop_gradient(self.action)
        advantage_static = tf.stop_gradient(advantage)
        pred_val_static  = tf.stop_gradient(pred_val)
        mask_static      = tf.stop_gradient(self.mask)

        # Multiply masks together
        full_mask        = mask_static*self.time_mask

        # Policy loss
        self.pol_loss = -tf.reduce_mean(full_mask*advantage_static*action_static*tf.log(epsilon+self.pol_out))

        # Value loss
        self.val_loss = 0.5*par['val_cost']*tf.reduce_mean(full_mask*tf.square(val_out[:-1,:,:]-pred_val_static))

        # Entropy loss
        self.entropy_loss = -par['entropy_cost']*tf.reduce_mean(tf.reduce_sum(full_mask*self.pol_out*tf.log(epsilon+self.pol_out), axis=2))

        # Collect losses
        self.total_loss = self.pol_loss + self.val_loss - self.entropy_loss

        # Compute and apply gradients
        self.train_op = adam_optimizer.compute_gradients(self.total_loss)


def main(gpu_id=None):
    """ Run RL training loop """

    # Isolate requested GPU
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    # Start records
    accuracy_record = []

    # Print parameters of note
    print_key_info()

    # Reset Tensorflow graph before running anything
    tf.reset_default_graph()

    # Start TensorFlow session
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8) if gpu_id == '0' else tf.GPUOptions()
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

        locs = tf.placeholder(tf.float32, shape=par['starting_locations'].shape, name='starting_locs')
        rews = tf.placeholder(tf.float32, shape=par['reward_scores'].shape, name='reward_scores')
        vecs = tf.placeholder(tf.float32, shape=par['reward_vectors'].shape, name='reward_vectors')

        # Select CPU or GPU
        device = '/cpu:0' if gpu_id is None else '/gpu:0'
        with tf.device(device):
            model = Model(locs, rews, vecs)

        # Initialize variables and start the timer
        sess.run(tf.global_variables_initializer())
        t0 = time.time()

        # Run training loop
        for i in range(par['n_train_batches']):

            # Refresh stimulus environment
            starting_locations, reward_scores, reward_vectors = refresh_reward_locations()
            #print(np.mean(np.argmax(par['reward_scores'])))
            feed_dict = {locs:starting_locations, rews:reward_scores, vecs:reward_vectors}

            # Calculate and apply gradients
            _, total_loss, hidden_state, reward, action, pol_loss, val_loss, entropy_loss, \
                location_history, inputs \
                = sess.run([model.train_op, model.total_loss, model.h, model.reward, \
                    model.action, model.pol_loss, model.val_loss, model.entropy_loss, \
                    model.location_history, model.input_data], feed_dict=feed_dict)



            # Process network response
            reward = np.stack(reward)
            rew = np.mean(np.sum(reward, axis=0))
            acc = np.mean(np.sum(reward>0, axis=0))

            # Record responses
            accuracy_record.append(acc)

            # Intermittently display network performance
            if i%200 == 0:

                stringA = 'Iter: {:>7} | Task: {} || Pol Loss: {:8.5f} | Val Loss: {:8.5f} | Ent Loss: {:8.5f} || '.format( \
                    i, par['task'], pol_loss, val_loss, -entropy_loss)
                stringB = 'Accuracy: {:5.3f} | Reward: {:5.3f} | Spiking: {:8.5f}'.format(\
                    acc, rew, np.mean(np.stack(hidden_state)))
                print(stringA + stringB)

    print('\nModel execution complete.')


def print_key_info():
    """ Display requested information """

    key_info = ['training_method', 'architecture', 'n_hidden', 'num_time_steps', \
        'learning_rate', 'batch_size', 'n_train_batches', 'discount_rate', 'task', \
        'room_width', 'room_height', 'rewards', 'spike_cost', 'entropy_cost', 'val_cost']

    print('-'*60)
    for k in key_info:
        print(k.ljust(30), par[k])
    print('-'*60)


if __name__ == '__main__':
    try:
        if len(sys.argv) > 1:
            main(sys.argv[1])
        else:
            main()
    except KeyboardInterrupt:
        quit('Quit by KeyboardInterrupt.')