### Authors: Nicolas Y. Masse, Gregory D. Grant

import numpy as np
from itertools import product, chain
from testing import action_transform

print("\n--> Loading parameters...")

##############################
### Independent parameters ###
##############################

global par
par = {
    # Setup parameters
    'save_dir'              : './savedir/',
    'plot_dir'              : './plotdir/',
    'save_fn'               : 'navigation',
    'save_fn_suffix'        : '_v0',
    'save_plots'            : False,

    # Network configuration
    'stabilization'         : 'pathint',    # 'EWC' (Kirkpatrick method) or 'pathint' (Zenke method)
    'synapse_config'        : 'std_stf',     # Full is 'std_stf'
    'exc_inh_prop'          : 0.8,          # Literature 0.8, for EI off 1
    'var_delay'             : False,
    'training_method'       : 'RL',         # 'SL', 'RL'
    'architecture'          : 'LSTM',       # 'BIO', 'LSTM'

    # Network shape
    'include_rule_signal'   : False,
    'n_hidden'              : 200,

    # Timings and rates
    'dt'                    : 20,
    'learning_rate'         : 1e-3,
    'membrane_time_constant': 100,
    'connection_prob'       : 1.0,
    'discount_rate'         : 0.95,

    # Variance values
    'clip_max_grad_val'     : 1.0,
    'input_mean'            : 0.0,
    'noise_in_sd'           : 0.0,
    'noise_rnn_sd'          : 0.05,

    # Task specs
    'task'                  : 'basic',      # Currently doesn't do anything; placeholder
    'num_nav_tuned'         : 4,
    'num_rew_tuned'         : 10,
    'num_fix_tuned'         : 0,
    'num_rule_tuned'        : 0,
    'n_val'                 : 1,
    'num_actions'           : 5,
    'room_width'            : 6,
    'room_height'           : 5,
    'rewards'               : [1.,2.],
    'use_default_rew_locs'  : True,
    'failure_penalty'       : 0.,
    'trial_length'          : 300,

    # Cost values
    'spike_cost'            : 0.,
    'weight_cost'           : 0.,
    'entropy_cost'          : 0.01,
    'val_cost'              : 0.01,
    'error_cost'            : 0.,

    # Synaptic plasticity specs
    'tau_fast'              : 200,
    'tau_slow'              : 1500,
    'U_stf'                 : 0.15,
    'U_std'                 : 0.45,

    # Training specs
    'batch_size'            : 128,
    'n_train_batches'       : 500000, #50000,

    # Omega parameters
    'omega_c'               : 0.,
    'omega_xi'              : 0.001,
    'EWC_fisher_num_batches': 16,   # number of batches when calculating EWC

    # Gating parameters
    'gating_type'           : None, # 'XdG', 'partial', 'split', None
    'gate_pct'              : 0.8,  # Num. gated hidden units for 'XdG' only
    'n_subnetworks'         : 4,    # Num. subnetworks for 'split' only

}


############################
### Dependent parameters ###
############################


def update_parameters(updates):
    """
    Takes a list of strings and values for updating parameters in the parameter dictionary
    Example: updates = [(key, val), (key, val)]
    """
    for (key, val) in updates.items():
        par[key] = val
        print('Updating : ', key, ' -> ', val)
    update_dependencies()


def update_dependencies():
    """ Updates all parameter dependencies """

    ###
    ### Putting together network structure
    ###
    c = 0.05

    print('Using LSTM networks; setting to EI to False')
    par['EI'] = False
    par['exc_inh_prop'] = 1.
    par['synapse_config'] = None
    par['spike_cost'] = 0.

    # Number of input neurons
    par['n_input'] = par['num_nav_tuned'] + par['num_rew_tuned'] + par['num_fix_tuned'] + par['num_rule_tuned']

    # Number of output neurons
    par['n_output'] = par['num_actions']
    par['n_pol'] = par['num_actions']

    # Specify time step in seconds and neuron time constant
    par['dt_sec'] = par['dt']/1000
    par['alpha_neuron'] = np.float32(par['dt'])/par['membrane_time_constant']

    # Generate noise deviations
    par['noise_rnn'] = np.sqrt(2*par['alpha_neuron'])*par['noise_rnn_sd']
    par['noise_in'] = np.sqrt(2/par['alpha_neuron'])*par['noise_in_sd']

    # Set trial step length
    par['num_time_steps'] = par['trial_length']//par['dt']

    # State initialization
    par['state_init'] = np.float32(np.zeros([par['batch_size'],4]))
    for i in range(par['batch_size']):
        par['state_init'][i,np.random.choice(4)] = 1.

    # Action transform reference
    par['x_transform'] = np.zeros([par['num_actions'],par['num_actions']]).astype(np.float32)
    par['x_transform'][0,0] = 1
    par['x_transform'][2,2] = -1

    par['y_transform'] = np.zeros([par['num_actions'],par['num_actions']]).astype(np.float32)
    par['y_transform'][1,1] = 1
    par['y_transform'][3,3] = -1

    # Specify one-hot vectors matching with each reward
    condition = True
    while condition:
        par['reward_vector_refs'] = np.random.choice([0,1], size=[len(par['rewards']), par['num_rew_tuned']])
        condition = (np.mean(np.std(par['reward_vector_refs'], axis=0)) == 0.) and len(par['rewards']) != 1

    refresh_reward_locations()

    ###
    ### Setting up weights, biases, masks, etc.
    ###

    # Specify initial RNN state
    par['h_init'] = 0.1*np.ones((par['batch_size'], par['n_hidden']), dtype=np.float32)

    # Initialize RL-specific weights
    par['W_pol_out_init'] = np.float32(np.random.uniform(-c, c, size = [par['n_hidden'], par['n_pol']]))
    par['b_pol_out_init'] = np.zeros((1,par['n_pol']), dtype = np.float32)

    par['W_val_out_init'] = np.float32(np.random.uniform(-c, c, size = [par['n_hidden'], par['n_val']]))
    par['b_val_out_init'] = np.zeros((1,par['n_val']), dtype = np.float32)

    ### Setting up LSTM weights and biases
    LSTM_var_names = ['Wf', 'Wi', 'Wo', 'Wc', 'Uf', 'Ui', 'Uo', 'Uc', 'bf', 'bi', 'bo', 'bc']
    for name in LSTM_var_names:
        if name.startswith('W'):
            par[name + '_init'] = np.float32(np.random.uniform(-c, c, size = [par['n_input'], par['n_hidden']]))
        elif name.startswith('U'):
            par[name + '_init'] = np.float32(np.random.uniform(-c, c, size = [par['n_hidden'], par['n_hidden']]))
        elif name.startswith('b'):
            par[name + '_init'] = np.float32(np.random.uniform(-c, c, size = [1, par['n_hidden']]))


def refresh_reward_locations():

    par['reward_scores'] = np.float32(np.zeros([par['batch_size'], par['room_width'], par['room_height']]))
    par['reward_vectors'] = np.float32(np.zeros([par['batch_size'], par['room_width'], par['room_height'], par['num_rew_tuned']]))
    par['reward_locations'] = np.float32(np.zeros([par['batch_size'], 2]))

    par['reward_positions'] = [[1,1], [par['room_width']-2, par['room_height']-2]]

    for i in range(par['batch_size']):
        reward_order = np.random.permutation(len(par['rewards']))
        for j, ind in enumerate(reward_order):

            locs = par['reward_positions'][j]
            rew = par['rewards'][ind]
            vec = par['reward_vector_refs'][ind]
            
            par['reward_scores'][i,locs[0],locs[1]] = rew
            par['reward_vectors'][i,locs[0],locs[1],:] = vec
            par['reward_locations'][i,:] = [locs[0], locs[1]]

    par['starting_locations'] = \
        np.stack([np.random.choice(par['room_width'], size=par['batch_size']), \
        np.random.choice(par['room_height'], size=par['batch_size'])], axis=1)

    return par['starting_locations'], par['reward_scores'], par['reward_vectors']


def initialize_weight(dims, connection_prob):
    w = np.random.gamma(shape=0.25, scale=1.0, size=dims)
    w *= (np.random.rand(*dims) < connection_prob)
    return np.float32(w)


update_dependencies()
print("--> Parameters successfully loaded.\n")
