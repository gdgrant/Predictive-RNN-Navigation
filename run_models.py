import numpy as np
from parameters import *
import model
import sys, os
import pickle


def try_model(save_fn):
    # To use a GPU, from command line do: python model.py <gpu_integer_id>
    # To use CPU, just don't put a gpu id: python model.py
    try:
        if len(sys.argv) > 1:
            model.main(save_fn, sys.argv[1])
        else:
            model.main(save_fn)
    except KeyboardInterrupt:
        print('Quit by KeyboardInterrupt.')

###############################################################################
###############################################################################
###############################################################################

updates = {
    'save_fn'           : 'testing',
    'save_fn_suffix'    : '_v0',
    'entropy_cost'          : 0.0001,
    'val_cost'              : 0.1,
}

update_parameters(updates)
try_model(par['save_fn'])
