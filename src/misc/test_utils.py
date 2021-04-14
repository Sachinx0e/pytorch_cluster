import misc.utils as utils
import numpy as np
import multiprocessing

def test_extract_entities():
    # build test array
    params = params = {
        "ne_p" : np.arange(0.1,1.0,0.1),
        "ne_q" : np.arange(0.1,1.0,0.1),
        "ne_dimensions" : np.arange(64,256,16),
        "ne_num_walks" : np.arange(10,100,10),
        "ne_walk_length" : np.arange(80,240,20),
        "ne_window_size" : np.arange(10,100,10),
        "ne_num_iter" : np.arange(1,5,1),
        "ne_workers" : [multiprocessing.cpu_count()]
    }  

    # generate a set of 10 random params
    random_params = utils.generate_random_params(params, 10)
    
    # assert that 10 sets are present
    assert len(random_params) == 10

    # assert that ne_p has a scalar value
    first_param_set = random_params[0]
    assert isinstance(first_param_set["ne_p"],float)