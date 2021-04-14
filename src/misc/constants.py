import multiprocessing
import numpy as np

## Constants
MODEL_TYPE_XGBOOST = "XGBOOST"
MODEL_TYPE_RF = "RF"
MODEL_TYPE_ANN = "ANN"
SCORING_TYPE_F1 = "f1"
SCORING_TYPE_ROC = "roc"

## Params
INPUT_FILE = "data/input/MV_EVENT_human_only.tsv"
NUM_OF_JOBS = int(multiprocessing.cpu_count()*0.9)
CV_NUM_OF_JOBS = NUM_OF_JOBS
ANN_DEVICE = "cuda"
ANN_VERBOCITY = 0
GRID_VERBOCITY=1
SCORING = "f1"
PARAMS = {
    "model_type" : MODEL_TYPE_RF,
    "num_rows" : 0,
    "cv_iters" : 100,
    "node_2_vec_params": {
            "ne_p" : 0.6,
            "ne_q" : 0.5,
            "ne_dimensions" : 208,
            "ne_num_walks" : 10,
            "ne_walk_length" : 140,
            "ne_window_size" : 120,
            "ne_num_iter" : 7,
            "ne_workers" : NUM_OF_JOBS
    }
}

NODE_2_VEC_PARAMS_SET = {
    "ne_p" : np.arange(0.1,1.0,0.1),
    "ne_q" : np.arange(0.1,1.0,0.1),
    "ne_dimensions" : range(64,256,16),
    "ne_num_walks" : range(10,20,4),
    "ne_walk_length" : range(80,240,20),
    "ne_window_size" : range(80,240,20),
    "ne_num_iter" : range(5,10,2),
    "ne_workers" : [NUM_OF_JOBS]
}

