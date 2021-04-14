import random
from collections.abc import Iterable
import pickle
import numpy as np
from tqdm import tqdm
from pytorch_lightning.metrics import Accuracy
import torch
import os
from p_tqdm import p_umap
from pathos.multiprocessing import ProcessingPool as Pool
import multiprocessing

def generate_random_params(params, n_iter, seed=10):
    
    random_params = []
    random.seed(seed)

    # loop for n_iter
    for _iter in range(0,n_iter):
        random_param = {}
        # loop over params
        for key, values in params.items():
            # assert values is a list
            assert isinstance(values,Iterable)

            # select a random item out of values
            value = random.choice(values)

            random_param[key] = value

        random_params.append(random_param)

    return random_params


def create_chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

def model_qualifier(model_type):
    return model_type[:2]

def save_ann_model(model,file_name):
    # save actual model
    import joblib
    import copy 

    # copy the nn module
    torch_nn_module = copy.deepcopy(model.module_).to('cpu')

    # reset the net to cpu
    model.set_params(device="cpu")
    model = model.initialize()
    
    # attach the old nn module
    model.module_ = torch_nn_module

    # dump to disk
    joblib.dump(model,file_name)
    

def load_ann_model(file_name):
    import joblib
    model = joblib.load(file_name)
    return model


def save_model(model, file_name):
    import joblib
    from sklearn.utils.validation import check_is_fitted
    check_is_fitted(model)
    joblib.dump(model,file_name)

def load_model(file_name):
    import joblib
    return joblib.load(file_name)

def save_scaler(scaler, file_name):
    import joblib
    from sklearn.utils.validation import check_is_fitted
    check_is_fitted(scaler)
    joblib.dump(scaler,file_name)
    

def load_scaler(file_name):
    import joblib
    return joblib.load(file_name)

def clean_proteins(protein):
    cleaned = protein.split("-")[0]
    return cleaned

def get_embedding_vector_for_edge(edge,embeddings_pd,operator="l1"):
    source_node = edge[0]
    target_node = edge[1]
    source_embedding = embeddings_pd[embeddings_pd["protein"] == source_node].drop(columns=["protein"]).to_numpy()
    target_embedding = embeddings_pd[embeddings_pd["protein"] == target_node].drop(columns=["protein"]).to_numpy()
    
    # if we get empty vectors then zero arrs
    if source_embedding.size == 0:
        source_embedding = np.zeros(target_embedding.shape)

    if target_embedding.size == 0:
        target_embedding = np.zeros(source_embedding.shape)

    if operator == "l1":
        embedding_np = operator_l1(source_embedding,target_embedding)
    elif operator == "l2":
        embedding_np = operator_l2(source_embedding,target_embedding)
    elif operator == "hadamard":
        embedding_np = operator_hadamard(source_embedding,target_embedding)
    elif operator == "avg":
        embedding_np = operator_avg(source_embedding,target_embedding)
    elif operator == "cat":
        embedding_np = operator_cat(source_embedding,target_embedding)
    else:
        raise Exception(f"Unknown operator : {operator}")
    
    return embedding_np

def get_embeddings_vector(edge_list,embeddings_pd,operator="l1",show_progress=True):
    embeddings = []
    cpu_count = multiprocessing.cpu_count()
    pool = Pool(processes=cpu_count-1)
    do_work = lambda edge: get_embedding_vector_for_edge(edge,embeddings_pd,operator=operator)

    if show_progress == True:
        embeddings = p_umap(do_work,
                        edge_list,
                        num_cpus=cpu_count)
    else:
        embeddings = pool.map(do_work, edge_list)
    
    embeddings_vec = np.concatenate(embeddings,axis=0)
    return embeddings_vec

def calculate_accuracy(labels_predicted,labels_actual):
    accuracy = Accuracy()
    return accuracy(labels_predicted,labels_actual)


# define the binary operators
def operator_hadamard(u, v):
    return u * v

def operator_l1(u, v):
    return np.abs(u - v)

def operator_l2(u, v):
    return (u - v) ** 2

def operator_avg(u, v):
    return (u + v) / 2.0

def operator_cat(u, v):
    return np.concatenate([u,v],axis=1)

class IKGDataSet(torch.utils.data.Dataset):

    def __init__(self,embeddings, labels):
        self.embedddings = embeddings
        self.labels = labels

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self,index):
        # get embedding row
        embedding = self.embedddings[index]

        # get the label
        label = self.labels[index]

        return embedding, label


def get_best_experiment(results_dir):
    files = os.listdir(results_dir)
    
    # sort in descending ordert
    files.sort(reverse=True)

    best_result = files[0]

    # get the absolute path
    best_result = os.path.join(results_dir,best_result)

    return best_result

def extract_config(result_dict):
    config = {}
    for key,value in result_dict.items():
        # check if this is config key
        if "config." in key:
            # if yes then extract the name
            key_name = key.replace("config.","")

            config[key_name] = value

    return config
    
    