import pandas as pd
from loguru import logger
from functools import partial
from tqdm import tqdm
import numpy as np
import torch

def generate_seq_tuple(protein_name, sequences_df):
    result_df = sequences_df[sequences_df["ID"] == protein_name]
    if len(result_df.index) > 0:
        sequence = result_df.iloc[0]["SEQ"]
        return (protein_name, sequence)
    else:
        return (protein_name, "AAAAAAAAAAA")

def create_chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

def build():
    # load the sequences
    logger.info("Loading sequences")
    sequences_df = pd.read_csv("data/input/ml_data/iptmnet_seq/sequence.csv")
    print(sequences_df.head())

    # load protein sequences
    proteins_txt = open("data/input/ml_data/iptmnet_network_proteins.txt")
    proteins = proteins_txt.read().splitlines()

    # create protein,seq pairs
    logger.info("Creating sequence tuples")
    seq_tuples = []
    for protein in tqdm(proteins):
        seq_tuple = generate_seq_tuple(protein,sequences_df)
        seq_tuples.append(seq_tuple)
    
    # create batches
    batch_size = 1
    seq_tuple_batches = list(create_chunks(seq_tuples,batch_size))
    logger.info(f"Number of batches : {len(seq_tuple_batches)}")
    
    # load the model
    logger.info("Loading model")
    from allennlp.commands.elmo import ElmoEmbedder
    from pathlib import Path
    model_dir = Path('data/pretrained/seqvec/uniref50_v2')
    weights = model_dir / 'weights.hdf5'
    options = model_dir / 'options.json'
    embedder = ElmoEmbedder(options,weights, cuda_device=0)

    sequence_embeddings = []
    
    logger.info("Generating embeddings")
    for protein, sequence in tqdm(seq_tuples):
        # create embedding
        embedding = embedder.embed_sentence(list(sequence))
        protein_embd = torch.tensor(embedding).sum(dim=0).mean(dim=0)
        seq_embedding_np = np.array([protein_embd.cpu().numpy()])
        sequence_embeddings.append(seq_embedding_np)
        torch.cuda.empty_cache()
        

    # save
    logger.info("Concatenating embeddings")
    sequence_embeddings_np_all = np.concatenate(sequence_embeddings,axis=0)
    sequence_embedding_df = pd.DataFrame(data=sequence_embeddings_np_all)
    sequence_embedding_df.insert(0, "protein", proteins)
    sequence_embedding_df.to_csv("data/output/seq_embeddings.csv",index=False)

    logger.info("Done")    

if __name__ == "__main__":
    build() 
