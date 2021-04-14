# %%
import pandas as pd
from loguru import logger
from functools import partial
from tqdm import tqdm
import numpy as np

# %%
# load the sequences
logger.info("Loading sequences")
sequences_df = pd.read_csv("../../data/input/ml_data/iptmnet_seq/sequence.csv")
print(sequences_df.head())

# load protein sequences
proteins_txt = open("../../data/input/ml_data/iptmnet_network_proteins.txt")
proteins = proteins_txt.read().splitlines()


# %%
# create protein,seq pairs
def generate_seq_tuple(protein_name, sequences_df):
    result_df = sequences_df[sequences_df["ID"] == protein_name]
    if len(result_df.index) > 0:
        sequence = result_df.iloc[0]["SEQ"]
        return (protein_name, sequence)
    else:
        return (protein_name, "AAAAAAAAAAA")

logger.info("Creating sequence tuples")
seq_tuples = []
for protein in tqdm(proteins):
    seq_tuple = generate_seq_tuple(protein,sequences_df)
    seq_tuples.append(seq_tuple)

sequences_df = pd.DataFrame.from_records(seq_tuples)

# %%
sequences_df["len"] = sequences_df[1].apply(lambda x: len(x))
sequences_df.head()


# %%
sequences_df["len"].hist(bins=100)


# %%
sequences_df = sequences_df.sort_values(by=["len"],ascending=False)
sequences_df.head()


# %%
# load the model
from allennlp.commands.elmo import ElmoEmbedder
from pathlib import Path
model_dir = Path('../../data/pretrained/seqvec/uniref50_v2')
weights = model_dir / 'weights.hdf5'
options = model_dir / 'options.json'
embedder = ElmoEmbedder(options,weights, cuda_device=0)

# %%
sequence = sequences_df.iloc[1300:1385][1]
#print(sequence.to_list())
embedding = embedder.embed_sentence(list(sequence.to_list()))
print(embedding)
import torch
#torch.cuda.empty_cache()

# %%
import torch
protein_embd = torch.tensor(embedding).sum(dim=0).mean(dim=0)
protein_embd.shape
# %%
