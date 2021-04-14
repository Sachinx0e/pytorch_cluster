import networkx as nx
from loguru import logger
from node2vec import Node2Vec
import torch
import obonet
from tqdm import tqdm

# generate a random graph
logger.info("Generaring graph")
graph = nx.gn_graph(50)
#graph = obonet.read_obo("data/input/ontology/pro_reasoned.obo")

# init tensorboard
from datetime import datetime
date_serialized = str(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(f'runs/embedding_explore/{date_serialized}')

# convert to digraph
graph = nx.DiGraph(graph)

# remove node attributes
for i, feat_dict in tqdm(graph.nodes(data=True)):
    feat_dict.clear()

# graph properties
logger.info(f"Nodes : {len(graph.nodes())}")
logger.info(f"Edges : {len(graph.edges())}")

# convert to pytorch-geomtric dataset
from torch_geometric.utils import from_networkx
logger.info("Converting to torch_geomtric data")
data = from_networkx(graph)

# create the ndode2vec model
from torch_geometric.nn import Node2Vec

logger.info("Creating the model")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Node2Vec(data.edge_index, embedding_dim=128, walk_length=50,
                     context_size=10, walks_per_node=10,
                     num_negative_samples=1, p=1, q=1, sparse=True).to(device)

# define a dataloader
loader = model.loader(batch_size=128, shuffle=True, num_workers=4)

# define optimizer
optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

# define training loop
def train():
    model.train()
    total_loss = 0
    for pos_rw, neg_rw in loader:
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def get_embedding():
    model.eval()
    embedding = model()
    return embedding

# start training
num_epochs = 10
for epoch in range(0, num_epochs):
    loss = train()
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')
    writer.add_scalar('loss',loss,epoch)

# get the embeddings
embeddings = get_embedding()

logger.info(f"Shape of embeddings : {embeddings.shape}")
