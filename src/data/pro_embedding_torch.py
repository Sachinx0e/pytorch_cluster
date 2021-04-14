import networkx as nx
from loguru import logger
from torch_geometric.nn import Node2Vec
import torch
import obonet
from tqdm import tqdm
import pandas as pd

def build(seed=10):

    torch.manual_seed(seed)

    # init logger and tensorboard
    from datetime import datetime
    date_serialized = str(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
    logger.add(f"logs/log_pro_embedding_torch_{date_serialized}.txt")
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(f'runs/pro_embedding/{date_serialized}')

    # load the obo file
    logger.info("Reading PRO ontology file")
    #go_graph = nx.gn_graph(50)
    go_graph = obonet.read_obo("data/input/ontology/pro_reasoned.obo")

    # convert to plain old digraph
    go_graph = nx.DiGraph(go_graph)

    # remove node attributes
    logger.info("Clearing noe attributes")
    for _node, feat_dict in tqdm(go_graph.nodes(data=True)):
        feat_dict.clear()

    # print summary
    logger.info(f"Number of nodes : {len(go_graph.nodes())}")
    logger.info(f"Number of edges : {len(go_graph.edges())}")

    # convert to pytorch-geomtric dataset
    from torch_geometric.utils import from_networkx
    logger.info("Converting to torch_geomtric data")
    data = from_networkx(go_graph)
    nodes = list(go_graph.nodes())

    # create the node2vec model
    logger.info("Creating the model")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Node2Vec(data.edge_index,
                     embedding_dim=128,
                     walk_length=100,
                     context_size=10,
                     walks_per_node=10,
                     num_negative_samples=1,
                     p=1, 
                     q=0.5,
                     sparse=True).to(device)


    # define a dataloader
    logger.info("Creating the data loader")
    import multiprocessing
    num_workers = multiprocessing.cpu_count() - 1
    loader = model.loader(batch_size=128, shuffle=True, num_workers=num_workers)

    # define optimizer
    logger.info("Creating the optimizer")
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.001)

    # define training loop
    def train():
        model.train()
        total_loss = 0
        for pos_rw, neg_rw in tqdm(loader):
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
    logger.info("Started training")
    num_epochs = 10000
    best_loss = 0
    patience = 10
    num_of_epochs_waited = 0
    early_stopping_delta = 0.0001
    change_delta = 0.0
    for epoch in range(0, num_epochs):
        loss = train()

        # check if we need to stop early
        if epoch == 0:
            # then just set the best loss value
            best_loss = loss
        else:
            
            # if loss is better than previous loss and difference is more than early stopping delta
            change_delta = best_loss - loss
            if loss < best_loss and change_delta >= early_stopping_delta :

                # update the best loss
                best_loss = loss

                # reset num of epochs waited to zero
                num_of_epochs_waited = 0
            
            # if loss is worse than best loss
            else:
                # if the number of epochs has not improved more than patience, then stop
                if num_of_epochs_waited > patience:
                    logger.info("EARLY STOPPPING")
                    break

                # else update the number of epochs waited
                else:
                    num_of_epochs_waited = num_of_epochs_waited + 1

            
        # log the params
        print(f'Epoch: {epoch:02d} / Loss: {loss:.4f} / Best Loss : {best_loss:.4f} / Change delta : {change_delta: .4f} ')
        writer.add_scalar('loss',loss,epoch)

        # reset the invariants
        loss = 0.0



    # get the embeddings
    embeddings = get_embedding()
    logger.info(f"Shape of embeddings : {embeddings.shape}")

    # to dataframe
    embeddings_np = embeddings.detach().cpu().numpy()
    embeddings_df = pd.DataFrame(data=embeddings_np)
    embeddings_df.insert(0,"protein",nodes)

    # save embeddings
    logger.info("Saving embeddings")
    embeddings_df.to_csv("data/output/pro_embeddings.csv",index=False)
    logger.info("Done")

if __name__ == "__main__":
    build()

    



