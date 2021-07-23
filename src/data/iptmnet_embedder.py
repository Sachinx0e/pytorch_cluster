from loguru import logger
import networkx as nx
import torch
from torch_geometric.nn import Node2Vec
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split

def embed(iptmnet_graph=None,
          embedding_dim=128,
          walk_length=100,
          context_size=20,
          walks_per_node=10,
          num_negative_samples=1,
          p=0.6,
          q=0.9,
          sparse=True,
          save=True,
          num_epochs=10001,
          show_progress=True,
          patience=10,
          early_stopping_delta=0.0001,
          device=None
        ):

    # load the iptmnet graph
    if iptmnet_graph == None:
        iptmnet_graph_file = "data/input/iptmnet_graph_cleaned_directed.gml"
        logger.info(f"Loading data from {iptmnet_graph_file}")
        iptmnet_graph = nx.read_gml(iptmnet_graph_file)

    # print summary
    logger.info(f"Number of nodes : {len(iptmnet_graph.nodes())}")
    logger.info(f"Number of edges : {len(iptmnet_graph.edges())}")

    # remove experimentally validated edges
    import data.data_utils as du
    experimental_edges = du.get_experimental_edges()
    positive_edges = list(iptmnet_graph.edges())
    _, val_edges = train_test_split(positive_edges,random_state=20,shuffle=True,test_size=0.10)
    edges_to_remove = experimental_edges + val_edges

    iptmnet_graph.remove_edges_from(edges_to_remove)

    # print summary
    logger.info(f"Number of nodes after removing experimental : {len(iptmnet_graph.nodes())}")
    logger.info(f"Number of edges after removing experimental: {len(iptmnet_graph.edges())}")       
    
    # convert to pytorch-geomtric dataset
    from torch_geometric.utils import from_networkx
    logger.info("Converting to torch_geomtric data")
    data = from_networkx(iptmnet_graph)
    nodes = list(iptmnet_graph.nodes())

    # create the node2vec model
    logger.info("Creating the model")
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = Node2Vec(data.edge_index,
                     embedding_dim=embedding_dim,
                     walk_length=walk_length,
                     context_size=context_size,
                     walks_per_node=walks_per_node,
                     num_negative_samples=num_negative_samples,
                     p=p, 
                     q=q,
                     sparse=True).to(device)

    # define a dataloader
    logger.info("Creating the data loader")
    loader = model.loader(batch_size=128, shuffle=True, num_workers=0)


    # define optimizer
    logger.info("Creating the optimizer")
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.001)

    # define training loop
    def train():
        model.train()
        total_loss = 0
        for pos_rw, neg_rw in tqdm(loader,disable= not show_progress):
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.detach().item()
        return total_loss / len(loader)

    @torch.no_grad()
    def get_embedding():
        model.eval()
        embedding = model()
        return embedding


    # start training
    logger.info("Started training")
    num_epochs = num_epochs
    best_loss = 0
    num_of_epochs_waited = 0
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
    if save == True:
        logger.info("Saving embeddings")
        embeddings_df.to_csv("data/output/iptmnet_embeddings.csv",index=False)
        logger.info("Done")

    return embeddings_df