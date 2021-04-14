import networkx as nx
from node2vec import Node2Vec
import obonet
from loguru import logger

def build():
    # load the obo file
    logger.info("Reading PRO ontology file")
    pro_graph = obonet.read_obo("data/input/ontology/pro_reasoned.obo")
    
    #go_graph = nx.subgraph(go_graph,list(go_graph.nodes()))
    
    logger.info(f"Number of nodes : {len(pro_graph.nodes())}")
    logger.info(f"Number of edges : {len(pro_graph.edges())}")

    # set the number of workers
    import multiprocessing
    num_workers = multiprocessing.cpu_count() - 6

    # TODO : Use a metric to determine the walk length based on graph structure
    walk_length = 200

    # define the model
    node2vec = Node2Vec(pro_graph,
                        dimensions=128,
                        workers=num_workers,
                        walk_length=walk_length, 
                        num_walks=400,
                        seed=10,
                        p=1,
                        q=1
                    )
    
    # train the model
    logger.info("Started training")
    model = node2vec.fit()
    logger.info("Done training")

    # save vectors
    logger.info("Saving vectors")
    model.wv.save_word2vec_format("data/output/pro_embedding.wv")
    logger.info("Done")

if __name__ == "__main__":
    build()