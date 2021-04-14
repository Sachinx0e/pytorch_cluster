import networkx as nx
from loguru import logger
from node2vec import Node2Vec

# generate a random graph
logger.info("Generaring graph")
graph = nx.gn_graph(500)

# longest path
from networkx.algorithms.dag import dag_longest_path_length
diameter = dag_longest_path_length(graph)
logger.info(f"Graph diameter : {diameter}")

# graph properties
logger.info(f"Nodes : {len(graph.nodes())}")
logger.info(f"Edges : {len(graph.edges())}")

# node2vec
node2vec = Node2Vec(graph,dimensions=128, workers=4)

logger.info("Started training")
model = node2vec.fit()

X = model.wv.get_normed_vectors()
logger.info(f"Embedding shape : {X.shape}")

# generate scatter plot of the embedding
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X)

logger.info(f"Explained variance : {pca.explained_variance_ratio_}")

X_t = pca.transform(X)

import matplotlib.pyplot as plt
plt.scatter(X_t[:,0], X_t[:,1], alpha=0.5)
plt.title('Scatter plot pythonspot.com')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig("scatter.png")

