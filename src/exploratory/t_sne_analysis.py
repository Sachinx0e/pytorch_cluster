from sklearn.manifold import TSNE
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.express as px
from loguru import logger

def perform_go():
    logger.info("Loading data")
    go_embeddings_pd = pd.read_csv("data/output/go_embeddings_copy.csv")
    go_mapping_pd = pd.read_csv("data/output/go_parent_mapping.csv")
    root_parent_labels = go_mapping_pd["root_parent"].apply(lambda x: to_go_label(x))

    # get the embeddings
    go_embeddings_np = go_embeddings_pd.drop(columns="protein").to_numpy()
    
    # perform tSNE
    n_terms = 5000
    logger.info("Performing tSNE") 
    embedding_transformed = TSNE(n_components=2,perplexity=30,n_iter=1000).fit_transform(go_embeddings_np[:n_terms])

    # plot
    logger.info("Plotting")
    fig = px.scatter(x=embedding_transformed[:,0],
                        y=embedding_transformed[:,1],
                        color=root_parent_labels[:n_terms],
                        hover_name=go_mapping_pd["go_node"][:n_terms])
    #fig.write_image("results/exploratory/figures/go_embedding.png")
    fig.show()

def perform_proteins():
    logger.info("Loading data")
    features_pd = pd.read_csv("data/output/embeddings_only_go.csv")
    ec_category = pd.read_csv("data/output/ec_category.csv")

    proteins_pd = features_pd
    proteins_pd["ec_category"] = ec_category["ec_category"]
    #proteins_pd = proteins_pd[proteins_pd["ec_category"] != "-1"]
    #proteins_pd = proteins_pd[proteins_pd["ec_category"] != "[]"]

    # get the embeddings
    features_np = proteins_pd.drop(columns=["protein","ec_category"]).to_numpy()
    
    # perform tSNE
    n_terms = features_np.shape[0]
    #n_terms = 1000
    logger.info("Performing tSNE") 
    embedding_transformed = TSNE(n_components=2,perplexity=50,n_iter=1000).fit_transform(features_np[:n_terms])

    # plot
    logger.info("Plotting")
    fig = px.scatter(x=embedding_transformed[:,0],
                        y=embedding_transformed[:,1],
                        color=proteins_pd["ec_category"][:n_terms],
                        hover_name=proteins_pd["protein"][:n_terms])
    #fig.write_image("results/exploratory/figures/go_embedding.png")
    fig.show()

def to_go_label(go_id):
    if go_id == "GO:0008150":
        return "biological_process"
    elif go_id == "GO:0003674":
        return "molecular_function"
    elif go_id == "GO:0005575":
        return "cellular_component"
    else:
        return "unknown"
