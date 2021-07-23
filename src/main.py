import argparse
from loguru import logger

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--command', metavar='--c')
args = parser.parse_args()
command = args.command

logger.info(f"Command is : {command}")

################ DATA #################
if command == "generate_iptmnet_gml":
    from exploratory import generate_graph
    generate_graph.generate_iptmnet_gml()
elif command == "generate_iptmnet_gml_only_phosph":
    from exploratory import generate_graph
    generate_graph.generate_iptmnet_gml_only_phosph()
elif command == "generate_go_gml":
    from exploratory import generate_graph
    generate_graph.generate_go_gml()
elif command == "generate_go_parent_mapping":
    from exploratory import generate_graph
    generate_graph.generate_go_parent_mapping()
elif command == "build_go_embedding":
    from data import go_embedding_torch
    go_embedding_torch.build()
elif command == "build_iptmnet_embedding":
    from data import iptmnet_embedder
    iptmnet_embedder.embed()

################ EXPLORATORY #################
elif command == "tsne_go":
    from exploratory import t_sne_analysis
    t_sne_analysis.perform_go()
elif command == "tsne_proteins":
    from exploratory import t_sne_analysis
    t_sne_analysis.perform_proteins()
elif command == "graph_stats_go":
    from exploratory import graph_stats
    graph_stats.generate_stats_go()
elif command == "graph_stats_iptmnet":
    from exploratory import graph_stats
    graph_stats.generate_stats_iptmnet()
elif command == "build_features_only_go":
    from data import feature_builder
    feature_builder.build_features_only_go(None)
elif command == "build_features_only_go_bp":
    from data import feature_builder
    feature_builder.build_features_only_go_term_type(None,"bp")
elif command == "build_features_only_go_cc":
    from data import feature_builder
    feature_builder.build_features_only_go_term_type(None,"cc")
elif command == "build_features_only_go_mp":
    from data import feature_builder
    feature_builder.build_features_only_go_term_type(None,"mp")
elif command == "train_supervised_rf":
    from train import train_random_forest_supervised
    train_random_forest_supervised.train(max_depth=0.8,
                                         criterion="entropy",
                                         max_features="log2",
                                         min_samples_split=52,
                                         min_samples_leaf=9,
                                         predict=False,
                                         plots=True,
                                         perform_testing=True
                                        )
                                        
# ************* EXPERIMENTS *********** #
elif command == "go_ontology_exp":
    from experiments import go_ontology_exp
    go_ontology_exp.start()

# ************ REPORTS ************ # 
elif command == "go_ontology_exp_report":
    from report import go_ontology_exp_report
    go_ontology_exp_report.report()

elif command == "go_ontology_predict_report_data":
    from report import go_ontology_predict_report
    go_ontology_predict_report.report_data()

elif command == 'go_ontology_predict_report_figures':
    from report import go_ontology_predict_report
    go_ontology_predict_report.report_figures()

elif command == "calculate_metric_for_experimental":
    from report import go_ontology_predict_report
    go_ontology_predict_report.calculate_metric_for_experimental()

elif command == "ptm_event_report":
    from report import ptm_event_report
    ptm_event_report.report()

# *********** TRAIN ************** #
elif command == "train_go_ontology":
    from train import train_go_ontology
    train_go_ontology.train() 

# ************ PREDICT ********** #
elif command == "predict_go_ontology":
    from predict import predict_go_ontology
    predict_go_ontology.predict()

else:
    logger.error(f"Unknown command : {command}")
    import sys
    sys.exit()

