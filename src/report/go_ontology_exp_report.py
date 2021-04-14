from loguru import logger
import pandas as pd
from visualization import plotter
import numpy as np
import plotly.express as px

def report():
    
    report_file = "results/go_experiment/result_2021_03_19_07_04_27.csv"
    logger.info(f"Generating go ontology report for - {report_file}")

    report_dir = "results/reports/go_experiment"

    # read the report
    report_df = pd.read_csv(report_file)

    # number of columns and number of rows
    logger.info(f"Number of rows: {len(report_df)}")

    # drop the experiments that failed
    report_df = report_df.dropna(subset=["au_pr"])
    logger.info(f"Rows after dropping NA: {len(report_df)}")

    # keep only experiments with go terms
    #report_df = report_df[report_df["config.go_term_types"].isin(["mp","cc","bp","all"])]

    # sort by au_pr,au_roc
    report_df = report_df.sort_values(by=["au_pr","au_roc"],ascending=False)

    # get number of experiments for every go term type
    count_per_go_term_type = report_df.groupby(["config.go_term_types"]).count()["au_pr"]
    logger.info("Number of observatons per term type")
    print(count_per_go_term_type)

    # get the highest scoring term on average
    count_per_go_term_type = report_df.groupby(["config.go_term_types"]).mean()["au_pr"]
    logger.info("Avergae AU-PR")
    print(count_per_go_term_type)
    
    # duplicate lowest row for cc
    cc_df = report_df[report_df["config.go_term_types"] == "cc"]
    cc_df_subset = cc_df.sort_values(by=["au_pr"],ascending=True).head(1)
    
    # duplicate lowest two rows for mp
    mp_df = report_df[report_df["config.go_term_types"] == "mp"]
    mp_df_subset = mp_df.sort_values(by="au_pr",ascending=True).head(2)
    
    # append these to main df
    report_df = report_df.append(cc_df_subset)
    report_df = report_df.append(mp_df_subset)

    # get number of experiments for every go term type after appending
    count_per_go_term_type = report_df.groupby(["config.go_term_types"]).count()["au_pr"]
    logger.info("Number of observatons per term type after appending")
    print(count_per_go_term_type)
    
    # create heatmap for go_term_types : au_pr
    heatmap_data,x_labels,y_labels = _create_data_for_heatmap(report_df,"config.go_term_types","au_pr")
    fig = px.imshow(heatmap_data,
                    labels=dict(x="Experiment Index", y="GO Term Type", color="AU_PR"),
                    x=x_labels,
                    y=y_labels
                )
    img_file = f"{report_dir}/go_term_au_pr.png"
    logger.info(f"Saving go_term_type:au_pr heatmap to : {img_file}")
    fig.write_image(img_file)

    # create heatmap for go_term_types : au_roc
    heatmap_data,x_labels,y_labels = _create_data_for_heatmap(report_df,"config.go_term_types","au_roc")
    fig = px.imshow(heatmap_data,
                    labels=dict(x="Experiment Index", y="GO Term Type", color="AU_ROC"),
                    x=x_labels,
                    y=y_labels
                )
    img_file = f"{report_dir}/go_term_au_roc.png"
    logger.info(f"Saving go_term_type:au_roc heatmap to : {img_file}")
    fig.write_image(img_file)
    
    # create scatter plot for rf_edge_operator
    fig = px.scatter(report_df, x="au_pr", y="au_roc",color="config.rf_edge_operator")
    img_file = f"{report_dir}/rf_edge_operator_scatter.png"
    logger.info(f"Saving rf_edge_operator_scatter plot to : {img_file}")
    fig.write_image(img_file)

    # create scatter plot for feature_edge_operator
    fig = px.scatter(report_df, x="au_pr", y="au_roc",color="config.fb_binary_operator")
    img_file = f"{report_dir}/fb_edge_operator_scatter.png"
    logger.info(f"Saving fb_edge_operator_scatter plot to : {img_file}")
    fig.write_image(img_file)
  
    # create scatter plot for go_node2vec_num_negative_samples
    report_df["config.go_node2vec_num_negative_samples"] = report_df["config.go_node2vec_num_negative_samples"].apply(lambda x: str(x))
    fig = px.scatter(report_df, 
                     x="au_pr", 
                     y="au_roc",
                     color="config.go_node2vec_num_negative_samples",
                     labels={"config.go_node2vec_num_negative_samples":"Num negative Samples"},
                     title="Gene ontology graph negative sampling"
                    )
    img_file = f"{report_dir}/go_node2vec_num_negative_samples_scatter.png"
    logger.info(f"Saving go_node2vec_num_negative_samples scatter plot to : {img_file}")
    fig.write_image(img_file)

    # create scatter plot for iptmnet_node2vec_num_negative_samples
    report_df["config.iptmnet_node2vec_num_negative_samples"] = report_df["config.iptmnet_node2vec_num_negative_samples"].apply(lambda x: str(x))
    fig = px.scatter(report_df, 
                     x="au_pr",
                     y="au_roc",
                     color="config.iptmnet_node2vec_num_negative_samples",
                     labels={"config.iptmnet_node2vec_num_negative_samples":"Num negative Samples"},
                     title="Iptmnet graph negative sampling"
                     )
    img_file = f"{report_dir}/iptmnet_node2vec_num_negative_samples_scatter.png"
    logger.info(f"Saving iptmnet_node2vec_num_negative_samples scatter plot to : {img_file}")
    fig.write_image(img_file)
    

    logger.info(f"Saved report to : {report_dir}")


def _create_data_for_heatmap(report_df,index_var,value_var):
    # get term type names
    term_types = report_df[index_var].unique().tolist()

    # get the number of observations for every term type
    num_obs = report_df.groupby([index_var]).count()[value_var].iloc[0]
    
    # create empty array to hold values
    data_np = np.empty((len(term_types),num_obs))

    reports_df_value = report_df[[index_var,value_var]]

    for index,term_type in enumerate(term_types):
        values_np = reports_df_value[reports_df_value[index_var] == term_type].drop(columns=[index_var]).to_numpy().reshape(-1)
        data_np[index] = values_np

    y_axis_labels = term_types
    x_axis_labels = list(map(lambda x: f"{value_var}_exp_{x}",range(0,num_obs)))

    return data_np,x_axis_labels,y_axis_labels

def _create_data_for_scatter_plot(report_df,target_variable,x_value_var,y_value_var):
    pass

if __name__ == "__main__":
    report()

