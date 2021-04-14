# Getting started

Please ensure that you have installed [anaconda](https://docs.anaconda.com/anaconda/install/linux/)

* To create the environment
```
conda env create -f environment.yml
conda activate ikg_env
```

* To run the pipeline
```
python src/main.py --c "go_ontology_exp"
```

* To train the model with best hyper params and generate predictions
```
python src/main.py --c "train_go_ontology"
python src/main.py --c "predict_go_ontology"
```

* To generate reports
```
python src/main.py --c "go_ontology_exp_report"
python src/main.py --c "calculate_metric_for_experimental"
```

## Code Structure
The entry point is `src/main.py`. Modules for every step of the pipeline are organized into their respective
subdirectories. To call the individual modules please take a look at the available commands in `src/main.py`

## Requirements
1. Anaconda
2. Python 3.7
3. Linux
4. NVIDIA GPU with atleast 16GB RAM

