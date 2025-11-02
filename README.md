# ML_evolution_SARS-CoV-2
Scripts for reproducing data and figures in manuscript:  

Aleksander E. P. Durumeric, Sean McCarty, Jay Smith, Jonas Köhler, Katarina Elez, Lluís Raich, Patricia A. Suriana, and Terra Sztain. "Machine Learning Driven Simulations of the SARS-CoV-2 Fitness Landscape from Deep Mutational Scanning Experiments."

### ML training and MCMC simulation
Code for training ML models and conducting MCMC simulations can be found at: //github.com/SztainLab/mavenets

### Trained models
All trained models used in final publication can be found in [trained_models](https://github.com/SztainLab/ML_evolution_SARS-CoV-2/tree/main/trained_models)

### Model predictions
Prediction tables are formatted as: 

|reference|experiment|tuned|raw|
|---------|----------|-----|---|

where reference refers to the experimental value, experiment refers to the DMS library test set, tuned is the tuned prediction if a tuner is present, otherwise the tuner and raw columns will be identical.

The prediction tables are grouped into: 
1. [base_data](https://github.com/SztainLab/ML_evolution_SARS-CoV-2/tree/main/base_data) containing predictions from models trained on the original WT DMS dataset
2. [exp_heads](https://github.com/SztainLab/ML_evolution_SARS-CoV-2/tree/main/exp_heads) containing predictions from models for comparing predictions with and without tuning.
3. [other_datasets](https://github.com/SztainLab/ML_evolution_SARS-CoV-2/tree/main/other_datasets) containing predictions models trained on each of the additional dataset libraries individually
4. [train_eval](https://github.com/SztainLab/ML_evolution_SARS-CoV-2/tree/main/train_eval) containing 'denoised' predictions of training data with the original model and model trained on aggregated data with per-experiment tuning.

###  Simulation results
Simulation tables are formatted such that each of the first 201 columns contains the amino acid at each of 201 positions in the RBD. The next column 'time' refers to the MCMC step, and 'energy' corresponds to predicted &Delta;logK<sub>D</sub>

Simiulation results can be found in [mcmc](https://github.com/SztainLab/ML_evolution_SARS-CoV-2/tree/main/mcmc). 

Each sim_tables directory contains results from training on either the original (base) dataset or the aggregated dataset with per-experiment tuning. 

[sim_ba1_start](https://github.com/SztainLab/ML_evolution_SARS-CoV-2/tree/main/mcmc/sim_ba1_start/sim_tables_1) contains results of simulations centered around the omicron BA.1 sequence.

The [unlimited](https://github.com/SztainLab/ML_evolution_SARS-CoV-2/tree/main/mcmc/unlimited/sim_tables_1) simulation refers to the simulation without the wild type bias (lambda) and only the maximum mutation number set. The file without the maximum mutation number was too large for Github and may be requested from corresponding author. 

### Reproducing figures 
Code for processing DMS data into training the files found in [raw](https://github.com/SztainLab/ML_evolution_SARS-CoV-2/tree/main/raw) folder \
and for reproducing manuscript figures can be found in the notebook [Final_figures.ipynb](https://github.com/SztainLab/ML_evolution_SARS-CoV-2/blob/main/Final_figures.ipynb)

### Processing Genbank data

The genbank_processing.py and genbank_functions.py scripts are for processing, aligning, and selecting RBD region from Genbank sequences. Raw Genbank data is not provided.

### Creating pie charts

Code necessary for reproducing pie charts can be found in [pie_charts](https://github.com/SztainLab/ML_evolution_SARS-CoV-2/tree/main/pie_charts)
