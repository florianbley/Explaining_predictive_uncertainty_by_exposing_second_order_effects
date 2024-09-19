# Explaining Predictive Uncertainty By Exposing Second-Order Effects
This repository contains the code implementation for [Explaining Predictive Uncertainty By Exposing Second-Order Effects](https://arxiv.org/pdf/2401.17441) by Florian Bley, Sebastian Lapuschkin, Wojciech Samek and Grégoire Montavon.

# Setup
Necessary python packages are specified in the requirements.txt file. In
addition to the packages listed in the requirements.txt file, a working installation
of torch is required. 

# Overview
This code implements the feature-flipping experiment described in the main paper. 
The featureflipping_experiment.py script loads datasets, trains and saves deep ensembles 
or MC-dropout ensembles, trains a KDE generative model, and performs feature flipping 
for specified uncertainty explanations.

Datasets and preprocessing scripts are in the datasets folder. 
Single models are implemented in CNN.py and MLP.py.

Deep ensemble uncertainty estimators are in Ensemble-regressors.py, while MC-Dropout is 
implemented in Ensemble_regressor_MC_dropout.py. For reproducibility, MC-Dropout is treated as 
an ensemble of neural networks with different dropout seeds.

The demo.py script provides a simple example of explaining predictive uncertainty for a 
tabular dataset.

# Running Feature-Flipping
To rerun the feature flipping experiment, run the featureflipping_experiment.py script. 
The featureflipping_experiment.py will
1) load the datasets specified in the dataset_names list of the main block
2) Train either a deep ensemble or MC-dropout ensemble, 
as specified by the uncertainty_type ('ensemble' or 'MC_dropout')
3) explain predictive uncertainty with the methods specified in the explanation_methods list
4) run the feature flipping experiment, calculate the AUFC values
5) save results as pkl files in the results folder.

To change the tested datasets, number of models in the ensembles, the uncertainty estimator,
or the explanation methods, modify the corresponding variables in the main block of the script.
 
For your convenience, refer to the analyse_featureflipping_results.py script
which loads all pkl files and prints them out in a latex table format.