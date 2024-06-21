# XAI-CMAPPS-Encoder-Transformer
explainable AI implemented for RUL prediction for the CMAPPS dataset, t-SNE and LIME are used for explainability, an Encoder Transformer model is used as prognostics model

Adapted from: https://github.com/Ogunfool/EncoderTransformerArchitecture_CMAPPS.git


# Analyse_data.ipynb:

This notebook is all my own work used for the analysis of the data. 
It includes data cleaning, data exploration, and data visualization. 
It includes the data analysis, including correlation covariance and t-SNE plots.
This file was also used by me personally to get a general feeling of the data. 
The t-SNE embeddings are pre-traine by me and can be found in the foler 'Embedded_Arrays'.
To rerun the t-SNE embeddings, run the code at the bottom of the notebook.
Lastly, to change the dataset to be analysed, please alter the dataset_id at the top of the notebook

# helper_functions.py:

Contains all the functions used for the data analysis in the Analyse_data.ipynb notebook.

# dataset_prep.py:

This prepares the dataset for the Encoder-Transformer, 
I left most unchanged, except for the splitting of the 
training set in a test and validation set. instead of 
only a validation set.

# Main_Single_Condt.ipynb:

Runs the Encoder-Transformer on the single condition dataset. The general structure of 
the notebook is the same as the original notebook, but I have made this notebook work 
with the new test set. Also I added my own data visualisation methods from helper_functions.py.
Furthermore, this notebook now uses pretrained models for evaluating the test set, 
these models can be found in the folder 'Pretrained_Models'. All code for the section Post-Model 
Interpretability is mine. 

# Encoder_Transformer_Layers.ipynb:

Did not alter this notebook, this is the original notebook

# Transformer_loss_activation_and_optimization.ipynb:

Did not alter this notebook, this is the original notebook

# kmeans_Clustering_Model.ipynb:

Did not alter this notebook, this is the original notebook.
Copied the kmeans clustering algorithm to the helper_functions.py file
to visualise the effect of the clustering. 
