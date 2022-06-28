# deeptimeML

# Overview

This repository contains data source and codes used in the paper entitled "Reconstructing Earth’s Atmospheric Oxygenation History Using Machine Learning" by G. Chen et al. We propose an independent new strategy – (unsupervised/supervised) machine learning with global mafic igneous geochemistry big data to explore atmospheric oxygenation over the last 4.0 Gyr. We observe an overall two-step rise of atmospheric O2 similar to the published curves derived from independent sediment-hosted paleo-oxybarometers but with a more detailed fabric of O2 fluctuations superimposed. These additional, shorter-term fluctuations are also consistent with previous but less well-established suggestions of O2 variability. We compiled 1 data source and 8 codes to generate the diagrams in Fig. 3-Fig. 6 and supplementary figures. Here is the list of data and code files.

Data source -- We compiled the global mafic igneous composition data (New_ign1_mafic_final.xlsx) from EarthChem data repository (http://portal.earthchem.org/, assessed Feb, 2022) which includes PetDB, GEOROC, NAVDAT, and USGS database simultaneously. 

MCbg.m -- Calculating the time series of global mean mafic geochemistry composition using the weighted bootstrap sampling method of Keller and Schoene (2012). We have added some data filtering processes in this code, including outlier filtering, RAR-based filtering and others. Please update the mcstask.m when using the StatisticsGeochemistry-master toolboox. 

ClusterA.py -- Calculating the correlation matrix of results from global mafic igneous geochemical big data for all elements.

SOMbg.m -- Self-organizing analysis of global mafic igneous geochemical compositon data, to delineate major trends, jumps, and clusters in big igneous geochemical time series dataset.

PCA.py -- Principal component analysis of global mafic igneous geochemistry data, to supplement SOM results by investigating the details of temporal variations observed via the first principal component (PCA1).

SVR_main.py -- Predicting atmospheric O2 content using Support Vector Regression with global mafic igneous geochemical big data.

RF_main.py -- Predicting atmospheric O2 content using Random Forests with global mafic igneous geochemical big data.

ANN_main.py -- Predicting atmospheric O2 content using Artificial Neural Network with global mafic igneous geochemical big data.

# System Requirements

The codes used in this paper were compiled on the MATLAB and PYTHON.

# Installation Guide

The mat codes require the MATLAB platform installed on the PC or Laptop computer. To use the included MCbg.m and SOMbg.m codes, one must add the toolbox of StatisticalGeochemistry-master (https://github.com/brenhinkeller/StatisticalGeochemistry) and SOM-Toolbox-master (http://www.cis.hut.fi/projects/somtoolbox/), respectively. This can be done (among other ways) by right-clicking the folder containing this repository in Matlab and selecting “Add to Path” > “Selected Folders and Subfolders.” Individual functions and scripts can then be run from the Matlab editor, the command window, or from the command line.

The py codes require the Python complier installed on the PC or Laptop computer. To use the included PCA.py, SVR_main.py, RF_main.py and AA_main.py codes, one must install the sklearn, matplotlib, pandas and other basic function libraries on your Python complier. More details for installation and instruction of sklearn can be found at  https://scikit-learn.org/stable/install.html.


