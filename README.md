# **DeeptimeML**
## Overview
This repository contains data source and codes used in the paper entitled "Reconstructing Earth’s Atmospheric Oxygenation History Using Machine Learning" by G. Chen et al. We propose an independent new strategy – (unsupervised/supervised) machine learning with global mafic igneous geochemistry big data to explore atmospheric oxygenation over the last 4.0 Gyr. We observe an overall two-step rise of atmospheric O² similar to the published curves derived from independent sediment-hosted paleo-oxybarometers but with a more detailed fabric of O² fluctuations superimposed. These additional, shorter-term fluctuations are also consistent with previous but less well-established suggestions of O² variability. We compiled 1 data source and 8 codes to generate the diagrams in Fig. 3-Fig. 6 and supplementary figures. Here is the list of data and code files.
### Data files 
- [Data source](https://github.com/myscren/deeptimeML/tree/main/Data%20source)&nbsp;&nbsp;-- We compiled the global mafic igneous composition data (New_ign1_mafic_final.xlsx) from EarthChem data repository (http://portal.earthchem.org/, assessed Feb, 2022) which includes PetDB, GEOROC, NAVDAT, and USGS database simultaneously. This file also includes mafic igneous geochemical time series data (New_ign1_mafic_ts_final.xlsx) obtained by the weighted bootstrap sampling method with different filtering (e.g., RAR based filtering) and other schemes.


### Code files 
- [MCbg.m](https://github.com/myscren/deeptimeML/tree/main/Codes/Data%20prepration/)&nbsp;&nbsp;-- Calculating the time series of global mean mafic geochemistry composition using the weighted bootstrap sampling method of Keller and Schoene (2012). We have added some data filtering processes in this code, including outlier filtering, RAR-based filtering and others. Please also update the mcstask.m when using the StatisticsGeochemistry-master toolboox.

- [ClusterA.py](https://github.com/myscren/deeptimeML/tree/main/Codes/Unsupervised%20learning)&nbsp;&nbsp;-- Calculating the correlation matrix of results from global mafic igneous geochemical big data for all elements.

- [SOMbg.m](https://github.com/myscren/deeptimeML/tree/main/Codes/Unsupervised%20learning)&nbsp;&nbsp;-- Self-organizing analysis of global mafic igneous geochemical compositon data, to delineate major trends, jumps, and clusters in big igneous geochemical time series dataset.

- [PCA.py](https://github.com/myscren/deeptimeML/tree/main/Codes/Unsupervised%20learning)&nbsp;&nbsp;-- Principal component analysis of global mafic igneous geochemistry data, to supplement SOM results by investigating the details of temporal variations observed via the first principal component (PCA1).

- [SVR_main.py](https://github.com/myscren/deeptimeML/tree/main/Codes/Supervised%20learnig)&nbsp;&nbsp;-- Predicting atmospheric O² content using Support Vector Regression with global mafic igneous geochemical big data.

- [RF_main.py](https://github.com/myscren/deeptimeML/tree/main/Codes/Supervised%20learnig)&nbsp;&nbsp;-- Predicting atmospheric O² content using Random Forests with global mafic igneous geochemical big data.

- [ANN_main.py](https://github.com/myscren/deeptimeML/tree/main/Codes/Supervised%20learnig)&nbsp;&nbsp;-- Predicting atmospheric O² content using Artificial Neural Network with global mafic igneous geochemical big data.


## System Requirements
### OS Requirements
The developmental version of the package has been tested on the following systems:
- Windows 10

Installation time is less than 10 minutes.
### Software Requirements
The codes used in this paper were compiled on the MATLAB 2016a and Python 3.8.<br>
The versions of other packages are, specifically:
```
numpy==1.21.0
pandas==1.3.0
sklearn==0.0
matplotlib==3.4.2
seaborn=0.11.2
...
```

## Installation Guide
The mat codes require the MATLAB platform installed on the PC or Laptop computer. To use the included MCbg.m and SOMbg.m codes, one must add the toolbox of [StatisticalGeochemistry](https://github.com/brenhinkeller/StatisticalGeochemistry) and [SOM-Toolbox](http://www.cis.hut.fi/projects/somtoolbox/), respectively. This can be done (among other ways) by right-clicking the folder containing this repository in Matlab and selecting “Add to Path” > “Selected Folders and Subfolders.” Individual functions and scripts can then be run from the Matlab editor, the command window, or from the command line.

The py codes require the Python complier installed on the PC or Laptop computer. To use the included PCA.py, SVR_main.py, RF_main.py and AA_main.py codes, one must install the sklearn, matplotlib, pandas and other basic function libraries on your Python complier. More details for installation and instruction of sklearn can be found at https://scikit-learn.org/stable/install.html.
### Download from Github
```
git clone https://github.com/myscren/deeptimeML.git
cd deeptimeML
```
## Instructions to run
- The mat codes just click the "run" button in MATLAB.<br>
- The py codes need to run according to the following instructions:
```
run PCA.py:
```
python PCA.py
```
run SVR_main.py
```
python SVR_main.py
```
run RF_main.py
```
python RF_main.py
```
run AA_main.py
```
python AA_main.py
```
## Demo
![Figure1](./docs/Demo_test_figure1.jpg)<br>
![Figure2](./docs/Demo_test_figure2.jpg)<br>
##
