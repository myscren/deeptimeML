# deeptimeML
Overview
This repository contains data source and code used in the paper entitled "Reconstructing Atmospheric Oxygenation History Using Machine Learning". Unsupervised and supervised machine learning were used to process global mafic igneous geochemistry big data (spanning 45 elements) across the last 4.0 Gyrs. We observe an overall two-step rise of atmospheric O2 similar to the published curves derived from independent sediment-hosted paleo-oxybarometers but with a more detailed fabric of O2 fluctuations superimposed. We compiled 1 data source and 3 codes to generate the diagrams in Fig. 3-Fig. 5. Here is the list of data and code files.

Basalt.xlsx -- We compiled the global mafic igneous composition data based on Keller & Schoene and Dien et al that was originally extracted from EarthChem, GEOROC, and USGS repositories. 

MCbg.m -- Calculating the time series of global mean mafic geochemistry composition using the weighted bootstrap sampling method of Keller and Schoene (2012).

ClusterA.py -- Calculating the correlation matrix of results from mafic igneous geochemical big data for 45 elements.

SOMbg.m -- Self-organizing analysis of mafic igneous geochemical compositon data.

SVR_main.py -- Calculating the atmospheric O2 content using Support Vector Regression with mafic igneous geochemical big data.

RF_main.py -- Calculating the atmospheric O2 content using Random Forests with mafic igneous geochemical big data.

System Requirements
The codes used in this paper were compiled on the MATLAB and PYTHON.

Installation Guide
The mat codes require the MATLAB platform installed on the PC or Laptop computer. To use the included MCbg.m and SOMbg.m codes, one must add the toolbox of StatisticalGeochemistry-master (https://github.com/ brenhinkeller/StatisticalGeochemistry) and SOM-Toolbox-master (http://www.cis.hut.fi/projects/somtoolbox/), respectively. This can be done (among other ways) by right-clicking the folder containing this repository in Matlab and selecting “Add to Path” > “Selected Folders and Subfolders.” Individual functions and scripts can then be run from the Matlab editor, the command window, or from the command line.

The mat codes require the Python complier installed on the PC or Laptop computer. To use the included SVR_main.py and RF_main.py codes, one must install the sklearn, matplotlib, pandas and other basic function libraries on your Python complier. More details for installation and instruction of sklearn can be found at  https://scikit-learn.org/stable/install.html.


