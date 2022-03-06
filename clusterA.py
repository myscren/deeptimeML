import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
data=pd.read_excel('D:/Pycharm/DeeptimeML/Ign/PCA/Mafic_NEW_50.xlsx','Sheet1')
data=data.loc[:,'SiO2':'Cs']
### correlation
data_normalization=(data-np.min(data,axis=0))/(np.max(data,axis=0)-np.min(data,axis=0))
data_corr=data_normalization.corr()
plt.rcParams['font.family']='Times New Roman'
#plt.rcParams['font.size']=8
#plt.rcParams['xtick.labelsize']=8
#plt.rcParams['ytick.labelsize']=8
plt.rcParams['savefig.dpi']=300
num=np.cumsum([0,9,3,29,3])
color=sns.color_palette('hls',8)[4:8]
g=sns.clustermap(data_corr)
x=pd.Series(index=g.data2d.columns)
label=['a','b','c','d','e']
for i in range(4):
    x[num[i]:num[i+1]]=label[i]
color_index=x.map(dict(zip(label,color)))
plt.close()
g=sns.clustermap(data_corr,
               center=0,
               cmap='vlag',
               linewidths=.01,linecolor='w',
               cbar_kws={"orientation": "horizontal"},cbar_pos=(.385, .85, .56, .06),
               figsize=(10,6),dendrogram_ratio=(0.4,0.2),
               row_colors=color_index,
               col_colors=color_index,
               tree_kws={'colors':'k','linewidths':1},
                )         
g.ax_col_dendrogram.remove()
#g.ax_row_dendrogram.remove() #移除树状图
g.ax_heatmap.set_yticks(np.linspace(0.5,44.5,44))
g.ax_heatmap.set_yticklabels(g.data2d.columns)
g.ax_heatmap.set_xticks(np.linspace(0.5,44.5,44))
g.ax_heatmap.set_xticklabels(g.data2d.columns)
g.ax_heatmap.yaxis.set_tick_params(which='major', 
                                   labelcolor='k',
                                   labelleft=True, labelright=False,
                                   length=0,
                                   pad=10,
                                   )
g.ax_heatmap.xaxis.set_tick_params(which='major',
                                   labelcolor='k',
                                   labeltop=True, labelbottom=False,
                                   length=0,
                                   pad=3,
                                   labelrotation=45
                                   )
g.ax_col_colors.yaxis.set_tick_params(length=0)
g.ax_row_colors.xaxis.set_tick_params(length=0)
plt.show()