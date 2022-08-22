# Principal component analysis of global magic igneous geochemical data
# initiated by Guoxiong Chen, 2020,2,12
# updated 2020,1,10

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

### read doc ,index_col='Time'
prmdata=pd.read_excel('D:/Pycharm/DeeptimeML/Ign/PCA/Mafic_NEW_50.xlsx','Sheet1')
data=prmdata.apply(pd.to_numeric, errors='coerce') # using coerce makes character becomes null
datax=data.loc[:,'SiO2':'Cs'].values

### Principal component Analysis
# Data normalization
#datax=np.log2(datax)
#datax=(datax-np.mean(datax,axis=0))/np.std(datax,axis=0)
datax=(2*(datax-np.min(datax,axis=0)))/(np.max(datax,axis=0)-np.min(datax,axis=0))-1
pca=PCA(n_components=2,whiten=False,svd_solver='auto',tol=0.1)
pca1=pca.fit(datax)

pca_var1=pca1.explained_variance_ratio_
pca_var2=pca1.explained_variance_
pca_var=pd.DataFrame([pca_var2,pca_var1],index=['Covariance','Cumulative contribution'],columns=[i+1 for i in range(2)])
pca1.singular_values_
pca1.noise_variance_
pca1.n_features_
pca1.n_samples_
pca_cov=pd.DataFrame(pca.get_covariance())
pca_compo=pd.DataFrame(np.around(pca1.components_.T,3),index=data.columns[1::])
pca_res=pd.DataFrame(pca.fit_transform(datax),index=data.index)
pca_var
np.sum(pca_var,axis=1)
pca.components_
pca.explained_variance_ratio_
pca.explained_variance_

plt.rcParams['font.family']='Times New Roman'
plt.rcParams['font.size']=9
plt.rcParams['savefig.dpi']=300
plt.rcParams['xtick.major.pad']=1
plt.rcParams['ytick.major.pad']=1
plt.rcParams['figure.figsize']=[8,8]
ax1=plt.subplot2grid((10,10),(3,0),rowspan=7,colspan=7)
ax1.scatter(pca_compo.values[:,1],pca_compo.values[:,0],s=5,c='r')
for i in range(len(pca_compo)):
    ax1.text(pca_compo.values[i,1],pca_compo.values[i,0],pca_compo.index[i])
ax1.set_xlabel('PCA2',fontsize=10)
ax1.set_ylabel('PCA1',fontsize=10)
ax2=plt.subplot2grid((10,10),(3,7),rowspan=7,colspan=3)
datasns=pca_compo.iloc[:,0]
index=datasns.sort_values(ascending=False)
sns.barplot(y=index.index, x=index.values,
            color='tomato',
            #alpha=0.5,0000000
            orient='h',
            ax=ax2)
ax2.axvline(0, color="k", clip_on=False,lw=0.5)
ax2.set_ylabel("Elements",fontsize=10)
ax2.set_xlabel('Loading',fontsize=10)
ax3=plt.subplot2grid((10,10),(0,0),rowspan=3,colspan=7)
datasns=pca_compo.iloc[:,1]
index=datasns.sort_values(ascending=False)
sns.barplot(x=index.index, y=index.values,
            color='deepskyblue',
            #alpha=0.5,
            orient='v',
            ax=ax3)
plt.xticks(np.arange(44),index.index,rotation=45)
ax3.axhline(0, color="k", clip_on=False,lw=0.5)
ax3.set_ylabel("Loading",fontsize=10)
ax3.set_xlabel('Elements',fontsize=10)
plt.subplots_adjust(wspace=1.5, hspace=3.5)
ax4=plt.subplot2grid((10,10),(0,7),rowspan=3,colspan=3)
precent=np.round(np.array([pca_var1[0],pca_var1[1],1-np.sum(pca_var1)])*100,2)
color=['tomato','deepskyblue','#E0FBFC']
labels=['PCA1','PCA2','Others']
explode=[0.05,0.02,0]
ax4.pie(precent,
        explode=explode,
        labels=labels,
        colors=color,
        startangle=120,
        shadow=True,
        radius=1.5,
        labeldistance=1.2,
        autopct='%1.1f%%')
plt.show()

dataxy=prmdata.iloc[:,0]
pcadata=pd.concat([dataxy,pca_res],axis=1,join='inner')
pcadata[pcadata.isnull().values==True]

#print(pcadata)

plt.rc("font",size=14) # fontsize
plt.rcParams["font.sans-serif"]="Arial" # typeface
plt.rcParams["axes.unicode_minus"]=False
plt.plot(np.array(data.iloc[:,0]), 1*np.array(pca_res.iloc[:,0]), 'b-', linewidth=1)
plt.xlabel('Age(Ma)')
plt.ylabel('PCA1')
plt.show()

# save result
write=pd.ExcelWriter('D:/Pycharm/DeeptimeML/Ign/PCA/PCA_R.xlsx')
pcadata.to_excel(write,sheet_name='PCA result')
pca_var.to_excel(write,sheet_name='Covariance')
write.save()
