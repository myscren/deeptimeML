# Reconstructing atmospheric O2 using artifical neural network with global magic igneous geochemical data
# No labels during Boring billion
# initiated by Guoxiong Chen, 2020,2,12
# updated 2020,1,10

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.utils import shuffle
from sklearn.metrics import r2_score
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import train_test_split
from scipy.interpolate import make_interp_spline
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error,r2_score


def train(file_path,models,epochs):
    
    data=pd.read_excel(file_path,'Sheet1',index_col='Time')
    data=data.apply(pd.to_numeric, errors='coerce')

    predic=data.loc[:,'SIO2':'CS'].values  # input data for prediction
    ### data nomalization
    predic=(predic-np.min(predic,axis=0))/(np.max(predic,axis=0)-np.min(predic,axis=0))
    #predic=np.log(predic-np.min(predic,axis=0)+[[1] * 45 for _ in range(79)])
    data.loc[:,'SIO2':'CS']=predic
    final = data.index

#     Loss = np.zeros(shape=(models,epochs))
    pre_fit = np.zeros(shape=(models,len(final)))
    Loss = pd.DataFrame()
    r2 = []
    for k in range(models):
        ## training data preparation
        time_index=[(0,500),(2500,3850)]
        s1 = np.random.randint(6,8,1)
        s2 = np.random.randint(10,20,1)
        sample=[s1,s2]
        O2=[0,-7]
        train_data=pd.DataFrame()
        for i in range(2):
            # random sampling
            train_data1=data.iloc[(data.index>time_index[i][0])&(data.index<time_index[i][1]),:].sample(n=sample[i])
            train_data1['O2']=O2[i]
            train_data=pd.concat([train_data,train_data1])
        #noise=np.random.randint(1,10,len(train_data['O2']))*0.2-1
        noise1 = np.random.randint(0, 10, s1) * 0.08 - 0.6 # 0-500 Ma, 20% O2, (1) 0.08 - 0.6; (2) 0.04 - 0.3; (3) 0.13 - 1
        noise2 = np.random.randint(0, 10, s2) * 0.2 - 1    # 2500-4000 Ma,(1) 0.2-1; (2) 0.1 - 0.5 ；(3) 0.4 - 2
        noise = np.hstack((noise1, noise2))
        datay=train_data['O2']+noise # add uncertainty
        datax=train_data.loc[:,'SIO2':'CS']

        kl=KFold(n_splits=5)
        pre_fiti = pd.DataFrame()
        true_y = pd.DataFrame()
        for x,y in  kl.split(datax):
            train_x,valid_x=datax.values[x,:],datax.values[y,:]
            train_y,valid_y=datay.values[x],datay.values[y]
            # building the network of three layers (128,64,32)
            mlp = MLPRegressor(hidden_layer_sizes = (256,128,64,32),activation='tanh',
                               solver='adam',alpha=0.0001,max_iter=epochs,
                               random_state=2022,early_stopping=False,tol=0.0001)
            mlp.fit(train_x,train_y)
            # test dataset R2
            pre_fit_y_valid = pd.DataFrame(mlp.predict(valid_x)) 
            
            pre_fiti = pre_fiti.append(pre_fit_y_valid)
            true_y = true_y.append(pd.DataFrame(valid_y))

            
        valid_r2 =  r2_score(true_y,pre_fiti)
        print("number of models",k,"testd_r2=",valid_r2)
        
        r2.append(valid_r2)
        pre_fitx = mlp.predict(predic)
        pre_fit[k] = pre_fitx
        loss = np.array(mlp.loss_curve_)
        loss = pd.DataFrame(loss)
        Loss = Loss.append(loss.T)
        
    r2 = np.array(r2)    
    r2[r2<0]=0
    valid_r2_mean = np.mean(np.array(r2))
    print("mean r2",valid_r2_mean)
    return Loss,pre_fit,final

def plot(final,pre_fit):
    
    final = pd.DataFrame(final)
    pre_fit = pd.DataFrame(pre_fit)
    final_data = pd.concat([final,pre_fit.T],axis=1)
    final_res = final_data.set_index('Time')
    final_mean=np.mean(final_res,axis=1)
    final_std=np.std(final_res,axis=1)
    final_res['mean']=final_mean
    final_res['std']=final_std

    figure2=plt.figure(2,figsize=(9, 4))

    xnew = np.linspace(final_mean.index.min(), final_mean.index.max(), 300)  # 300 represents number of points to make between T.min and T.max

    power_smooth = make_interp_spline(final_mean.index, final_mean.values)(xnew)
    plt.plot(xnew,power_smooth,'b-', linewidth=1.5)
    plt.errorbar(final_mean.index,final_mean.values,final_std.values,fmt='ro',
                ecolor='lightskyblue',
                elinewidth = 10,
                ms=3,
                mfc='red',
                mec='red')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Age (Ma)',fontsize=12)
    plt.ylabel('$\mathregular{Log_{10}}$ ($\mathregular{O_2}$) PAL',fontsize=12)
    plt.xlim(0, 4000)
    plt.ylim(-8.5, 0.5)
    plt.savefig('MLP.jpg',dpi=300)
    plt.show()
    
def loss_plot(Loss):
    Loss_res = pd.DataFrame(Loss.T)
    loss_mean=np.mean(Loss_res,axis=1)
    loss_std=np.std(Loss_res,axis=1)
    Loss_res['mean']=loss_mean
    Loss_res['std']=loss_std
#     print(Loss_res)

    figure=plt.figure(2,figsize=(9, 4))

    xnew = np.linspace(loss_mean.index.min(), loss_mean.index.max(), 100)  # 300 represents number of points to make between T.min and T.max

    power_smooth = make_interp_spline(loss_mean.index, loss_mean.values)(xnew)
    plt.plot(xnew,power_smooth,'b-', linewidth=1.5)
    plt.errorbar(loss_mean.index,loss_mean.values,loss_std.values,fmt='ro',
                ecolor='lightskyblue',
                elinewidth = 10,
                ms=3,
                mfc='red',
                mec='red')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Epochs',fontsize=12)
    plt.ylabel('Loss',fontsize=12)
    plt.xlim(0, Loss_res.shape[0])
    plt.savefig('Loss.jpg',dpi=300)
    plt.show()


models = 100
epochs = 500
file_path = 'D:\\Pycharm\\DeeptimeML\\Ign\\data\\Data source\\New_ign1_mafic_ts_final.xlsx'
Loss,pre_fit,final = train(file_path,models,epochs)
plot(final,pre_fit)
loss_plot(Loss)


