# Reconstructing atmospheric O2 using random forests with global magic igneous geochemical data
# No labels during Boring billion
# initiated by Guoxiong Chen, 2020,2,12
# updated 2020,1,10

from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import train_test_split
from scipy.interpolate import make_interp_spline

####imput data
data=pd.read_excel('D:\\Pycharm\\DeeptimeML\\Ign\\data\\Data source\\New_ign1_mafic_ts_final.xlsx','Sheet1',index_col='Time')
data=data.apply(pd.to_numeric, errors='coerce')

predic=data.loc[:,'SIO2':'CS'].values  # input data for prediction
### data nomalization
predic=(predic-np.min(predic,axis=0))/(np.max(predic,axis=0)-np.min(predic,axis=0))
#predic=np.log(predic-np.min(predic,axis=0)+[[1] * 44 for _ in range(79)])
data.loc[:,'SIO2':'CS']=predic
final=np.array(data.index)
imt=pd.DataFrame([],index=data.columns[1::])

Loss=[]
R22=[]
for k in range(10):
    ## training data preparation
    time_index=[(0,500),(2500,4000)]
    s1 = np.random.randint(6,8,1)
    s2 = np.random.randint(10,20,1)
    sample=[s1,s2]
    O2=[0,-7]
    train_data=pd.DataFrame()
    for i in range(2):
        #random sampling
        train_data1=data.iloc[(data.index>time_index[i][0])&(data.index<time_index[i][1]),:].sample(n=sample[i])
        train_data1['O2']=O2[i]
        train_data=pd.concat([train_data,train_data1])
    #noise=np.random.randint(1,10,len(train_data['O2']))*0.2-1
    noise1 = np.random.randint(0, 10, s1) * 0.08 - 0.6 # 0-500 Ma, 20% O2, (1) 0.08 - 0.6; (2) 0.04 - 0.3; (3) 0.13 - 1
    noise2 = np.random.randint(0, 10, s2) * 0.2 - 1    # 2500-4000 Ma,(1) 0.2-1; (2) 0.1 - 0.5 ；(3) 0.4 - 2
    noise = np.hstack((noise1, noise2))
    datay=train_data['O2']+noise # add uncertainty
    datax=train_data.loc[:,'SIO2':'CS']
    '''
    row=[50,100,150,200,250,300,350,400,450,500]
    col=[5,7,9,11,13,15,17,19,21,23,25,27]
    '''
    row = [100, 500]
    col = [5, 7]

    parameters={'n_estimators':row,
            'criterion':['mse'],
            'max_features':col
            }
    params=ParameterGrid(parameters)
    kl=KFold(n_splits=5)
    pre_fit=np.array(datay.index).reshape(len(datay),1)
    for i in params:
        pre_fiti=np.array([])
        for x,y in  kl.split(datax):
            train_x,test_x=datax.values[x,:],datax.values[y,:]
            train_y,test_y=datay.values[x],datay.values[y]
            clf=RandomForestRegressor(criterion=i['criterion'],
                                      n_estimators=i['n_estimators'],
                                      max_features=i['max_features'])
            clf.fit(train_x,train_y)
            pre_fitx=clf.predict(test_x)
            pre_fiti=np.concatenate((pre_fiti,pre_fitx),axis=0)
        pre_fit=np.hstack([pre_fit,pre_fiti.reshape(len(datay),1)])

    pre_fit=pd.DataFrame(pre_fit,columns=[i for i in range(len(pre_fit[0]))])
    pre_fit=pre_fit.set_index(0)

    def r2score(true_value,pre_value):
        y_mean=np.mean(true_value)
        global sse,sst,ei
        ei=pre_value-true_value
        sse=np.sum(ei**2,axis=0)
        sst=np.sum((true_value-y_mean)**2)
        score=1-sse/sst
        #用0替换小于0的值
        #score[score<0]=0
        return score
    r2=r2score(datay.values.reshape(len(datay),1),pre_fit.values) #R方检验
    r_index=np.argmax(r2)
    print('Optimal parameters:',params[r_index])
    print('Optimal R2：',r2[r_index])
    clf=RandomForestRegressor(criterion=params[r_index]['criterion'],
                              n_estimators=params[r_index]['n_estimators'],
                              max_features=params[r_index]['max_features'])
    clf.fit(datax,datay)
    train_pre=clf.predict(datax)
    loss=mean_squared_error(datay,train_pre)
    # r22=r2_score(datay,train_pre)
    Loss.append(loss)
    R22.append(r2[r_index])

    predict=clf.predict(predic)
    importance=clf.feature_importances_
    imt[k]=importance #np.column_stack([imt,importance])
    final=np.column_stack([final,predict])

    plt.figure(1, figsize=(9, 4))
    #fig = plt.figure()
    #ax.grid(True,color='0.6',linestyle=':')
    plt.plot(datax.index,datay,'r.', markersize=4.)
    plt.plot(data.index,predict,'b--', linewidth=0.2)
    plt.legend(['training data','prediction'])
    plt.xlabel('Age (Ma)')
    plt.ylabel('$\mathregular{Log_{10}}$ ($\mathregular{O_2}$) PAL')
    plt.xlim(0, 4000)
    #plt.ylim(-6, 1)

    #plt.gca().invert_xaxis()
#plt.show()
print(np.mean(Loss))
print(np.mean(R22))

## feature importance
imtmean=np.mean(imt,axis=1)
imtmean.columns='mean_importance'
imtmean=imtmean.sort_values(ascending=True)
figure=plt.figure(5,figsize=(8,8))
plt.barh(imtmean.index,imtmean.values.reshape(44)*100)
plt.xlabel('Importance(%)')
plt.yticks(fontsize=9)
plt.ylabel('Elements')
#plt.show()

final=pd.DataFrame(final,columns=[i for i in range(np.shape(final)[1])])
final=final.set_index(0)
final_res=final.set_index(final.index.rename('Time'))
final_mean=np.mean(final_res,axis=1)
final_std=np.std(final_res,axis=1)
final_res['mean']=final_mean
final_res['std']=final_mean
#final_res.to_excel('predict3.xls')

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
#lt.gca().invert_xaxis()
#plt.show()

final_cummean=final.cumsum(axis=1)/[i+1 for i in range(len(final.columns))]
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
mse=[rmse(final_cummean.iloc[:,i],final_cummean.iloc[:,i+1]) for i in range(len(final.columns)-1)]
figure3=plt.figure(3,figsize=(9, 4))
plt.plot(np.arange(1,len(final.columns)),mse,'k-', markersize=0.5)
plt.plot(np.arange(1,len(final.columns)),mse,'r.', markersize=5.)
plt.xlabel('Simulation Number')
plt.ylabel('RMS')
plt.show()

#save to excel
'''
preb=pd.DataFrame(mse,columns=['X'])
write=pd.ExcelWriter('D:/Pycharm/DeeptimeML/Ign/mse2.xls')
preb.to_excel(write,sheet_name='nearmiss')
write.save()
'''

