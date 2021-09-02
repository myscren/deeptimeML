# No labels during Boring billion
from sklearn.svm import SVR
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
###导入数据,导出训练数据，及预测数据
data=pd.read_excel('E:/Manuscripts/陈国雄稿件2021/data analysis/SVR/O2_NEW.xlsx','Sheet1',index_col='Time')
data=data.apply(pd.to_numeric, errors='coerce') #coerce为字符变为空值

predic=data.loc[:,'SiO2':'Ga'].values  #待预测数据
### 数据标准化
predic=(predic-np.min(predic,axis=0))/(np.max(predic,axis=0)-np.min(predic,axis=0))
#predic=np.log(predic-np.min(predic,axis=0)+[[1] * 45 for _ in range(79)])
data.loc[:,'SiO2':'Ga']=predic
final=np.array(data.index)

for k in range(1000):
    ##训练数据选取
    time_index=[(0,500),(2500,4000)]
    s1 = np.random.randint(6,8,1) # random labels
    s2 = np.random.randint(7,20,1) # random labels
    sample=[s1,s2]
    O2=[0,-7]
    train_data=pd.DataFrame()
    for i in range(2):
        #随机取样
        train_data1=data.iloc[(data.index>time_index[i][0])&(data.index<time_index[i][1]),:].sample(n=sample[i])
        train_data1['O2']=O2[i]
        train_data=pd.concat([train_data,train_data1])
    #noise=np.random.randint(1,10,len(train_data['O2']))*0.2-1
    noise1 = np.random.randint(0, 10, s1) * 0.08 - 0.6 # 0-500 Ma, 20% O2, (1) 0.08 - 0.6; (2) 0.04 - 0.3; (3) 0.13 - 1
    noise2 = np.random.randint(0, 10, s2) * 0.2 - 1    # 2500-4000 Ma,(1) 0.2-1; (2) 0.1 - 0.5 ；(3) 0.4 - 2
    noise = np.hstack((noise1, noise2))
    datay=train_data['O2']+noise #生成噪声
    datax=train_data.loc[:,'SiO2':'Ga']

    #row=[5,10,15,20,25,30,35,40,45,50]
    #col=[0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01]
    row = [0.1, 1, 10, 100, 1000, 10000, 100000]
    col = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
    kernel=['rbf']
    parameters={'C':row,
                'kernel':kernel,
                'gamma':col
                }

    params=ParameterGrid(parameters)
    kl=KFold(n_splits=5)
    pre_fit=np.array(datay.index).reshape(len(datay),1)
    for i in params:
        pre_fiti=np.array([])
        for x,y in  kl.split(datax):
            train_x,test_x=datax.values[x,:],datax.values[y,:]
            train_y,test_y=datay.values[x],datay.values[y]
            clf=SVR(kernel=i['kernel'],C=i['C'],gamma=i['gamma'])
            clf.fit(train_x,train_y)
            pre_fitx=clf.predict(test_x)
            pre_fiti=np.concatenate((pre_fiti,pre_fitx),axis=0)
        pre_fit=np.hstack([pre_fit,pre_fiti.reshape(len(datay),1)])

    pre_fit=pd.DataFrame(pre_fit,columns=[i for i in range(len(row)*len(col)+1)])
    pre_fit=pre_fit.set_index(0)

    def r2score(true_value,pre_value):
        y_mean=np.mean(true_value)
        global sse,sst,ei 
        ei=pre_value-true_value
        sse=np.sum(ei**2,axis=0)
        sst=np.sum((true_value-y_mean)**2)
        score=1-sse/sst
        #用0替换小于0的值
        score[score<0]=0
        return score
    r2=r2score(datay.values.reshape(len(datay),1),pre_fit.values) #R方检验
    r_index=np.argmax(r2)
    print('最优参数：',params[r_index])
    print('最优得分R2：',r2[r_index])
    clf=SVR(kernel=params[r_index]['kernel'],
            C=params[r_index]['C'],
            gamma=params[r_index]['gamma'])
    clf.fit(datax,datay)
    predict=clf.predict(predic)

    final=np.column_stack([final,predict])

    plt.figure(1, figsize=(9, 4))
    #fig = plt.figure()
    #ax.grid(True,color='0.6',linestyle=':')
    plt.plot(datax.index,datay,'r.', markersize=5.)
    plt.plot(data.index,predict,'b--', linewidth=0.2)
    plt.legend(['training labels','prediction'],fontsize=12)
    plt.xlabel('Age (Ma)',fontsize=12)
    plt.ylabel('$\mathregular{Log_{10}}$ ($\mathregular{O_2}$) PAL',fontsize=12)
    plt.xlim(0, 4000)
    plt.ylim(-9, 1)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    #plt.ylim(-6, 1)

    #plt.gca().invert_xaxis()
#plt.show()


lencol=len(col)
lenrow=len(row)
fig=plt.figure(4,figsize=(6,6))
ax = fig.add_subplot()
imsh=np.flipud(r2.reshape(lenrow,lencol))
im = ax.imshow(imsh,aspect='equal',extent=(-0.5,-0.5+lencol,-0.5,-0.5+lenrow),cmap = plt.cm.rainbow)
#cb=plt.colorbar(im)
#cb.ax.tick_params(labelsize=16)
ax.set_xticks(np.arange(lencol))
ax.set_yticks(np.arange(lenrow))
# ... and label them with the respective list entries
col=np.log10(col)
row=np.log10(row)
ax.set_xticklabels(col)
ax.set_yticklabels(row)
ax.set_xlabel('$\log_{10}$($\gamma$)',fontsize=16)
ax.set_ylabel('$\log_{10}$(C)',fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax)
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
#plt.plot(final_mean.index,final_mean.values,'k-', linewidth=1.5)
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

#写入excel
'''
preb=pd.DataFrame(mse,columns=['X'])
write=pd.ExcelWriter('E:/Manuscripts/陈国雄稿件2021/data analysis/mse2.xls')
preb.to_excel(write,sheet_name='nearmiss')
write.save()
'''
