%% clear all 
clc
clear all
%% imput data
[data,txt]=xlsread('D:\Pycharm\DeeptimeML\Ign\data\O2_NEW.xlsx','Sheet1'); % 训练数据
data1=data(:,3:47);
%% data nomarlization
mpdata=mapminmax(data1',0,1);
% mpdata=log(data1'-min(data1',2)+1);
bpdata=mpdata';
n=minmax(mpdata);
%% SOM clustering
net=newsom(n,4,'hextop','linkdist'); %创建神经网络;网络结构函数：hextop，gridtop，randtop；
...距离函数：boxdist，linkdist  初始分类数
w1=net.iw{1}; %初始权值
net.trainparam.epochs=100; %迭代次数
net.trainParam.lr=10;%学习速率
net=train(net,mpdata); %训练函数
w2=net.iw{1,1};  %训练后权值 
y=sim(net,mpdata); %仿真训练
Y=vec2ind(y); %分类结果
SOM_re=[data(:,2),Y']; %%分类结果
SOMw=w2'; % 聚类中心矩阵
plot(SOM_re(:,1),5-SOM_re(:,2),'-')

%% BP神经网络
% k=35;
% trainx=bpdata(1:k,:);
% trainy=data(1:k,1);
% yuce=bpdata;
% net=newff(minmax(bpdata'),[50,1],{'logsig' 'logsig'},'trainbfg');% BP网络创建
% % 设置训练参数
% net.trainParam.show=50; %两次显示之间的训练次数
% net.trainParam.epochs=2000; %训练次数
% net.trainParam.goal=0.01;%目标精度
% net.trainParam.lr=0.01; %学习速率
% net=train(net,trainx',trainy'); %进行训练 测试数据都是行向量
% y=sim(net,yuce')'; %预测测试训练集
% result=[data(:,2),round(y)];
% plot(data(1:k,2),data(1:k,1),'r-') 
% hold on
% plot(data(:,2),result(:,2),'b--')
% legend('original','prediction')
%xlswrite('result.xls',result);
