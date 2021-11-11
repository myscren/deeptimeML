% This program is to implment the self-orgnaized mapping (SOM) analysis 
% of igneous geochemical composition time-series data. 
% Begin initialization code
clc;clear all;
%% imput data
[filename1 filepath1] = uigetfile({'*.xlsx';'*.xls'},'Select grd file');
input_path = strcat(filepath1,filename1);
data =  xlsread (input_path);
data1 = data(:,3:47);
%% data nomarlization
mpdata  =mapminmax(data1',0,1);
% mpdata=log(data1'-min(data1',2)+1);
bpdata = mpdata';
n = minmax(mpdata);
%% SOM analysis
% creat network;
% structural function：hextop，gridtop，randtop；
% distance function：boxdist，linkdist
net=newsom(n,4,'hextop','linkdist'); 
%% parameterization
w1=net.iw{1}; % initial value
net.trainparam.epochs=100; % iterations
net.trainParam.lr=10; % learning rate
net=train(net,mpdata); % data training
w2=net.iw{1,1};  % weights
y=sim(net,mpdata); % prediction
Y=vec2ind(y); % classification
SOM_re=[data(:,2),Y']; %% time-series result
SOMw=w2'; % Clustering center matrix
%% Figure
figure(1)
plot(SOM_re(:,1),5-SOM_re(:,2),'r-')
xlabel('Age(Ma)','FontSize',10);
ylabel('Class','FontSize',10);
ylim([0,5])

%% BP neural network prediciton  
% k=79;
% trainx=bpdata(1:k,:);
% trainy=data(1:k,1);
% trainy=trainy.*rand(1,79)';
% yuce=bpdata;
% net=newff(trainx',trainy',[50,1],{'logsig' 'logsig'},'trainbfg');% BP网络创建
% % 设置训练参数
% net.trainParam.show=10; %两次显示之间的训练次数
% net.trainParam.epochs=100; %训练次数
% net.trainParam.goal=0.01;%目标精度
% net.trainParam.lr=0.01; %学习速率
% net=train(net,trainx',trainy'); %进行训练 测试数据都是行向量
% y=sim(net,yuce')'; %预测测试训练集
% result=[data(:,2),y];
% plot(data(1:k,2),data(1:k,1),'r*') 
% hold on
% plot(data(:,2),result(:,2),'b--')
% legend('original','prediction')
% xlswrite('result.xls',result);
