%% clear all 
clc
clear all
%% imput data
[data,txt]=xlsread('D:\Pycharm\DeeptimeML\Ign\data\O2_NEW.xlsx','Sheet1'); % ѵ������
data1=data(:,3:47);
%% data nomarlization
mpdata=mapminmax(data1',0,1);
% mpdata=log(data1'-min(data1',2)+1);
bpdata=mpdata';
n=minmax(mpdata);
%% SOM clustering
net=newsom(n,4,'hextop','linkdist'); %����������;����ṹ������hextop��gridtop��randtop��
...���뺯����boxdist��linkdist  ��ʼ������
w1=net.iw{1}; %��ʼȨֵ
net.trainparam.epochs=100; %��������
net.trainParam.lr=10;%ѧϰ����
net=train(net,mpdata); %ѵ������
w2=net.iw{1,1};  %ѵ����Ȩֵ 
y=sim(net,mpdata); %����ѵ��
Y=vec2ind(y); %������
SOM_re=[data(:,2),Y']; %%������
SOMw=w2'; % �������ľ���
plot(SOM_re(:,1),5-SOM_re(:,2),'-')

%% BP������
% k=35;
% trainx=bpdata(1:k,:);
% trainy=data(1:k,1);
% yuce=bpdata;
% net=newff(minmax(bpdata'),[50,1],{'logsig' 'logsig'},'trainbfg');% BP���紴��
% % ����ѵ������
% net.trainParam.show=50; %������ʾ֮���ѵ������
% net.trainParam.epochs=2000; %ѵ������
% net.trainParam.goal=0.01;%Ŀ�꾫��
% net.trainParam.lr=0.01; %ѧϰ����
% net=train(net,trainx',trainy'); %����ѵ�� �������ݶ���������
% y=sim(net,yuce')'; %Ԥ�����ѵ����
% result=[data(:,2),round(y)];
% plot(data(1:k,2),data(1:k,1),'r-') 
% hold on
% plot(data(:,2),result(:,2),'b--')
% legend('original','prediction')
%xlswrite('result.xls',result);
