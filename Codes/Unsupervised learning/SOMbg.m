clc; clear; close all;
%% Input data
% [FileName,PathName]=uigetfile({'*.csv';'*.txt';'*.dat'},'Open data','E:\');%
% filepath = fullfile(PathName, FileName);
filepath = 'D:\Pycharm\DeeptimeML\Ign\SOM\O2_NEW_SOM2.xlsx';
rawdata = importdata(filepath);
[s,v] = listdlg('ListString',rawdata.textdata,'SelectionMode','multiple') 
dataselect = rawdata.data(:, s);
names=rawdata.textdata(:, s);
sD = som_data_struct(dataselect,'name','JN','comp_names',names);
location='D:\Pycharm\DeeptimeML\Ign\SOM\';% save the outputs to this path

%% Data Normalization
[sel ok]=listdlg('ListString',{'var','range','log','logistic'},...
    'Name','Chose','OKString','OK','CancelString','Cancel','SelectionMode','single','ListSize',[180 80]);
switch sel
    case 1
      sD = som_normalize(sD,'var')  
    case 2
      sD = som_normalize(sD,'range') 
    case 3
      sD = som_normalize(sD,'log') 
    case 4
      sD = som_normalize(sD,'logistic') 
end
% sD = som_denormalize(sD);
[sel ok]=listdlg('ListString',{'hexa','rect'},...
    'Name','Chose','OKString','OK','CancelString','Cancel','SelectionMode','single','ListSize',[180 80]);
switch sel
    case 1
        sM = som_make(sD,'lattice', 'hexa','mapsize','big'); %sM:map struct.'mask',[1;3;2]
    case 2
        sM = som_make(sD,'lattice', 'rect','mapsize','small'); %sM:map struct.
end


%% Plots
figure(1);colormap(jet)
som_show(sM,'umat','all');
saveName=inputdlg('Input the name','name',[1 30],{'Umat'});
saveas(gcf,[location,saveName{1}],'jpeg');

figure(2);colormap(jet);set(gcf,'Position',get(0,'ScreenSize'))
som_show(sM,'norm','d');
saveName=inputdlg('Input the name','name',[1 30],{'≥£πÊœ‘ æ'});
saveas(gcf,[location,saveName{1}],'jpeg');

%% Kmeans and DB methods for determining optimal clusters
[c,p,err,ind] = kmeans_clusters(sM,8,100); % find at most 10 clusters
figure(3);%set(gcf,'Position',get(0,'ScreenSize'))
plot(1:length(ind),ind,'x-')
[dummy,i] = min(ind)
cl = p{i};
saveName=inputdlg('Input the name','name',[1 30],{'DB'});
saveas(gcf,[location,saveName{1}],'jpeg');
% pause

%% Clustering
[dummy,i] = min(ind); % select the one with smallest index
i=4; % you can chose the value adaptively
figure(4); %set(gcf,'Position',get(0,'ScreenSize'))
som_show(sM,'color',{p{i},sprintf('%d clusters',i)}); % visualize   som_show(sM,'umat',{p{i},sprintf('%d clusters',i)});
colormap(jet(i)), som_recolorbar, % change colormap
saveName=inputdlg('Input the name','name',[1 30],{'clusters'});
saveas(gcf,[location,saveName{1}],'jpeg');

%% Save the result to excel
Bmus = som_bmus(sM, sD);
sD = som_label(sD,'add',[1:1:size(sD.data,1)],num2str(p{i}(Bmus)));
test=cell2mat(sD.labels);class=str2num(test);
result=[rawdata.data,class];
figure(5)
plot(rawdata.data(:,1),class,'o-')';
saveName=inputdlg('Input the name','name',[1 30],{'newClass'});
rawdata.textdata(size(rawdata.textdata,2)+1)=cellstr(saveName);

xlswrite(strcat(location,saveName{1}),rawdata.textdata,'Sheet1','A1');
xlswrite(strcat(location,saveName{1}),result,'Sheet1','A2');
sD = som_label(sD,'clear','all');













