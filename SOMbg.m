clc; clear; close all;
% [FileName,PathName]=uigetfile({'*.csv';'*.txt';'*.dat'},'��txt����dat�ļ�����','E:\');%�ҵ���������λ��
% filepath = fullfile(PathName, FileName);
filepath = 'D:\Pycharm\DeeptimeML\Ign\SOM\O2_NEW_SOM2.xlsx';
rawdata = importdata(filepath);
[s,v] = listdlg('ListString',rawdata.textdata,'SelectionMode','multiple')%ѡȡ�Լ���Ҫ�ı�����
dataselect = rawdata.data(:, s);
names=rawdata.textdata(:, s);
sD = som_data_struct(dataselect,'name','JN','comp_names',names);
location='E:\Manuscripts\�¹��۸��2021\data analysis\SOM\';%�ƶ��洢λ��

%-��һ����ѡ�����˽ṹ--------------------------------------------------------------------
[sel ok]=listdlg('ListString',{'var','range','log','logistic'},...
    'Name','ѡ��һ��','OKString','ȷ��','CancelString','ȡ��','SelectionMode','single','ListSize',[180 80]);
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
    'Name','ѡ��һ��','OKString','ȷ��','CancelString','ȡ��','SelectionMode','single','ListSize',[180 80]);
switch sel
    case 1
        sM = som_make(sD,'lattice', 'hexa','mapsize','big'); %sM:map struct.'mask',[1;3;2]
    case 2
        sM = som_make(sD,'lattice', 'rect','mapsize','small'); %sM:map struct.
end


%7===============================7=========================7===============
figure;colormap(jet)
som_show(sM,'umat','all');
saveName=inputdlg('�����뱣������','name',[1 30],{'Umat'});
saveas(gcf,[location,saveName{1}],'jpeg');

figure;colormap(jet);set(gcf,'Position',get(0,'ScreenSize'))
som_show(sM,'norm','d');
saveName=inputdlg('�����뱣������','name',[1 30],{'������ʾ'});
saveas(gcf,[location,saveName{1}],'jpeg');

%14����k��ֵ���з��࣬DB�õ���Ѿ�����=============================14================================14=======
[c,p,err,ind] = kmeans_clusters(sM,8,100); % find at most 10 clusters
figure;%set(gcf,'Position',get(0,'ScreenSize'))
plot(1:length(ind),ind,'x-')
[dummy,i] = min(ind)
cl = p{i};
saveName=inputdlg('�����뱣������','name',[1 30],{'DB'});
saveas(gcf,[location,saveName{1}],'jpeg');
% pause
%15=============================15==============================15========
[dummy,i] = min(ind); % select the one with smallest index
i=4;%����ָ���Լ���Ҫ�ľ�����3
figure;%set(gcf,'Position',get(0,'ScreenSize'))
som_show(sM,'color',{p{i},sprintf('%d clusters',i)}); % visualize   som_show(sM,'umat',{p{i},sprintf('%d clusters',i)});
colormap(jet(i)), som_recolorbar % change colormap
saveName=inputdlg('�����뱣������','name',[1 30],{'clusters'});
saveas(gcf,[location,saveName{1}],'jpeg');
%16=�����ǩ��excel���===========================16==========================16============
Bmus = som_bmus(sM, sD);

sD = som_label(sD,'add',[1:1:size(sD.data,1)],num2str(p{i}(Bmus)));
test=cell2mat(sD.labels);class=str2num(test);
result=[rawdata.data,class];
plot(rawdata.data(:,1),class,'o-')';
saveName=inputdlg('�����뱣������','name',[1 30],{'newClass'});
rawdata.textdata(size(rawdata.textdata,2)+1)=cellstr(saveName);

xlswrite(strcat(location,saveName{1}),rawdata.textdata,'Sheet1','A1');
xlswrite(strcat(location,saveName{1}),result,'Sheet1','A2');
sD = som_label(sD,'clear','all');%ȥ��֮ǰ���ɵı�ǩ













