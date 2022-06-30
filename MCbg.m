% MCbg.m
% Updated from Montecarlo.m (of Keller)
% Run a full monte carlo simulation
% Each actual simulation task is conducted by mctask.m, this code provides
% the data needed for mctask to work and stores the results Any variables 
% needed during the simulation must exist in simitemsin; these variables
% will be read into data, filtered, and passed to mctask.
clear all
%% Load required variables
[dataraw,name]=xlsread('D:\Pycharm\DeeptimeML\Ign\data\Data source\New_ign1_mafic_final.xlsx','Sheet1');
[NN,nameT]=xlsread('D:\Pycharm\DeeptimeML\Ign\data\Data source\New_ign1_mafic_final.xlsx','Name');
% load newign;
titlename=name(1,:); 
% Simitems is a cell array holding the names of all the variables to
% examine. Names must be formatted as in ign.elements
% simitemsin={'SIO2';'TIO2';'AL2O3';'FEO';'MGO';'CAO';'MNO';'NA2O';'K2O';'P2O5';'CR';'CO';...
%      'NI';'LA';'CE';'PR';'ND';'SM';'EU';'GD';'TB';'DY';'HO';'ER';'TM';'YB';'LU';'Y';'SC';'V';...
%      'ZR';'HF';'RB';'SR';'BA';'NB';'TA';'U';'TH';'S';'CU';'PB';'ZN';'BE';'GA'};

simitemsin=nameT;
 for i=1:length(simitemsin)
    XP{i}=find(strcmp(titlename,simitemsin{i}));
 end
% See what the age of each sample is; this will be our independent variable
age_uncert_min = 100; % Set minimum age uncertainty (analagous to kernel width)
ages = dataraw(:,5);
age_uncert = dataraw(:,4)-dataraw(:,3);
% Calcualte the relative age range (RAR)
RAR=age_uncert./ages*100;
RAR(isinf(RAR))=0;

age_uncert(age_uncert<age_uncert_min) = age_uncert_min;

% Construct a matrix holding all the data to be used in the simulation
uncert=zeros(1,length(simitemsin)+3);
data_f=zeros(length(ages),length(simitemsin)+3);
data_f(:,1) = age_uncert; % Independent variable uncert goes in position one
data_f(:,2) = ages; % Independent variable values go in position two
data_f(:,3) = RAR; % Independent variable RAR go in position three
%% sort the data by ages
N=find(ages==540);if length(N)>1 N=N(1);end
M=find(ages==2500);if length(M)>1 M=N(1);end
%% creat histogram plot
i=length(simitemsin);
while i>0
    figure
    data_Phan=dataraw(1:N,XP{i});
    data_Prec=dataraw(N+1:end,XP{i});
    h1 = histogram(log(data_Phan),50); hold on
    h2 = histogram(log(data_Prec),50);
    ylabel('Count')
    xlabel(strcat(simitemsin{i},' (log(ppm))'))
    set (gcf,'Position',[400,400,350,250], 'color','w');
    i=i-1;
end
%% delete outliers in igneous geochemistry data
for i=1:length(simitemsin)
    A=find(strcmp({'SIO2','MGO','CAO','AL2O3','NAO','FEOT'},simitemsin{i}));
    if length(A) 
        %% Phanerozoic era
        data1=dataraw(1:N,XP{i});
        data1(find(data1<=0))=NaN;
        % delete outliers using mean¡À2std
        T1=nanmean(data1)+3*nanstd(data1);
        T2=nanmean(data1)-3*nanstd(data1);
        data1(find(data1>T1 | data1<T2))=NaN;
        %% Precambrian era
        data2=dataraw(N+1:M,XP{i});
        data2(find(data2<=0))=NaN;
        % delete outliers using mean¡À2std
        T3=nanmean(data2)+3*nanstd(data2);
        T4=nanmean(data2)-3*nanstd(data2);
        data2(find(data2>T3 | data2<T4))=NaN;
        %% Archean era
        data3=dataraw(M+1:end,XP{i});
        data3(find(data3<=0))=NaN;
        % delete outliers using mean¡À2std
        T5=nanmean(data3)+3*nanstd(data3);
        T6=nanmean(log10(data3))-3*nanstd(data3);
        data3(find(data3>T5 | data3<T6))=NaN;
        %% combine
        data_f(:,i+3)=[data1;data2;data3];
        uncert(i+3)=0.01; %ign.err.(simitemsin{i});
    else 
        %% Phanerozoic era
        data1=dataraw(1:N,XP{i});
        data1(find(data1<=0))=NaN;
        % delete outliers using mean¡À2std
        T1=nanmean(log10(data1))+3*nanstd(log10(data1));
        T2=nanmean(log10(data1))-3*nanstd(log10(data1));
        data1(find(data1>10^T1 | data1<10^T2))=NaN;
        %% Proterozoic era
        data2=dataraw(N+1:M,XP{i});
        data2(find(data2<=0))=NaN;
        % delete outliers using mean¡À2std
        T3=nanmean(log10(data2))+3*nanstd(log10(data2));
        T4=nanmean(log10(data2))-3*nanstd(log10(data2));
        data2(find(data2>10^T3 | data2<10^T4))=NaN;
        %% Archean era
        data3=dataraw(M+1:end,XP{i});
        data3(find(data3<=0))=NaN;
        % delete outliers using mean¡À2std
        T5=nanmean(log10(data3))+3*nanstd(log10(data3));
        T6=nanmean(log10(data3))-3*nanstd(log10(data3));
        data3(find(data3>10^T5 | data3<10^T6))=NaN;
        %% combine
        data_f(:,i+3)=[data1;data2;data3];
        uncert(i+3)=0.01; %ign.err.(simitemsin{i});

    end
end

%% Produce sample weights for bootstrap resamplingaaa

% Range of silica and age values to examine
simin=43;
simax=51;
agemin=0;
agemax=4000;

% filtering defined by relative age range (RAR)
RAR_f=50;

% Reject data that is out of the range of interest, is all NANs, or isn't from a contienent
% ign.MgO>simin &ign.MgO<simax 
% test=(ign.SiO2>simin &ign.SiO2<simax &ign.Age>agemin&ign.Age<agemax &~any(isnan(data(:,1:2)),2) &any(~isnan(data(:,4:end)),2) &ign.Elevation>-100 &~ign.oibs); 
% test=(data(:,3)>simin &data(:,3)<simax & ages>agemin & ages<agemax & ~any(isnan(data(:,1:2)),2));
% test=(data_f(:,2)>agemin & data_f(:,2)<agemax & ~any(isnan(data_f(:,1:2)),2) & data_f(:,3)<RAR_f);
test=(data_f(:,2)>agemin & data_f(:,2)<agemax & ~any(isnan(data_f(:,1:2)),2));
data=data_f(test,:);

% Compute weighting coefficients
% k=invweight(ign.Latitude(test),ign.Longitude(test),ign.Age(test));
k=invweight(dataraw(test,1),dataraw(test,2),dataraw(test,5));

% Compute probability keeping a given data point when sampling
prob=1./((k.*median(5./k))+1); % Keep rougly one-fith of the data in each resampling

% What to call the simulation results
simtitle=sprintf('%.2g-%.2g%% SiO_2',simin, simax);
savetitle=sprintf('mc%g%g',simin,simax);

%% Run the monte carlo simulation and bootstrap resampling

tic;

% Number of simulations
nsims=10000;

% Number of variables to run the simulation for
ndata=length(simitemsin);

% Number of age divisions
nbins=80;

% Edges of age divisions
binedges=linspace(agemin,agemax,nbins+1)';

% Create 3-dimensional variables to hold the results
simaverages=zeros(nsims,nbins,ndata);
simerrors=zeros(nsims,nbins,ndata);
simratio=zeros(nsims,1);


% Run the simulation in parallel. Running the simulation task as a function
% avoids problems with indexing in parallelized code.
gcp; %Start a parellel processing pool if there isn't one already
parfor i=1:nsims
    
    % mctask does all the hard work; simaverages and simerrors hold the results
    [simaverages(i,:,:), simerrors(i,:,:), simratio(i)]=mctask(data,prob,uncert,binedges,nbins);
        
end
% Names of output variables
simitemsout = simitemsin; % This would not be true if we were to perform calculations (e.g. ratios, etc) on the input variables in mctest

% Adjust uncertaintites such that as nsims approaches infinity, the
% uncertainties approach sigma/sqrt(N) where N is the number of original
% real data points.
simerrors=simerrors.*(sqrt(nanmean(simratio))+1/sqrt(nsims));

toc % Record time taken to run simulation

% Compute vector of age-bin centers for plotting
bincenters=linspace(0+(agemax-agemin)/2/nbins,agemax-(agemax-agemin)/2/nbins,nbins)';



%% Plot the results

i=length(simitemsout);
while i>0
    figure    
    plot(data(:,2),data(:,3+i),'.');    hold on
    errorbar(bincenters(1:end-1),nanmean(simaverages(:,1:end-1,i)),2.*nanmean(simerrors(:,1:end-1,i)),'.r')
    xlabel('Age (Ma)')
    ylabel(simitemsout{i})
    xlim([0,4000])
    ylim([min(data(:,3+i)),max(data(:,3+i))])
    set(gca,'tickdir','out')
%     title(simtitle)
%     formatfigure
    i=i-1;
end



%% Print results to csv files

i=length(simitemsout);
averages = NaN(nbins-1,length(simitemsout)+1);
uncertainties = NaN(nbins-1,length(simitemsout)+1);
averages(:,1)=bincenters(1:nbins-1);
uncertainties(:,1)=bincenters(1:nbins-1);
for i=1:length(simitemsout)
    averages(:,i+1) = nanmean(simaverages(:,1:end-1,i))';
    uncertainties(:,i+1) = nanmean(simerrors(:,1:end-1,i))';
end
exportmatrix(averages,'averages50_basalt.csv',',');
exportmatrix(uncertainties,'uncertainties50_basalt.csv',',');

%% Save results

% eval([savetitle '.bincenters=bincenters;']); eval([savetitle '.simaverages=simaverages;']); eval([savetitle '.simerrors=simerrors;']); eval([savetitle '.simitems=printnames;']);
% for i=1:length(printnames)
%     eval([savetitle '.' printnames{i} '=simaverages(:,:,' num2str(i) ');'])
%     eval([savetitle '.' printnames{i} '_err=simerrors(:,:,' num2str(i) ');'])
% end
% eval(['save ' savetitle ' ' savetitle]);
