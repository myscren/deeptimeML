% MCbg.m
% Updated from Montecarlo.m (of Keller)
% Run a full monte carlo simulation
% Each actual simulation task is conducted by mctask.m, this code provides
% the data needed for mctask to work and stores the results Any variables 
% needed during the simulation must exist in simitemsin; these variables
% will be read into data, filtered, and passed to mctask.
%% Load required variables
dataraw=xlsread('E:\Manuscripts\³Â¹úÐÛ¸å¼þ2021\StatisticalGeochemistry-master\igneous\res_final.xlsx');
% load newign;
load titlename; 
% Simitems is a cell array holding the names of all the variables to
% examine. Names must be formatted as in ign.elements
simitemsin={'TIO2';'AL2O3';'FEOT';'MGO';'CAO';'MNO';'NA2O';'K2O';'P2O5';'CR';'CO';...
     'NI';'LA';'CE';'PR';'ND';'SM';'EU';'GD';'TB';'DY';'HO';'ER';'TM';'YB';'LU';'Y';'SC';'V';...
     'ZR';'HF';'RB';'SR';'BA';'NB';'TA';'U';'TH';'S';'CU';'PB';'ZN';'BE';'GA'};
%  simitemsin={'YB'};
 for i=1:length(simitemsin)
    XP{i}=find(strcmp(titlename,simitemsin{i}));
 end
% See what the age of each sample is; this will be our independent variable
age_uncert_min = 50; % Set minimum age uncertainty (analagous to kernel width)
% ages = dataraw(:,5);
% age_uncert = dataraw(:,7);
% age_uncert(age_uncert<age_uncert_min | isnan(age_uncert)) = age_uncert_min;

ages = dataraw(:,7);
age_uncert = dataraw(:,6)-dataraw(:,5);
age_uncert(age_uncert<age_uncert_min | isnan(age_uncert)) = age_uncert_min;

% Construct a matrix holding all the data to be used in the simulation
uncert=zeros(1,length(simitemsin)+2);
data=zeros(length(ages),length(simitemsin)+2);
data(:,1) = age_uncert; % Independent variable uncert goes in position one
data(:,2) = ages; % Independent variable values go in position two
for i=1:length(simitemsin)
    data(:,i+2)=dataraw(:,XP{i});
    uncert(i+2)=0.01; %ign.err.(simitemsin{i});
end



%% Produce sample weights for bootstrap resamplingaaa

% Range of silica and age values to examine
simin=43;
simax=51;
agemin=0;
agemax=4000;

% Reject data that is out of the range of interest, is all NANs, or isn't from a contienent
% ign.MgO>simin &ign.MgO<simax &
% test=(ign.SiO2>simin &ign.SiO2<simax &ign.Age>agemin&ign.Age<agemax &~any(isnan(data(:,1:2)),2) &any(~isnan(data(:,4:end)),2) &ign.Elevation>-100 &~ign.oibs); 
test=(ages>agemin & ages<agemax & ~any(isnan(data(:,1:2)),2));
data=data(test,:);

% Compute weighting coefficients
% k=invweight(ign.Latitude(test),ign.Longitude(test),ign.Age(test));
k=invweight(dataraw(test,3),dataraw(test,4),dataraw(test,7));

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

% For each item in the simulation output, create a figure with the results
i=length(simitemsout);
while i>0
    figure
    errorbar(bincenters(1:end-1),nanmean(simaverages(:,1:end-1,i)),2.*nanmean(simerrors(:,1:end-1,i)),'.r')
    xlabel('Age (Ma)')
    ylabel(simitemsout{i})
    xlim([0,4000])
%     title(simtitle)
%     formatfigure
    i=i-1;
end

% i=length(simitemsout);
% while i>0
%     figure    
%     plot(data(:,2),data(:,3),'.');    hold on
%     errorbar(bincenters(1:end-1),nanmean(simaverages(:,1:end-1,i)),2.*nanmean(simerrors(:,1:end-1,i)),'.r')
%     xlabel('Age (Ma)')
%     ylabel(simitemsout{i})
%     xlim([0,4000])
%     ylim([0,max(nanmean(simaverages(:,1:end-1,i)))]*1.2)
%     set(gca,'tickdir','out')
% %     title(simtitle)
% %     formatfigure
%     i=i-1;
% end



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
exportmatrix(averages,'averages50_basalt_newign1.csv',',');
exportmatrix(uncertainties,'uncertainties50_basalt_newign1.csv',',');

%% Save results

% eval([savetitle '.bincenters=bincenters;']); eval([savetitle '.simaverages=simaverages;']); eval([savetitle '.simerrors=simerrors;']); eval([savetitle '.simitems=printnames;']);
% for i=1:length(printnames)
%     eval([savetitle '.' printnames{i} '=simaverages(:,:,' num2str(i) ');'])
%     eval([savetitle '.' printnames{i} '_err=simerrors(:,:,' num2str(i) ');'])
% end
% eval(['save ' savetitle ' ' savetitle]);
