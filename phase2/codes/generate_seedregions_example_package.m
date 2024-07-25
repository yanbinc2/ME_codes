%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate seed regions for an example MNIST dataset.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear;

%directory='/mnt/sdb/NN/package_version4/data/';
directory='../data/';


% Load MNIST data and spectral cluster labels.
clustersfilename=sprintf('%sResNet18_PlantDisease_45K_Spec200_sampling.csv',directory);


% Load t-SNE projection data.
datafilename=sprintf('%sResNet18_PlantDisease_45K_Values_sampling.csv',directory);
augdatafileinfoname='dummy';


% The parameter values.
parametersfilename=sprintf('%sPlantVillage_generate_seedregions_params.txt',directory);


% Load the input data.
filename=sprintf('%s',datafilename);
fp=fopen(filename,'r');
s=fgets(fp);
ndim=sum(s==',')+1;
fseek(fp,0,-1);
form=[repmat('%f',1,ndim)];
de=sprintf(',');
cells=textscan(fp,form,'Delimiter',de,'Headerlines',1,'TreatAsEmpty',{'NA','na'});
fclose(fp);
ndim=length(cells); nimages=length(cells{1});
data=NaN*ones(nimages,ndim);%
for i=1:ndim
 data(:,i)=transpose(cells{i});
end
%tsnedata=data;


% Load the augmented data if the file exists.
filename=sprintf('%s',augdatafileinfoname);
fp=fopen(filename,'r');
if (fp<0)
 augdata=[];
else
 s=fgets(fp);
 augdirectory=s(1:(length(s)-1));
 s=fgets(fp);
 prefix=s(1:(length(s)-1));
 s=fgets(fp);
 naugmentations=atof2(s);
 fclose(fp);
 filename=sprintf('%s%s_001.csv',augdirectory,prefix);
 fp=fopen(filename,'r');
 if (fp<0)
  augdata=[];
 else
  fclose(fp);
  for ind=1:naugmentations
   str2=sprintf('%d',ind);
   l2=length(str2); l1=3-l2;
   str1=repmat('0',1,l1);
   str=sprintf('%s%s',str1,str2);
   filename=sprintf('%s%s_%s.csv',augdirectory,prefix,str);
   fp=fopen(filename,'r');
   form=[repmat('%f',1,ndim)];
   de=sprintf(',');
   cells=textscan(fp,form,'Delimiter',de,'Headerlines',1,'TreatAsEmpty',{'NA','na'});
   fclose(fp);
   augdata(:,:,ind)=NaN*ones(nimages,ndim);
   for i=1:ndim
    augdata(:,i,ind)=transpose(cells{i});
   end
  end
 end
end

%======= get t-SNE
fp=fopen(clustersfilename,'r');
s=fgets(fp);
ndim=sum(s==',')+1-3; % get first 3 columns
fseek(fp,0,-1);
%form=[repmat('%f',1,ndim)];
form=['%f','%f','%f','%s','%d','%d'];
de=sprintf(',');
cells=textscan(fp,form,'Delimiter',de,'Headerlines',1,'TreatAsEmpty',{'NA','na'});
fclose(fp);
tsnedata=NaN*ones(nimages,ndim);
for i=1:ndim
 tsnedata(:,i)=transpose(cells{i});
end
%========


% Load cluster labels of data points.
% Also load the true labels of data points.
% Find dominant class labels of regions.
filename=sprintf('%s',clustersfilename);
fp=fopen(filename,'r');
%form=['%s','%d','%s'];
form=['%f','%f','%f','%s','%d','%d']; %
de=sprintf(',');
cells=textscan(fp,form,'Delimiter',de,'Headerlines',1,'TreatAsEmpty',{'NA','na'});
fclose(fp);
regionneighborlabels=cells{6}; % get region index for each image
regionneighborlabels=transpose(regionneighborlabels);
nregions=max(regionneighborlabels);



% Obtain class labels of images.
imagelabels=zeros(1,nimages);
ulabels={}; nulabels=0;
for n=1:nimages
 %str=cells{6}{n};%
 str =int2str(cells{5}(n)); %get class label for each image
 %str=str(2:(length(str)-1)); %remove two quote marks
 [a,b]=ismember(str,ulabels);
 if (b<=0)
  nulabels=nulabels+1;
  ulabels{nulabels}=str;
  imagelabels(n)=nulabels;
 else
  imagelabels(n)=b;
 end
end

% Sort the class labels and reorder imagelabels.
[Y,I]=sort(ulabels);
I=1:nulabels;
sortedulabels=ulabels(I);
sortedinds=zeros(1,nulabels);
for i=1:nulabels
 j=I(i); sortedinds(j)=i;
end

sortedimagelabels=zeros(1,nimages);
for n=1:nimages
 i=imagelabels(n);
 j=sortedinds(i);
 sortedimagelabels(n)=j-1;
end

imagelabels=sortedimagelabels;
clear sortedimagelabels;
ulabels=sortedulabels;
clear sortedulabels;


% Load parameter values from the parameter file.
fp=fopen(parametersfilename,'r');
form=['%s','%f'];
de=sprintf('%c',9);
cells=textscan(fp,form,'Delimiter',de,'Headerlines',1,'TreatAsEmpty',{'NA','na'});
fclose(fp);
params=cells{2};

% Yan-Bin 2024/7
%filter=params(1); numthre=params(2); fracthre=params(3); topthre=params(4); nimagesthre=params(5);
%mthre=params(6); lwdthre=params(7); uppthre=params(8); nmarkersthre=params(9); rankthre=params(10);
%rankratiothre=params(11); overlapthre=params(12); sizethre=params(13); disparitythre=params(14); highthre=params(15); 
%dthre=params(16); nnbsthre=params(17); sharedthre=params(18); nvalidnbsthre=params(19); rthre=params(20);
%regionpairDmode=params(21); maxnseeds=params(22); nclassesthre=params(23); ratiothre=params(24); foldthre=params(25);
%diffratiothre=params(26);

filter=params(1); regionpairDmode=params(2); rthre=params(3); maxnseeds=params(4); diffratiothre=params(5); 


% Yan-Bin 2024/7
% Run generate_seedregions_package.m.
tic;
%[nseeds,seedinds,bilabels,regionpairD,nbregioninds]=generate_seedregions_package(data,augdata,tsnedata,regionneighborlabels,filter,numthre,fracthre,topthre,nimagesthre,mthre,lwdthre,uppthre,nmarkersthre,rankthre,rankratiothre,overlapthre,sizethre,disparitythre,highthre,dthre,nnbsthre,sharedthre,nvalidnbsthre,rthre,regionpairDmode,maxnseeds,nclassesthre,ratiothre,foldthre,diffratiothre);
[nseeds,seedinds,bilabels,regionpairD,nbregioninds]=generate_seedregions_package(data,augdata,tsnedata,regionneighborlabels,filter,regionpairDmode,rthre,maxnseeds,diffratiothre);
toc;


% The seed indices and bilabels are stored in output files.
seedindsfilename=sprintf('%sseedinds.txt',directory);
bilabelsfilename=sprintf('%sbilabels.txt',directory);
nbregionsfilename=sprintf('%sseedinds_neighborregions.txt',directory);


% Report seed indices.
filename=sprintf('%s',seedindsfilename);
fout=fopen(filename,'w');
for m=1:nseeds
 n=seedinds(m);
 fprintf(fout,'%d\n',n);
end
fclose(fout);


% Report bilabels.
filename=sprintf('%s',bilabelsfilename);
fout=fopen(filename,'w');
for i=1:nimages
 fprintf(fout,'%d%c%d\n',bilabels(i,1),9,bilabels(i,2));
end
fclose(fout);


% Report the top 6 neighboring regions of seed regions (including the seed regions themselves).

fout=fopen(nbregionsfilename,'w');
for m=1:nseeds
 n=seedinds(m);
 [Y,I]=sort(regionpairD(n,:),'ascend');
 for i=1:6
  fprintf(fout,'%d',I(i));
  if (i<6)
   fprintf(fout,'%c',9);
  end
 end
 fprintf(fout,'\n');
end
fclose(fout);


