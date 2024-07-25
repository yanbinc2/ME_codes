%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Merge seed regions for an example MNIST dataset.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear;

%directory='/mnt/sdb/NN/package_version4/data/';
directory='../data/';


% Load t-SNE projection data.
datafilename=sprintf('%sResNet18_PlantDisease_45K_Values_sampling.csv',directory);
augdatafileinfoname='dummy';


% Load spectral cluster labels.
clustersfilename=sprintf('%sResNet18_PlantDisease_45K_Spec200_sampling.csv',directory);


% Load valid seed indices and their labels.
seedindsfilename=sprintf('%sseedinds.txt',directory);


% Load bilabels of regions.
bilabelsfilename=sprintf('%sbilabels.txt',directory);


% The parameter values are storted in a file.
parametersfilename=sprintf('%sPlantVillage_merge_seedregions_params.txt',directory);


% Load seed indices and their labels.
seedinds=load(seedindsfilename);
seedinds=transpose(seedinds);
nseeds=length(seedinds);


% Set the file names of prediction outcomes of seed regions.
originalpredresultsfilename=sprintf('%sresults_of_original.mat',directory);
combpredresultsfilename=sprintf('%sresults_of_combination.mat',directory);
removepredresultsfilename=sprintf('%sresults_of_removal.mat',directory);


% The merged seed regions are stored in a file.
mergedclasslabelsfilename=sprintf('%smergedseedclasslabels.txt',directory);


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
data=NaN*ones(nimages,ndim);
for i=1:ndim
 data(:,i)=transpose(cells{i});
end


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


% Load cluster labels of data points.
% Also load the true labels of data points.
% Find dominant class labels of regions.
filename=sprintf('%s',clustersfilename);
fp=fopen(filename,'r');
form=['%s','%d','%s'];
de=sprintf(',');
cells=textscan(fp,form,'Delimiter',de,'Headerlines',1,'TreatAsEmpty',{'NA','na'});
fclose(fp);
regionneighborlabels=cells{2};
regionneighborlabels=transpose(regionneighborlabels);
nregions=max(regionneighborlabels);


% Obtain class labels of images.
imagelabels=zeros(1,nimages);
ulabels={}; nulabels=0;
for n=1:nimages
 str=cells{3}{n};
 str=str(2:(length(str)-1));
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


% Load the CNN prediction outcomes.
load(originalpredresultsfilename);
load(combpredresultsfilename);
load(removepredresultsfilename);



% Load parameters from an input file.
fp=fopen(parametersfilename,'r');
form=['%s','%f'];
de=sprintf('%c',9);
cells=textscan(fp,form,'Delimiter',de,'Headerlines',1,'TreatAsEmpty',{'NA','na'});
fclose(fp);
params=cells{2};

% Yan-Bin 2024/7
%confusionratiothre=params(1); usemarker=params(2); nprednumthre=params(3); pvalthre=params(4); ntoprankthre=params(5);
%importthre=params(6); quantilethre=params(7); cocontributionthre=params(8); replacementratiothre=params(9); localranksumpvalthre=params(10);
%sizethre=params(11); smallclustersizethre=params(12); jumpfoldthre=params(13); gapfoldthre=params(14); smallratiothre=params(15);
%pdiffthre=params(16); medvalthre=params(17); ninformativemarkersthre=params(18); ninformativemarkersthre2=params(19); pdiffthre2=params(20);
%cntdiffthre=params(21); reducedrankscorethre=params(22);

usemarker=params(1); nprednumthre=params(2); pvalthre=params(3); ntoprankthre=params(4);reducedrankscorethre=params(5);

% Load bilabels.
bilabels=load(bilabelsfilename);

regiontraininglabels=zeros(1,nimages);
for n=1:nregions
 ss=find((bilabels(:,1)==n)&(bilabels(:,2)==1));
 regiontraininglabels(ss)=n;
end

% Yan-Bin 2024/7
% Run merge_seedregions_package.m.
tic;
%[nmergeoutcomes,mergedclasslabels]=merge_seedregions_package(nseeds,seedinds,result_for_original,prob_for_original,combination_pairs,result_for_merge,result_for_removal,data,regiontraininglabels,confusionratiothre,usemarker,nprednumthre,pvalthre,ntoprankthre,importthre,quantilethre,cocontributionthre,replacementratiothre,localranksumpvalthre,sizethre,smallclustersizethre,jumpfoldthre,gapfoldthre,smallratiothre,pdiffthre,medvalthre,ninformativemarkersthre,ninformativemarkersthre2,pdiffthre2,cntdiffthre,reducedrankscorethre);
[nmergeoutcomes,mergedclasslabels]=merge_seedregions_package(nseeds,seedinds,result_for_original,prob_for_original,combination_pairs,result_for_merge,result_for_removal,data,regiontraininglabels,usemarker,nprednumthre,pvalthre,ntoprankthre,reducedrankscorethre);
toc;


% Report the merged seed classes.

nconfigs=length(mergedclasslabels(:,1));
filename=sprintf('%s',mergedclasslabelsfilename);
fout=fopen(filename,'w');
fprintf(fout,'index%c',9);
for n=1:nconfigs
 fprintf(fout,'class%d',n);
 if (n<nconfigs)
  fprintf(fout,'%c',9);
 else
  fprintf(fout,'\n');
 end
end
for i=1:nseeds
 fprintf(fout,'%d%c',seedinds(i),9);
 for j=1:nconfigs
  k=mergedclasslabels(j,i);
  fprintf(fout,'%d',k);
  if (j<nconfigs)
   fprintf(fout,'%c',9);
  else
   fprintf(fout,'\n');
  end
 end
end
fclose(fout);


%filename=sprintf('%s',mergedclasslabelsfilename);
%fout=fopen(filename,'w');
%fprintf(fout,'index%cclass\n',9);
%for n=1:nseeds
% fprintf(fout,'%d%c%d\n',seedinds(n),9,mergedclasslabels(n));
%end
%fclose(fout);
