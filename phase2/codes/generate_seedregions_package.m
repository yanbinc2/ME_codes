%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate seed regions from the input data.
% Inputs: embeded data, region labels, several free parameters.
% Outputs: indices of seed regions and the valid image labels in all regions.
% Execute the following procedures.
% 1. Calculate the pairwise distances of images and report the K nearest neighbors of each image.
% 2. Improve purity of regions by either incorporating geometric information in the embedded space or marker information.
% 3. Select seed regions from the spectral clusters.
% 4.1. Start with the region pairs which are far apart and equally split the data.
% 4.2. Incrementally select new regions that have max distances to existing seed regions.
% 4.3. If the selected region yields similar classification outcomes by replacing it with any one of existing seed regions, then skip this selected region and find the next one.
% 4.4. If several consecutive selected regions are skipped, then stop adding seed regions.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [nseeds, seedinds, bilabels, regionpairD, nbregioninds] = generate_seedregions_package(data, augdata, tsnedata, regionneighborlabels, filter, regionpairDmode, rthre, maxnseeds, diffratiothre)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set the fixed values for certain parameters.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

numthre=100;
fracthre=0.9;
sizethre=10;
nclassesthre=4;
ratiothre=10.0;
foldthre=10.0;

if (filter==2)
 topthre=20;
 nimagesthre=1000;
 mthre=110;
 lwdthre=120;
 uppthre=280;
 nmarkersthre=5;
 rankthre=100;
 rankratiothre=0.3;
 disparitythre=1.2;
 overlapthre=0.75;
 highthre=0.3;
 dthre=25;
 sharedthre=0.2;
 nnbsthre=10;
 nvalidnbsthre=5;
else
 topthre=0;
 nimagesthre=0;
 mthre=0;
 lwdthre=0;
 uppthre=0;
 nmarkersthre=0;
 rankthre=0;
 rankratiothre=0;
 overlapthre=0;
 disparitythre=0;
 hightre=0;
 dthre=0;
 nnbsthre=0;
 sharedthre=0;
 nvalidnbsthre=0;
end


[nimages,ndim]=size(data);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Improve purity of regions.
% In other words, try to find the member data points in the region which share the same underlying labels.
% Two alternative approaches are used.  Also allow not to improve region purity.
% (1) Incorporate geometric information of member data points in the region.
% Demarcate the center zone, which comprises a compact core of a region.
% The periphery data points usualy do not share the same class labels as the center zone.
% (2) Find the markers which separate member images of the region.
% If a significant number of separation markers exist, then use them to partition the region into subsets of images.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% bilabels report the confident members of regions.
% bilabels have two columns.
% column 1: region labels.
% column 2: within regions, indicate whether an image is a core member.

nregions=max(regionneighborlabels);
bilabels=zeros(nimages,2);
for n=1:nregions
 ss=find(regionneighborlabels==n); 
 bilabels(ss,1)=n;
 bilabels(ss,2)=0;
end


% Option 0: Do not apply filters.
regiontraininglabels=regionneighborlabels;

% Option 1: Incorporate geometric information to filter images.
% Apply the filter to find valid images according to tSNE projection data.
% If the distance distribution is too flat, then discard the region. 
if (filter==1)
 
 for n=1:nregions
  ss=find(regionneighborlabels==n); 
  subdata=tsnedata(ss,:);

% === Yan-Bin 2024/5 ===
%   [nhighvals,inds]=demarcate_center_zones(subdata);
%   if (nhighvals>=50)
%    inds=[];
%   end
  if (isempty(subdata)==0) %non empty
      [nhighvals,inds]=demarcate_center_zones(subdata);
      if (nhighvals>=50)
       inds=[];
      end
  else
      inds=[];
  end
% ==========
  bilabels(ss,1)=n;
  bilabels(ss(inds),2)=1;
 end

 regiontraininglabels=zeros(1,nimages);
 for m=1:nregions
  tt=find((bilabels(:,1)==m)&(bilabels(:,2)==1));
  regiontraininglabels(tt)=m;
 end

% Option 2: Incorporate marker information to filter images.
elseif (filter==2)

 % Convert the data and augmented data into rank values.
 dataranks=zeros(nimages,ndim);
 for i=1:nimages
  [Y,I]=sort(data(i,:),'descend');
  dataranks(i,I)=1:ndim;
 end

 naugs=length(augdata(1,1,:));
 augdataranks=zeros(nimages,ndim,naugs);
 for ind=1:naugs
  for i=1:nimages
   [Y,I]=sort(augdata(i,:,ind),'descend');
   augdataranks(i,I,ind)=1:ndim;
  end
 end

 % For each combination of (image,marker), report the rank of the marker in the output of the image.
 markerranks=zeros(nimages,ndim);
 for i=1:nimages
  [Y,I]=sort(data(i,:),'descend');
  for j=1:ndim
   k=I(j); markerranks(i,k)=j;
  end
 end

 % For each marker, count the number of images where it is within top k ranks.
 % Vary k.

 ks=10:10:100; nks=length(ks);
 markerrankcnts=zeros(nks,ndim);
 for i=1:nks
  k=ks(i);
  for j=1:ndim
   l=sum(markerranks(:,j)<=k);
   markerrankcnts(i,j)=l;
  end
 end

 % For each combination of (image,marker), report the rank of the marker in the output of the image.
 markerranks=zeros(nimages,ndim);
 for i=1:nimages
  [Y,I]=sort(data(i,:),'descend');
  for j=1:ndim
   k=I(j); markerranks(i,k)=j;
  end
 end

 % For each marker, count the number of images possessing <= topthre ranks. 
 toprankcnts=zeros(1,ndim);
 for n=1:ndim
  toprankcnts(n)=sum(markerranks(:,n)<=topthre);
 end

 % Select the markers where the top rank counts >= nimagesthre.
 nimagesthre=1000;
 markerinds=find(toprankcnts>=nimagesthre);
 nmarkers=length(markerinds);


 % For each region and each marker, sort members by marker values and subdivide the members into two parts.
 % Calculate the SNRs from dataranks and augdataranks for each region and each marker.
 % Also report the mean and stds of dataranks and augdataranks.
 % markerrankinfo is a naugsx5xnmarkersxnregions tensor.
 % dimension 1: results derived from dataranks and naugs augdataranks.
 % dimension 2: two means and two stds and snr of the two groups partitioned by marker rank values.
 % dimension 3: markers.
 % dimension 4: regions.

 markerrankinfo=zeros(11,5,nmarkers,nregions);
 for n=1:nregions
  %subset=find(regiontraininglabels==n);
  subset=find(regionneighborlabels==n);
  for p=1:nmarkers
   q=markerinds(p);
   [Y,I]=sort(dataranks(subset,q));
   k=round(length(subset)/2);
   g1=I(1:k); g2=I((k+1):length(subset));
   qvals1=dataranks(subset(g1),q);
   qvals2=dataranks(subset(g2),q);
   markerrankinfo(1,1,p,n)=mean(qvals1);
   markerrankinfo(1,2,p,n)=mean(qvals2);
   markerrankinfo(1,3,p,n)=std(qvals1);
   markerrankinfo(1,4,p,n)=std(qvals2);
   markerrankinfo(1,5,p,n)=(markerrankinfo(1,2,p,n)-markerrankinfo(1,1,p,n))*2/(markerrankinfo(1,3,p,n)+markerrankinfo(1,4,p,n));
   for i=1:10
    qvals1=augdataranks(subset(g1),q,i);
    qvals2=augdataranks(subset(g2),q,i);
    markerrankinfo(i+1,1,p,n)=mean(qvals1);
    markerrankinfo(i+1,2,p,n)=mean(qvals2);
    markerrankinfo(i+1,3,p,n)=std(qvals1);
    markerrankinfo(i+1,4,p,n)=std(qvals2);
    markerrankinfo(i+1,5,p,n)=(markerrankinfo(i+1,2,p,n)-markerrankinfo(i+1,1,p,n))*2/(markerrankinfo(i+1,3,p,n)+markerrankinfo(i+1,4,p,n));
   end
  end
 end

 % Find the markers to distinguish the two groups in each region.
 % For each region, sort markers by the differences between group 1 means and group 2 means.
 % Exclude the markers where the group 1 means are too high. 
 markervalids=zeros(nmarkers,nregions);
 for n=1:nregions
  for i=1:nmarkers
   qvals=markerrankinfo(2:11,1,i,n);
   val=mean(qvals);
   if (val<=mthre)
    markervalids(i,n)=1;
   end
  end
 end

 markerdiffvals=zeros(nmarkers,nregions);
 for n=1:nregions
  for i=1:nmarkers
   qvals=markerrankinfo(2:11,2,i,n)-markerrankinfo(2:11,1,i,n);
   markerdiffvals(i,n)=mean(qvals);
  end
 end

 sortedvalidmarkers=zeros(nmarkers,nregions);
 for n=1:nregions
  ss=find(markervalids(:,n)==1);
  [YY,II]=sort(markerdiffvals(ss,n),'descend');
  for i=1:length(II)
   j=ss(II(i));
   sortedvalidmarkers(i,n)=j;
  end
 end


 % Identify the markers which may distinguish member images with the dominant and non-dominant labels of the mixed regions.
 % Among the top ranking sorted valid markers, find the ones where the mean ranks of groups 1 and 2 are within the specified range.
 
 separatemarkers=zeros(nmarkers,nregions);
 for n=1:nregions
  cands=[];
  for i=1:topthre
   j=sortedvalidmarkers(i,n);
   if (j>0)
    qvals1=markerrankinfo(2:11,1,j,n);
    qvals2=markerrankinfo(2:11,2,j,n);
    val1=mean(qvals1); val2=mean(qvals2);
    if ((val1<=lwdthre)&(val2>=uppthre))
     cands=[cands j];
    end
   end
  end
  separatemarkers(1:length(cands),n)=cands;
 end


 % Find the majority and minority member images of a region in terms of the selected marker value.
 % If a region has too many (>=nmarkersthre) separation markers, then this region is too mixed.  Discard the region directly.
 % Among the remaining regions with separation markers, check whether each separation marker is valid.
 % Identify all the regions with good rank values of the separation marker.  Extract their member images.
 % Obtain the distribution of the separation marker values on those member images v1.
 % Obtain the distribution of the separation marker values on all images v0.
 % If the fraction of v0 with rank values <= rankthre is >= ratiothre, then too many images have good rank values.  Discard the marker.
 % Among the member images of a region, find the member images whose marker values <= rankthre and the member images whose marker values > rankthre.
 % If there are multiple valid markers in a region, find the partition that is consistent across multiple valid markers.  If there is no consistent partition, then do not incur partition.
 % If the partition yields two sets with similar sizes, then do not incur partition.
 % If the partition yields two sets with small sizes, then do not incur partition.
 % If the partition passes these tests, then choose the members of the majority set.

 regiontraininglabels=zeros(1,nimages);
 for n=1:nregions
  subset=find(regionneighborlabels==n);
  cands=transpose(separatemarkers(:,n));
  cands=cands(find(cands>0));
  if (length(cands)==0)
   regiontraininglabels(subset)=n;
  elseif (length(cands)<nmarkersthre)
   candsvalid=zeros(1,length(cands));
   for i=1:length(cands)
    q=markerinds(cands(i));
    v0=dataranks(:,q);
    r=sum(v0<=rankthre)/nimages;
    if (r<rankratiothre)
     candsvalid(i)=1;
    end
   end
   cands=cands(find(candsvalid==1));
   if (length(cands)==0)
    regiontraininglabels(subset)=n;
   else
    subpartitions=zeros(length(cands),length(subset));
    for i=1:length(cands)
     q=markerinds(cands(i));
     s1=find(dataranks(subset,q)<=rankthre);
     s2=find(dataranks(subset,q)>rankthre);
     subpartitions(i,s1)=1; subpartitions(i,s2)=2; 
    end
    consistent=1; i=1;
    while ((i<=(length(cands)-1))&(consistent==1))
     cnts=zeros(2,2);
     cnts(1,1)=sum((subpartitions(i,:)==1)&(subpartitions(i+1,:)==1));
     cnts(1,2)=sum((subpartitions(i,:)==1)&(subpartitions(i+1,:)==2));
     cnts(2,1)=sum((subpartitions(i,:)==2)&(subpartitions(i+1,:)==1));
     cnts(2,2)=sum((subpartitions(i,:)==2)&(subpartitions(i+1,:)==2));
     k=sum(sum(cnts));
     val1=(cnts(1,1)+cnts(2,2))/k; val2=(cnts(1,2)+cnts(2,1))/k;
     if ((val1<overlapthre)&(val2<overlapthre))
      consistent=0;
     end
     i=i+1;
    end
    if (consistent==0)
     regiontraininglabels(subset)=n;
    else
     s1=find(subpartitions(1,:)==1); l1=length(s1);
     s2=find(subpartitions(1,:)==2); l2=length(s2);
     if ((max(l1,l2)<sizethre)|((max(l1,l2)/min(l1,l2))<disparitythre))
      regiontraininglabels(subset)=n;
     else
      if (l1>=l2)
       regiontraininglabels(subset(s1))=n; 
      else
       regiontraininglabels(subset(s2))=n; 
      end
     end
    end
   end
  end
 end

 bilabels=zeros(nimages,2);
 for n=1:nregions
  s1=find(regionneighborlabels==n);
  s2=find(regiontraininglabels==n);
  bilabels(s1,1)=n;
  bilabels(s2,2)=1;
 end

 % Some regions do not have training set members.
 % To avoid possible trouble of extracting empty training set, set the training set members to all members.

 bilabels2=bilabels;
 for n=1:nregions
  if (sum(regiontraininglabels==n)==0)
   ss=find(regionneighborlabels==n);
   bilabels2(ss,2)=1;
  end
 end
 bilabels=bilabels2;
 clear bilabels2;

end

 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate region pair distances.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Calculate region pair distances by blocks.
regionpairD=zeros(nregions,nregions);
blocksize=10; nblocks=round(nregions/blocksize);

for n=1:nblocks

 i1=(n-1)*blocksize+1; i2=min(i1+blocksize-1,nregions);
 k=0; X=zeros(1,ndim); labels=zeros(1,1);
 for i=i1:i2
  if (regionpairDmode==0)
   ss=find(regionneighborlabels==i);
  else
   ss=find(regiontraininglabels==i);
  end
  if (length(ss)==0)
   ss=find(regionneighborlabels==i);
  end
  for j=1:length(ss)
   vec=data(ss(j),:);
   k=k+1; X(k,:)=vec; labels(1,k)=i;
  end
 end
 D=pdist(X,'euclidean');
 D=squareform(D);
 for i=i1:i2
  for j=(i+1):i2
   s1=find(labels==i); s2=find(labels==j);
   tmpD=D(s1,s2);
   val=mean(tmpD(:));
   regionpairD(i,j)=val;
   regionpairD(j,i)=val;
  end
 end

end

for n=1:nblocks
 i1=(n-1)*blocksize+1; i2=min(i1+blocksize-1,nregions);
 k=0; X1=zeros(1,ndim); labels1=zeros(1,1);
 for i=i1:i2
  if (regionpairDmode==0)
   ss=find(regionneighborlabels==i);
  else
   ss=find(regiontraininglabels==i);
  end
  if (length(ss)==0)
   ss=find(regionneighborlabels==i);
  end
  for j=1:length(ss)
   vec=data(ss(j),:);
   k=k+1; X1(k,:)=vec; labels1(1,k)=i;
  end
 end
 for m=(n+1):nblocks  

  j1=(m-1)*blocksize+1; j2=min(j1+blocksize-1,nimages);
  k=0; X2=zeros(1,ndim); labels2=zeros(1,1);
  for i=j1:j2
   if (regionpairDmode==0)
    ss=find(regionneighborlabels==i);
   else
    ss=find(regiontraininglabels==i);
   end
   if (length(ss)==0)
    ss=find(regionneighborlabels==i);
   end
   for j=1:length(ss)
    vec=data(ss(j),:);    
    k=k+1; X2(k,:)=vec; labels2(1,k)=i;
   end
  end
 
  X=[X1;X2]; D=pdist(X,'euclidean'); D=squareform(D);
  sD=D(1:length(labels1),(length(labels1)+1):(length(labels1)+length(labels2)));

  for i=i1:i2
   for j=j1:j2
    s1=find(labels1==i); s2=find(labels2==j);
    tmpD=sD(s1,s2);
    val=mean(tmpD(:));
    % === Yan-Bin 2024/5 ===
    %regionpairD(i,j)=val;
    %regionpairD(j,i)=val;
    if (isnan(val)==0) %non NaN
        regionpairD(i,j)=val;
        regionpairD(j,i)=val;
    end
    % ==========
   end
  end

 end
end



% Calculate region pair distances of tSNE data by blocks.
tsneregionpairD=zeros(nregions,nregions);
blocksize=10; nblocks=round(nregions/blocksize);

for n=1:nblocks

 i1=(n-1)*blocksize+1; i2=min(i1+blocksize-1,nregions);
 k=0; X=zeros(1,3); labels=zeros(1,1);
 for i=i1:i2
  if (regionpairDmode==0)
   ss=find(regionneighborlabels==i);
  else
   ss=find(regiontraininglabels==i);
  end
  if (length(ss)==0)
   ss=find(regionneighborlabels==i);
  end
  for j=1:length(ss)
   vec=tsnedata(ss(j),:);
   k=k+1; X(k,:)=vec; labels(1,k)=i;
  end
 end
 D=pdist(X,'euclidean');
 D=squareform(D);
 for i=i1:i2
  for j=(i+1):i2
   s1=find(labels==i); s2=find(labels==j);
   tmpD=D(s1,s2);
   val=mean(tmpD(:));
   tsneregionpairD(i,j)=val;
   tsneregionpairD(j,i)=val;
  end
 end

end


for n=1:nblocks
 i1=(n-1)*blocksize+1; i2=min(i1+blocksize-1,nregions);
 k=0; X1=zeros(1,3); labels1=zeros(1,1);
 for i=i1:i2
  if (regionpairDmode==0)
   ss=find(regionneighborlabels==i);
  else
   ss=find(regiontraininglabels==i);
  end
  if (length(ss)==0)
   ss=find(regionneighborlabels==i);
  end
  for j=1:length(ss)
   vec=tsnedata(ss(j),:);
   k=k+1; X1(k,:)=vec; labels1(1,k)=i;
  end
 end
 for m=(n+1):nblocks  

  j1=(m-1)*blocksize+1; j2=min(j1+blocksize-1,nimages);
  k=0; X2=zeros(1,3); labels2=zeros(1,1);
  for i=j1:j2
   if (regionpairDmode==0)
    ss=find(regionneighborlabels==i);
   else
    ss=find(regiontraininglabels==i);
   end
   if (length(ss)==0)
    ss=find(regionneighborlabels==i);
   end
   for j=1:length(ss)
    vec=tsnedata(ss(j),:);    
    k=k+1; X2(k,:)=vec; labels2(1,k)=i;
   end
  end
 
  X=[X1;X2]; D=pdist(X,'euclidean'); D=squareform(D);
  sD=D(1:length(labels1),(length(labels1)+1):(length(labels1)+length(labels2)));

  for i=i1:i2
   for j=j1:j2
    s1=find(labels1==i); s2=find(labels2==j);
    tmpD=sD(s1,s2);
    val=mean(tmpD(:));
    tsneregionpairD(i,j)=val;
    tsneregionpairD(j,i)=val;
   end
  end

 end
end


% Debug
%tsneregionpairD=regionpairD;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Identify valid regions using either geometric information or marker information.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Option 0: Do not apply filters.
validregions=ones(1,nregions);


% Option 1: Incorporate geometric information to filter images.
if (filter==1)

 % Use the center zones of the tSNE projections to filter out regions.
 % If a region has < numthre valid images, then the region is invalid.
 % If a region has < fracthre fraction of valid images, then the region is invalid.
 for n=1:nregions
  n1=sum(bilabels(:,1)==n);
  n2=sum((bilabels(:,1)==n)&(bilabels(:,2)==1));
  if ((n1<numthre)|((n2/n1)<fracthre))
   validregions(n)=0;  
  end
 end
 

% Option 2: Incorporate marker information to filter images.
elseif (filter==2)

 % For each combination of (image,vggmarker), report the rank of the vggmarker in the output of the image.
 markerranks=zeros(nimages,ndim);
 for i=1:nimages
  [Y,I]=sort(data(i,:),'descend');
  for j=1:ndim
   k=I(j); markerranks(i,k)=j;
  end
 end

 % For each VGG marker, count the number of images possessing <= topthre ranks.
 toprankcnts=zeros(1,ndim);
 for n=1:ndim
  toprankcnts(n)=sum(markerranks(:,n)<=topthre);
 end

 % Select the markers where the top rank counts >= nimagesthre.
 markerinds=find(toprankcnts>=nimagesthre);
 nmarkers=length(markerinds);

 % For each region, find the markers which have top ranking values in >= ratiothre fraction of member images.
 % Also report the fractions of images which have top ranking values in each region.
 regionmarkers=zeros(nregions,nmarkers);
 regionmarkerfracs=zeros(nregions,nmarkers);
 for n=1:nregions
  subset=find(regionneighborlabels==n);
  for i=1:nmarkers
   vec=markerranks(subset,markerinds(i));
   n1=sum(vec<=topthre); n0=length(subset);
   regionmarkerfracs(n,i)=n1/n0;
   if (n1>=(n0*highthre))
    regionmarkers(n,i)=1;
   end
  end
 end

 % Calculate the distance ranks of regions.
 % regiondistranks(i,j) reports the rank of region j in terms of the distances to region i.

 regiondistranks=zeros(nregions,nregions);
 for n=1:nregions
  [YY,II]=sort(regionpairD(n,:));
  for i=1:nregions
   j=II(i); regiondistranks(n,j)=i;
  end
 end

 % For each marker, extract the regions with high fractions of the marker and evaluate the their rank distance distributions.
 % Sort markers by the rank distance distributions.

 markermeddistranks=inf*ones(1,nmarkers);
 for n=1:nmarkers
  subset=find(regionmarkerfracs(:,n)>=highthre);
  if (length(subset)>=5)
   dG=regiondistranks(subset,subset);
   for i=1:length(subset)
    dG(i,i)=0;
   end
   vals=dG(find(dG>0));
   markermeddistranks(n)=median(vals(:));
  end
 end

 [YY,II]=sort(markermeddistranks);

 % Identify the informative markers where markermeddistranks <= dthre.
 isinformativemarker=zeros(1,nmarkers);
 isinformativemarker(find(markermeddistranks<=dthre))=1;

 % Find the neighboring regions of each region.

 regionnbs=zeros(nregions,21);
 for i=1:nregions
  vals=regionpairD(i,:);
  [Y,I]=sort(vals);
  regionnbs(i,:)=I(1:21);
 end


 % For each (region,neighboring region) pair, count the fraction of shared region informative markers among their union of region markers.

 regionnbsharedmarkerfracs=zeros(nregions,21);
 for n=1:nregions
  s1=find(regionmarkers(n,:)==1);
  s1=s1(find(isinformativemarker(s1)==1));
  for i=1:21
   j=regionnbs(n,i);
   s2=find(regionmarkers(j,:)==1);
   s2=s2(find(isinformativemarker(s2)==1));
   f=length(intersect(s1,s2))/length(union(s1,s2));
   regionnbsharedmarkerfracs(n,i)=f;
  end
 end

 % For each region find valid neighbors according to regionnbsharedmarkerfracs.
 % Within top nnbsthress neighbors, find the ones whose regionnbsharedmarkerfracs values >= sharedthre.
 % If less than nvalidnbsthre neighbors are valid, then the region is not considered. 
 validneighbors=zeros(nregions,nnbsthre+1);
 for i=1:nregions
  v=regionnbsharedmarkerfracs(i,1:(nnbsthre+1));
  ss=find(v>=sharedthre);
  validneighbors(i,ss)=1;
 end

 validregions=zeros(1,nregions);
 for n=1:nregions
  k=sum(validneighbors(n,2:(nnbsthre+1))==1);
  if (k>=nvalidnbsthre)
   validregions(n)=1;
  end
 end

 % If a region has no training member images, then it is not valid.
 for n=1:nregions
  if (sum(regiontraininglabels==n)==0)
   validregions(n)=0;
  end
 end

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate the distances of each image to nk nearest neighbors for each region.
% When doing kNN prediction, only need to consider those nk nearest neighbors.
% Distances of many of those nk nearest neighbors are already in neighborDs.
% Represent the distances as a sparse matrix.  Zero entries denote NA entries.
% Only consider the images in neighboring regions.
% Consider the top nk*2 nearest neighboring regions because only need to consider nk neighboring images of one image.
% Leave space for the pathological conditions when the nearest nk neighborinds images are not in the nearest nk regions.
% For each image and each region, only store the closest nk members of the region.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

nk=10; blocksize=5000; nblocks=round(nimages/blocksize);
if (regionpairDmode==0)
 neighborinds=find(regionneighborlabels>0);
else
 neighborinds=find(regiontraininglabels>0);
end
nneighbors=length(neighborinds);
imageneighborDs=sparse([],[],[],nimages,nneighbors);

topN=nk*2;


% For each image and each region, only store the closest nk members of the region.
% Group images by regions.  For images in region i, only consider the neighorinds in the regions close to region i.
% Need to consider all regions versus all valid regions.

for n=1:nregions

 subset1=find(regionneighborlabels==n);
 X1=data(subset1,:);
 [Y1,I1]=sort(regionpairD(n,:),'ascend');
 %candinds=I1(1:(topN+1));
 candinds=I1(find(validregions(I1)==1));
 for r=1:length(candinds)
  m=candinds(r);
  subset2=intersect(neighborinds,find(regionneighborlabels==m));
  if (length(subset2)>0)
   X2=data(subset2,:);
   D=pdist2(X1,X2,'euclidean');
   template=D;
   maxval=max(template(:));
   ss=find(D>0);
   template(ss)=maxval-template(ss);
   [Y,I]=sort(transpose(template),'descend');
   sinds=transpose(I(1:nk,:));
   tsinds=subset2(sinds);
   [a,b]=ismember(tsinds,neighborinds);
   nbtsinds=b;
   sD=maxval-transpose(Y(1:nk,:));
   c1=repmat(subset1,1,nk); c1=transpose(c1);
   c2=nbtsinds(:); 
   c3=sD(:);
   tmat=sparse(c1,c2,c3,nimages,nneighbors);
   imageneighborDs=imageneighborDs+tmat;
  end
 end

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate seed regions.
% Start with region pairs which are far apart and equally split the data.
% Incrementally select the regions which are the most distant from existing seed regions.
% Consider the regions which are labeled valid regions.
% Calculate the replacement score of this candidate region by replacing it for each of existing seeds.  The replacement score is the minimum of them.
% When running kNN, include only the data points in the center zones as training data.
% For each replacement, count the number of images assigned to each new class.
% Count the fraction of differentially labeled images which are labeled with the replaced class either before or after replacement.
% Find candidate replacements that best match the original class in terms of the fractions.
% Among the candidate replacements pick the one with the lowest ndiffs.
% Some spectral clusters have rather scattered data points.  Use the concentrated data points as the training images.
% If the replacement score is below a threshold, then skip it and consider the next candidate.
% If several consecutive candiates all have low replacement scores, then stop.
% If the number of seed regions reaches a predetermined number, then stop.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% For each region pair, count the number of other regions closer to each region.
npairs=0; paircnts=zeros(1,4);
for i=1:(nregions-1)
 for j=(i+1):nregions
  npairs=npairs+1; paircnts(npairs,:)=[i j 0 0];
  ii=setdiff(1:nregions,[i j]);
  vec=regionpairD([i j],ii);
  paircnts(npairs,3)=sum(vec(1,:)<vec(2,:));
  paircnts(npairs,4)=sum(vec(1,:)>vec(2,:));
 end
end

rs=zeros(npairs,1);
for i=1:npairs
 val1=max(paircnts(i,3:4)); val2=min(paircnts(i,3:4));
 rs(i)=val1/val2;
end

% Sort region pairs by their average distances.
% Find the region pairs from the top where the paircnts components 3 and 4 have similar numbers.
pairDs=zeros(npairs,1);
for n=1:npairs
 i=paircnts(n,1); j=paircnts(n,2);
 pairDs(n)=regionpairD(i,j);
end
[Y,I]=sort(pairDs,'descend');
pairDs=pairDs(I);
paircnts=paircnts(I,:);
rs=rs(I);

i=1; rind=0;
while ((i<=npairs)&(rind==0))
 if ((validregions(paircnts(i,1))==1)&(validregions(paircnts(i,2))==1)&(rs(i)<=rthre))  
  rind=i;
 end
 i=i+1;
end

seedinds=paircnts(rind,1:2);


% Incrementally incorporate seed regions that have max distances to all existing seed regions.
% Some regions have large distances to existing seed regions but yield similar classification outcomes by replacing them for one of existing seeds.
% Thus, calculate replacement scores and use them to filter out selected regions.
% In each iteration, fetch each candidate region, replace its neighbors for training images in one seed region, and run knn. 
% Compare the classification results using seed region neighbors as training images and replacing one seed region neighbors with those of the candidate region.
% The replacement score is the minimum difference of classification outcomes by replacing each seed region with the candidate region.
% If the replacement score is smaller than a threshold, then do not incorporate it in the seeds and consider the next candidate.
% If a consecutive number of candidate regions do not pass the replacement scores, then stop.
% If the maxinum number of seeds are selected, then stop.
% Compare to the previous version (generate_seedregions_threedatasets_20310311.m), evaluate the replacement scores only on one candidate region instead of all.
% Also, select regions based on replacement scores only will bias selecting regions without consensus labels.

% Report a fixed number of seed regions.  The fixed number is nclassesthre.
% In each iteration, find the 10 furtherest regions from all existing classes.
% Treat each of these 10 furtherest regions as a candidate region.
% Find the closest 3 existing classes to the candidate region.
% Replace the selected 3 existing classes with the candidate region to calculate diffratios and ndiffs. 
% Among the 10 candidate regions pick the one with the largest minndiff and proceed.
% Stop when the number of seed regions is the fixed number attained.
% Calculate the distances between all images and training images.


ncandsconsidered=1; nneighborclasses=3; 

blocksize=5000; nblocks=round(nimages/blocksize);

visited=zeros(1,nregions); cnt=0; flag=1;

seedminndiffs=nimages*ones(1,2);

while (flag==1)
 
 nclasses=length(seedinds); 
 traininginds=[];
 traininglabels=[];
 for m=1:nclasses
  n=seedinds(m);
  ss=find(regiontraininglabels==n);
  %ss=find(regionneighborlabels==n);
  traininginds=[traininginds ss];
  traininglabels=[traininglabels m*ones(1,length(ss))];
 end
 ntraining=length(traininginds);

 [a,b]=ismember(traininginds,neighborinds);
 trainingD=transpose(imageneighborDs(:,b));

 predlabels=kNN(nclasses,ntraining,traininginds,traininglabels,nimages,trainingD,nk); 
 
 pretraininglabels=traininglabels; 

 vals=zeros(1,nregions);
 for i=1:nregions
  if ((validregions(i)==1)&(ismember(i,seedinds)==0)&(visited(i)==0))
   vals(i)=min(regionpairD(i,seedinds));   
  end
 end
 
 [Y,I]=sort(vals,'descend');

 if (Y(1)<=0)

  flag=0;

 else

  minndiffs=zeros(1,ncandsconsidered);
  minvals=inf*ones(1,ncandsconsidered);

  for cind=1:ncandsconsidered

   maxind=I(cind);
   r=maxind;
   confuse=zeros(nclasses,nclasses,nclasses);
   extrainds=find(regiontraininglabels==r);   
   %extrainds=find(regionneighborlabels==r);   
   nextra=length(extrainds);  
   [a,b]=ismember(extrainds,neighborinds);
   extraD=transpose(imageneighborDs(:,b));         
   ndiffs=nimages*ones(1,nclasses);
   diffratios=inf*ones(1,nclasses);

   % Sort existing classes by their average distances to the candidate region.
   % Select the closest 5 classes to replace.  If there are less than or equal to 5 classes, then select all of them.
   ds=regionpairD(r,seedinds);
   %ds=tsneregionpairD(r,seedinds);
   [Y2,I2]=sort(ds,'ascend');
   classconsidered=zeros(1,nclasses);
   if (nclasses<nneighborclasses)
    classconsidered=ones(1,nclasses);
   else
    classconsidered(I2(1:nneighborclasses))=1;
   end

   for m=1:nclasses
    if (classconsidered(m)==1)
     tmptraininginds=traininginds;
     tmptraininglabels=traininglabels;
     tmptrainingD=trainingD;
     ss=find(traininglabels==m);
     i1=min(ss); i2=max(ss);
     tmptraininginds=[traininginds(1:(i1-1)) extrainds traininginds((i2+1):length(traininginds))];
     tmptraininglabels=[traininglabels(1:(i1-1)) m*ones(1,length(extrainds)) traininglabels((i2+1):length(traininginds))];
     tmptrainingD=[trainingD(1:(i1-1),:);extraD;trainingD((i2+1):length(traininginds),:)];    
     ntmptraining=length(tmptraininginds);
     tmppredlabels=kNN(nclasses,ntmptraining,tmptraininginds,tmptraininglabels,nimages,tmptrainingD,nk);

     for i=1:nclasses
      for j=1:nclasses
       k=sum((predlabels==i)&(tmppredlabels==j));
       confuse(i,j,m)=k;
      end
     end

     v1=confuse(m,:,m);
     v2=transpose(confuse(:,m,m));
     val1=sum(v1)-v1(m); val2=sum(v2)-v2(m);
     diffratios(m)=(val1+val2)/v1(m);
     k=sum(predlabels~=tmppredlabels);
     ndiffs(m)=k;
    end
   end

   minval=min(diffratios);
   minvals(cind)=minval;

   if (nclasses<=nclassesthre)
    subset=find(diffratios<=ratiothre);
   else
    subset=find((diffratios<=ratiothre)&(diffratios<=(minval*foldthre)));
   end
   if (length(subset)==0)
    subset=find(diffratios<inf);
   end

   if (length(subset)==0)
    subset=find(classconsidered==1);
   end

   vals=ndiffs(subset);
   k=subset(find(vals<=min(vals)));
   k=k(1);
   minndiff=ndiffs(k);   

   minndiffs(cind)=minndiff;

   %visited(maxind)=1;
  
  end

  minndiff=max(minndiffs);  
  maxind=find(minndiffs>=minndiff);
  maxind=maxind(1);
  minval=minvals(maxind);
  maxind=I(maxind);

  seedminndiffs=[seedminndiffs minndiff];

  
  % Debug for data.
  %fprintf('region %d, label %d, mindiffratio=%f, minndiff=%d\n',maxind,regionlabels(maxind),minval,minndiff);

  seedinds=[seedinds maxind];
  
  if (length(seedinds)>=maxnseeds)
   flag=0;
  end
  
 end

end

seedinds=seedinds(find(seedminndiffs>=(nimages*diffratiothre)));

nseeds=length(seedinds);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Find the neighboring regions of each seed region.
% Only consider validregions.
% Exclude the seed regions themselves.
% Sort regions by distances of all features and by tSNE projections.
% Choose the neighboring regions by the distances of tSNE projections.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

nnbs=4;
nbregioninds=zeros(nseeds,nnbs+1);
for m=1:nseeds
 n=seedinds(m);
 [Y,I]=sort(tsneregionpairD(n,:),'ascend');
 subset=[n]; i=2;
 while ((i<=nregions)&(length(subset)<(nnbs+1)))
  j=I(i);
  if ((validregions(j)==1)&(ismember(j,seedinds)==0))
  %if (validregions(j)==1)
   subset=[subset j];
  end
  i=i+1;
 end
 
 if (length(subset)<(nnbs+1))
  subset=[n]; i=2;
  while ((i<=nregions)&(length(subset)<(nnbs+1)))
   j=I(i);
   if (validregions(j)==1)
    subset=[subset j];
   end
   i=i+1;
  end
 end

 nbregioninds(m,:)=subset;
end


