%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate various information about markers in the data.
% The information is used for splitting the similar seed regions with different labels when merging seed regions.
% Inputs are data, regions, seed region indices.
% Outputs are the binary trees of markes and seeds from hierarchical clustering, permutation orders of markers and seeds induced from hierarchical clustering, partition of markers into groups, the indicators of high values of each marker groups in each seed region.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [markerranks, seedpas, seeddescendants, Pseeds, markerpas, markerdescendants, markernoderanges, Pmarkers, subimageinds, subimagebds, seedregionmarkergroupvals, seedmarkergroupindicators, npartitionnodes, partitionnodes] = generate_markerinfo(data, nregions, regiontraininglabels, nseeds, seedinds)


% Obtain the dimension of the data.
[nimages,ndim]=size(data);


% For each marker, obtain the ranks of images in terms of data values.
markerranks=zeros(nimages,ndim);
for i=1:nimages
 [Y,I]=sort(data(i,:),'descend');
 for j=1:ndim
  k=I(j); markerranks(i,k)=j;
 end
end


% Retrieve the image indices of members in the seed regions and boundaries between seed regions.
subimageinds=[]; subimagebds=[];
for m=1:nseeds
 n=seedinds(m);
 ss=find(regiontraininglabels==n);
 subimageinds=[subimageinds ss];
 subimagebds=[subimagebds length(subimageinds)];
end

nsubimages=length(subimageinds);



% Apply hierarchical clustering to markers and seed regions.
% For markers, cluster the data of images in the seed regions.
% For seed regions, generate the mean data for each seed region and cluster the mean data.
% Find the descendant leaves under each internal node of the trees.
% Obtain the permutation orders of markers and seed regions.

seedregiondata=data(subimageinds,:);
seedregionmeandata=zeros(nseeds,ndim);
for n=1:nseeds
 ss=find(regiontraininglabels==seedinds(n));
 vals=data(ss,:);
 vec=mean(vals);
 seedregionmeandata(n,:)=vec;
end

Zmarkers=linkage(transpose(seedregiondata),'ward','euclidean');
[Hmarkers,Tmarkers,Pmarkers]=dendrogram(Zmarkers,0);

Zseeds=linkage(seedregionmeandata,'ward','euclidean');
[Hseeds,Tseeds,Pseeds]=dendrogram(Zseeds,0);


% Build binary trees from Zmarkers and Zseeds.

nseednodes=max(max(Zseeds(:,1:2)))+1;
seedpas=zeros(1,nseednodes);
for m=1:length(Zseeds(:,1))
 i1=Zseeds(m,1); i2=Zseeds(m,2); i0=nseeds+m;
 seedpas(i1)=i0; seedpas(i2)=i0;
end
seeddescendants=zeros(nseednodes,nseeds);
for n=1:nseeds
 k=n;
 while (seedpas(k)>0)
  seeddescendants(k,n)=1;
  k=seedpas(k);
 end
end
seeddescendants(nseednodes,:)=1;

nmarkernodes=max(max(Zmarkers(:,1:2)))+1;
markerpas=zeros(1,nmarkernodes);
for m=1:length(Zmarkers(:,1))
 i1=Zmarkers(m,1); i2=Zmarkers(m,2); i0=ndim+m;
 markerpas(i1)=i0; markerpas(i2)=i0;
end
markerdescendants=zeros(nmarkernodes,ndim);
for n=1:ndim
 k=n;
 while (markerpas(k)>0)
  markerdescendants(k,n)=1;
  k=markerpas(k);
 end
end
markerdescendants(nmarkernodes,:)=1;


% For each marker tree node, find the range of sorted markers.
markernoderanges=zeros(nmarkernodes,4);
for p=1:nmarkernodes
 qq=find(markerdescendants(p,:)==1);
 [a,b]=ismember(qq,Pmarkers);
 i1=min(b); i2=max(b);
 markernoderanges(p,:)=[p i2-i1+1 i1 i2];
end


% Generate combinations of marker groups and seed groups.

% Identify the groups of markers (internal nodes of the marker tree) with >= 5 markers.
% For each such group of markers, find the groups of seeds (internal nodes in the seed tree) which have high values.
% Criteria: the difference of the median values between the entries of the two seed groups is big enough (>=3.0).

nthre=5;
markernodegroups=[];
for p=1:nmarkernodes
 if (sum(markerdescendants(p,:)==1)>=nthre)
  markernodegroups=[markernodegroups p];
 end
end
nmarkernodegroups=length(markernodegroups);


% markergroupseedgroupvals have 12 columns.
% column 1: markergroupnode index.
% column 2: seedgroupnode index.
% column 3: number of markers in the group.
% column 4: start index of the sorted markers.
% column 5: end index of the sorted markers.
% column 6: number of seeds in the group.
% column 7: number of images in the group.
% column 8: number of seeds in the complementary group.
% column 9: number of images in the complementary group.
% column 10: median value of the data in the targeted seed region images.
% column 11: median value of the data in the non-targeted seed region images.
% column 12: column 8 value - column 9 value.
% The complementary group contains the seeds of the sibling.

markergroupseedgroupvals=zeros(1,12);
nn=0;
for ind=1:nmarkernodegroups
 m=markernodegroups(ind);
 sinds3=find(markerdescendants(m,:)==1);
 for n=1:(nseednodes-1)
  pa=seedpas(n);
  sib=setdiff(find(seedpas==pa),n); 
  sinds1=find(seeddescendants(n,:)==1);  
  sinds2=[]; 
  for i=1:length(sinds1)
   j=sinds1(i); j=seedinds(j); ss=find(regiontraininglabels==j);
   sinds2=[sinds2 ss];
  end  
  %csinds1=setdiff(1:nseeds,sinds1);
  csinds1=find(seeddescendants(sib,:)==1);
  csinds2=[];
  for i=1:length(csinds1)
   j=csinds1(i); j=seedinds(j); ss=find(regiontraininglabels==j);
   csinds2=[csinds2 ss];
  end
  vals1=data(sinds2,sinds3);
  vals2=data(csinds2,sinds3);
  val1=median(vals1(:)); val2=median(vals2(:));
  nn=nn+1;
  markergroupseedgroupvals(nn,:)=[m n length(sinds3) markernoderanges(m,3:4) length(sinds1) length(sinds2) length(csinds1) length(csinds2) val1 val2 val1-val2];
 end
end


% Apply the top down approach to find separating markers.
% Start from the root of the marker tree and descend.
% For each node, check whether its two children have very distinct data patterns.
% Count the markergroupseedgroupvals rows with high difference scores and separate sufficient number of seed regions.
% Stop if the current marker group does not have the qualified rows, or the qualified rows span the whole range.

nmarkersthre=20; val1thre=2.0; diffvalthre=1.0; splitnthre=5;
nmarkersthre=20; val1thre=0.5; diffvalthre=1.0; splitnthre=5;
partitionnodes=[];
curinds=nmarkernodes;

while (length(curinds)>0)

 % Add the active nodes with <= nmarkersthre markers to partitionnodes, and proceed to the first active node with > nmarkersthre markers.
 i=1; curind=0; vec=zeros(1,length(curinds));
 while (i<=length(curinds))
  j=curinds(i);
  k=sum(markerdescendants(j,:)==1);
  if (k<=nmarkersthre)
   partitionnodes=[partitionnodes j];
   vec(i)=1;
  else
   if (curind==0)
    curind=j;
    vec(i)=1;
   end
  end
  i=i+1;
 end

 curinds=curinds(find(vec==0));
 
 % Find the two children of the current node.
 children=find(markerpas==curind);

 % Find the qualified markergroupseedgroupvals contained in curind.
 startind=markernoderanges(curind,3); endind=markernoderanges(curind,4);
 sinds=find((markergroupseedgroupvals(:,10)>=val1thre)&(markergroupseedgroupvals(:,12)>=diffvalthre)&((markergroupseedgroupvals(:,6)+markergroupseedgroupvals(:,8))>=splitnthre)&(markergroupseedgroupvals(:,4)>=startind)&(markergroupseedgroupvals(:,5)<=endind)&((markergroupseedgroupvals(:,5)-markergroupseedgroupvals(:,4)+1)>=nmarkersthre));

 % If there are no qualified markergroupseedgroupvals or the qualified markergroupseedgroupvals span the whole range, then stop splitting.
 % If there are pairs of qualified markergroupseedgroupvals of the same seed group and two marker groups for the two children, then they are not maximal.  Disable them.
 vec=zeros(1,length(sinds));
 for i=1:length(sinds)
  j=sinds(i);
  i1=markergroupseedgroupvals(j,4);
  i2=markergroupseedgroupvals(j,5);
  if (~((i1<=startind)&(i2>=endind)))
   vec(i)=1;
  end
 end
 uinds=unique(markergroupseedgroupvals(sinds(find(vec==1)),2));
 for i=1:length(uinds)
  j=uinds(i);
  ss=find(markergroupseedgroupvals(sinds,2)==j);
  ss=ss(find(vec(ss)==1));
  t1=find(markergroupseedgroupvals(sinds(ss),1)==children(1));
  t2=find(markergroupseedgroupvals(sinds(ss),1)==children(2));
  if ((length(t1)==1)&(length(t2)==1))
   t1=ss(t1); t2=ss(t2);
   vec([t1 t2])=0;
  end
 end

 newsinds=[];
 for i=1:length(sinds)
  j=sinds(i);
  if (vec(i)==1)
   newsinds=[newsinds j];
  end
 end
 sinds=newsinds;

 % If there are qualified markergroupseedgroupvals rows, then the current node is heterogeneous.  Continue splitting.
 if (length(sinds)>0)
  curinds=[curinds children];

 % Otherwise stop splitting and add the current node to partitionnodes.
 else
  partitionnodes=[partitionnodes curind];
 end

end


% Sort partitionnodes.
[Y,I]=sort(markernoderanges(partitionnodes,3));
partitionnodes=partitionnodes(I);
npartitionnodes=length(partitionnodes);


% For each set of partitioned markers, find the seed regions with high values.
seedregionmarkergroupvals=zeros(nseeds,npartitionnodes);
for n=1:nseeds
 p=seedinds(n);
 ss=find(regiontraininglabels==p);
 for i=1:npartitionnodes
  i1=markernoderanges(partitionnodes(i),3);
  i2=markernoderanges(partitionnodes(i),4);
  vals=data(ss,Pmarkers(i1:i2));
  seedregionmarkergroupvals(n,i)=median(vals(:));
 end
end



% Obtain several background distributions of high and low values.
% The markers in CIFAR10 are divided into two groups.
% Group 1 has dense high values.  Group 2 has sparse high values.
% In group 1 markers, divide seed regions into two groups: machines and animals.
% Machines and animals have different sets of markers.
% Choose the high markers of machines and high markers of animals.
% Use the combinations of (machines,machine markers) and (animals,animal markers) to extract the background high values.
% Generate two sets of background low values.
% In the marker group of high values (machine and animal markers), use the combinations of (machines,animal markers) and (animals,machine markers) to extract the background low values 1.
% In the marker group of low values, subdivide it into four subgroups.  Choose the two subgroups with the lowest average values to extract the background low values 2.

% Sort subimageinds according to Pseeds.
sortedsubimageinds=[]; sortedsubimagebds=[];
for m=1:nseeds
 n=Pseeds(m);
 p=seedinds(n);
 ss=find(regiontraininglabels==p);
 sortedsubimageinds=[sortedsubimageinds ss];
 sortedsubimagebds=[sortedsubimagebds length(sortedsubimageinds)];
end
  

% Find the high and low marker groups and their sorted indices.
children=find(markerpas==nmarkernodes);
mkinds1=find(markerdescendants(children(1),:)==1);
mkinds2=find(markerdescendants(children(2),:)==1);
vals1=data(sortedsubimageinds,mkinds1);
vals2=data(sortedsubimageinds,mkinds2);
val1=mean(vals1(:)); val2=mean(vals2(:));
if (val1>val2)
 highmarkerind=children(1);
 lowmarkerind=children(2);
 highmkinds=mkinds1;
 lowmkinds=mkinds2;
else
 highmarkerind=children(2);
 lowmarkerind=children(1);
 highmkinds=mkinds2;
 lowmkinds=mkinds1;
end

[a,b]=ismember(highmkinds,Pmarkers);
ishigh=zeros(1,ndim);
ishigh(b)=1;



% Find the machine and animal seed region groups and their sorted image indices.
children=find(seedpas==nseednodes);
qq=find(seeddescendants(children(1),:)==1);
[a,sinds1]=ismember(qq,Pseeds); sinds1=sort(sinds1);
qq=find(seeddescendants(children(2),:)==1);
[a,sinds2]=ismember(qq,Pseeds); sinds2=sort(sinds2);
if (ismember(1,sinds1)==1)
 bd=max(sinds1);
else
 bd=max(sinds2);
end
ibd=sortedsubimagebds(bd);
sinds1=sortedsubimageinds(1:ibd);
sinds=sortedsubimageinds((ibd+1):nsubimages);


% Subdivide highmkinds into two sub marker groups.
children=find(markerpas==highmarkerind);
mkinds3=find(markerdescendants(children(1),:)==1);
mkinds4=find(markerdescendants(children(2),:)==1);


% Generate four groups of (seed region group, marker group) combinations.
% Calculate their mean values.

combvals=zeros(2,2);
tmpvals={};
vals=data(sortedsubimageinds(1:ibd),mkinds3);
combvals(1,1)=mean(vals(:)); tmpvals{1}{1}=vals;
vals=data(sortedsubimageinds(1:ibd),mkinds4);
combvals(1,2)=mean(vals(:)); tmpvals{1}{2}=vals;
vals=data(sortedsubimageinds((ibd+1):nsubimages),mkinds3);
combvals(2,1)=mean(vals(:)); tmpvals{2}{1}=vals;
vals=data(sortedsubimageinds((ibd+1):nsubimages),mkinds4);
combvals(2,2)=mean(vals(:)); tmpvals{2}{2}=vals;

lookups=[
1 1 1
2 2 1
3 1 2
4 2 2];

[Y,I]=sort(combvals(:),'descend');

highvals=[]; 
for m=1:2
 n=I(m); 
 i=lookups(n,2); j=lookups(n,3);
 vals=tmpvals{i}{j}; vals=transpose(vals(:));
 highvals=[highvals vals];
end

lowvals1=[];
for m=3:4
 n=I(m); 
 i=lookups(n,2); j=lookups(n,3);
 vals=tmpvals{i}{j}; vals=transpose(vals(:));
 lowvals1=[lowvals1 vals];
end


% Subdivide lowmkinds into four sub marker groups.
children=find(markerpas==lowmarkerind);
grandchildren1=find(markerpas==children(1));
grandchildren2=find(markerpas==children(2));
grandchildren=[grandchildren1 grandchildren2];

% Choose the two grandchildren with the lowest mean values as lowvals2.
tmpvals={}; mvals=[];
for m=1:4
 n=grandchildren(m);
 ss=find(markerdescendants(n,:)==1);
 vals=data(subimageinds,ss);
 vals=transpose(vals(:));
 tmpvals{m}=vals;
 mvals(m)=mean(vals);
end

[Y,I]=sort(mvals);

lowvals2=[tmpvals{I(1)} tmpvals{I(2)}];




% For each marker group, quantize the values of each seed region into a binary value.
% Get the distribution of the seed region + marker group values.
% Calculate the pdiff scores wrt high and low values.
% For the markers in the high group, use a higher background lowvals2.
% Choose the distribution with better fit.

seedmarkergroupindicators=-1*ones(nseeds,npartitionnodes);
for n=1:npartitionnodes
  
 m=partitionnodes(n);
 i1=markernoderanges(m,3);
 i2=markernoderanges(m,4);
 
 for i=1:nseeds
  
  j=seedinds(i);
  ss=find(regiontraininglabels==j);
  vals=data(ss,Pmarkers(i1:i2));
  vals=vals(:);

  if (sum(ishigh(i1:i2)==1)==0)  
   [medscore,pdifflow1]=evaluate_score_significance4(lowvals2,vals,100,10000,-10,10);
   [medscore,pdifflow2]=evaluate_score_significance4(vals,lowvals2,100,10000,-10,10);
   [medscore,pdiffhigh1]=evaluate_score_significance4(highvals,vals,100,10000,-10,10);
   [medscore,pdiffhigh2]=evaluate_score_significance4(vals,highvals,100,10000,-10,10);

  else  
   [medscore,pdifflow1]=evaluate_score_significance4(lowvals1,vals,100,10000,-10,10);
   [medscore,pdifflow2]=evaluate_score_significance4(vals,lowvals1,100,10000,-10,10);  
   [medscore,pdiffhigh1]=evaluate_score_significance4(highvals,vals,100,10000,-10,10);
   [medscore,pdiffhigh2]=evaluate_score_significance4(vals,highvals,100,10000,-10,10);

  end

  % If vals distribution is on the right of highvals, then pdiffhigh1>pdiffhigh2 and pdifflow1>pdifflow2.  Assign it to the high level.
  if ((pdiffhigh1>pdiffhigh2)&(pdifflow1>pdifflow2))
   seedmarkergroupindicators(i,n)=1;

  % If vals distribution is on the left of lowvals, then pdifflow2>pdifflow1 and pdiffhigh2>pdiffhigh1.  Assign it to the low level.
  elseif ((pdiffhigh2>pdiffhigh1)&(pdifflow2>pdifflow1))
   seedmarkergroupindicators(i,n)=0;

  % If vals distribution is between lowvals and highvals, then pdiffhigh2>=pdiffhigh1 and pdifflow1>=pdifflow2.  Assign it to the closer state by comparing pdifflow1 and pdiffhigh2.
  % Assign a slighly bias against high value.
  % If pdiffhigh2<(pdifflow1-0.05), then assign it to high value.
  elseif ((pdifflow1>=pdifflow2)&(pdiffhigh2>=pdiffhigh1))
   %if (pdiffhigh2<pdifflow1)
   if (pdiffhigh2<(pdifflow1-0.05))
    seedmarkergroupindicators(i,n)=1;
   else
    seedmarkergroupindicators(i,n)=0;
   end
  end
 
  % Debug
  fprintf('%d %d-%d (%d %d) %f\n',n,i1,i2,i,seedmarkergroupindicators(i,n),seedregionmarkergroupvals(i,n));
   

 end
end

