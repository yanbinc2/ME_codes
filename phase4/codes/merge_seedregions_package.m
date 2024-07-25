%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The function of merging seed regions.
% Inputs include (1)information about regions and seed regions, (2)prediction outcomes.
% Outputs are identities of merged seed regions.
% Merge seed regions according to the following types of information.
% (1) Consensus of CNN predictions on original seed regions over multiple runs.
% (2) Normalized neuron outputs of CNN predictions on original seed regions.
% (3) CNN prediction outcomes of merging each pair of seed regions.
% (4) CNN prediction outcomes of removing each seed region.
% (1) Extracts the CNN prediction outcomes which are consistent over 5 trials.
% (2) Identifies the seed region pairs (say 1 and 2) where on the images labeled to seed region 1 also have considerable neuron outputs on seed region 2.
% (3) Evaluates the leakage scores of seed region pairs (say 1 and 2) where some images originally labeled to seed region 2 are relabeled to the merged seed regions when merging seed region 1 and each other seed region.
% (4) Evaluates the replacement scores of seed region pairs (say 1 and 2) where some images originally labeled to seed region 1 are relabeled to seed region 2 when seed region 1 data are removed from the training data.
% Combine scores from (2), (3), (4) to a single score.
% Build a graph of pairwise relations between seed regions with top-ranking scores.
% Also build a graph of pairwise repulsive relations of seed regions where the p-values of merging scores are large.
% Obtain connected components of the graph of pairwise relations.
% Enumerate all partitions that respect repulsive relations.  Among them find the one with the smallest number of singletons and the smallest joint rank scores.
% Take caution to handle the cases when there are singleton seed regions which do not share underlying labels with other seed regions, or when seed regions with distinct underlying labels are similar.
% Can use marker information to identify singletons.
% Can either import replacementratiothre, localranksumpvalthre, cocontributionthre or determine their values from the distributions of the three types of scores.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [nmergeoutcomes, mergedclasslabels] = merge_seedregions_package(nseeds, seedinds, result_for_original, prob_for_original, combination_pairs, result_for_merge, result_for_removal, data, regiontraininglabels, usemarker, nprednumthre, pvalthre, ntoprankthre, reducedrankscorethre)  

% Yan-Bin 2024/7
filter=1;  %CIFAR10: filter=2

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set the fixed values for certain parameters.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

confusionratiothre=0.1;
sizethre=7;
smallclustersizethre=5;
jumpfoldthre=5.0;
gapfoldthre=2.0;
smallratiothre=0.1;

if (filter==2)
 pdiffthre=0.7;
 pdiffthre2=0.5;
 medvalthre=7.0;
 ninformativemarkersthre=5;
 ninformativemarkersthre2=40;
 cntdiffthre=8;
else
 pdiffthre=0;
 pdiffthre2=0;
 medvalthre=0;
 ninformativemarkersthre=0;
 ninformativemarkersthre2=0;
 cntdiffthre=0;
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate four types of scores regarding evidence of merging seed regions.
% (1) Consensus predictions over 5 CNN predictions.
% (2) Neuron output probabilities over seed regions.
% (3) Leakage scores derived from predictions of merging seed region pairs.
% (4) Replacement scores derived from predictions of removing seed regions.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

nclasses=nseeds; nimages=length(result_for_original(1,:));

% For each image find the most frequent prediction labels over 5 trials.
% Count the frequencies of the consensus labels of all images.
% Find the images whose consensus frequencies exceed a threshold value.

ntrials=length(result_for_original(:,1));
consensuslabels=zeros(1,nimages);
consensusfreqs=zeros(1,nimages);
for i=1:nimages
 v=result_for_original(:,i);
 vals=unique(v);
 ns=zeros(length(vals),1);
 for j=1:length(vals)
  ns(j)=sum(v==vals(j));
 end
 [Y,I]=sort(vals,'descend');
 vals=Y; ns=ns(I);
 consensuslabels(i)=vals(1);
 consensusfreqs(i)=ns(1);
end
consensuslabels=consensuslabels+1;

freqthre=ntrials;

%freqthre=3;

validpredictions=zeros(1,nimages);
validpredictions(find(consensusfreqs>=freqthre))=1;


% Calculate a cocontributions matrix.
% cocontributions(i,j) is the normalized neuron output of class j among the images which are labeled as class i.

cocontributions=zeros(nclasses,nclasses);
for i=1:nclasses
 ss=find((validpredictions==1)&(consensuslabels==i));
 weights=zeros(1,nclasses);
 for j=1:nclasses
  vals=prob_for_original(:,ss,j);
  weights(j)=sum(sum(vals));
 end
 weights=weights/sum(weights);
 cocontributions(i,:)=weights;
end



% Compare the predicted labels before merge and predicted labels after each merge.
% Identify the images which are not assigned to the two classes to be merged before the merge, and are assigned to the merged class after the merge.
% Count their distribution in the labels before the merge.
nmerges=length(combination_pairs(:,1));
mergeinds=combination_pairs+1;
mergedistributions=zeros(nmerges,nclasses);
for p=1:nmerges
 i1=mergeinds(p,1); i2=mergeinds(p,2);
 for n=1:nclasses
  if ((n~=i1)&(n~=i2))
   k=sum((validpredictions==1)&(consensuslabels==n)&(result_for_merge(p,:)==(i1-1)));
   mergedistributions(p,n)=k;
  end
 end
end


% Quantify the partner leakage level by a GSEA-like score.
% For an ordered class pair (i1,i2), quantify the leakage level that images previously assigned to class i2 are re-assigned to a new class by merging class i1 and all other classes.
% Extract the column in mergedistributions corresponding to i2.
% Select the rows in mergedistributions corresponding to merging i1 and other classes.
% In the extracted column vector, check whether high leakage levels are concentrated in the selected rows.
% Sort entries in the extracted column vector by a descending order.  Mark the entries belong to the selected rows.
% Build a GSEA random walk.  Start with zero and proceed along the sorted entries, increment by one if encountering a mark.
% Quantify devaition with the max deviation of the random walk from a straight line.

uvals=transpose(unique(mergedistributions(:)));
uvals=sort(uvals,'descend');
ranks=zeros(nmerges,nclasses);
curind=1;
for i=1:length(uvals)
 val=uvals(i);
 ss=find(mergedistributions==val);
 ranks(ss)=curind;
 curind=curind+length(ss);
end

localranksums=-1*ones(nclasses,nclasses);
localranksumspvals=ones(nclasses,nclasses); 
globalranksums=-1*ones(nclasses,nclasses);
nperms=100000;
for m=1:nclasses
 colvec=mergedistributions(:,m);
 scores=transpose(colvec);
 uscores=unique(scores);
 [Y,I]=sort(uscores,'descend');
 lranks=zeros(1,nmerges); currank=1;
 for i=1:length(uscores)
  val=Y(i); ss=find(scores==val);
  lranks(ss)=currank;
  currank=currank+length(ss);
 end
 granks=transpose(ranks(:,m));
 nullranksums=zeros(1,nperms);
 for i=1:nperms
  vv=randperm(nmerges);
  vals=lranks(vv(1:(nclasses-1)));
  nullranksums(i)=sum(vals);
 end
 for n=1:nclasses
  if (n~=m)
   rowinds=find((mergeinds(:,1)==n)|(mergeinds(:,2)==n));
   vec=lranks(rowinds);
   k=sum(vec);
   localranksums(m,n)=k;
   val=sum(nullranksums<=k)/nperms;
   localranksumspvals(m,n)=val;
   vec=granks(rowinds);
   k=sum(vec);
   globalranksums(m,n)=k;
  end
 end
end


% Calculate the replacement scores of each pair of seed regions from the predictions of removing each class.
% For each seed region pair i and j, calculate #(images assigned to class i in original predictions and assigned to class j when class i is removed)/#(images assigned to class i in original predictions).

replacementratios=zeros(nclasses,nclasses);
for i=1:nclasses
 for j=1:nclasses
  if (i==j)
   replacementratios(i,j)=1;
  else
   n1=sum((validpredictions==1)&(consensuslabels==i));
   n2=sum((validpredictions==1)&(consensuslabels==i)&(result_for_removal(i,:)==(j-1)));
   replacementratios(i,j)=n2/n1;
  end
 end
end


% Find the ranks of replacementratios.

replacementratioranks=zeros(nclasses,nclasses);
for i=1:nclasses
 vals=replacementratios(i,:);
 [Y,I]=sort(vals,'descend');
 for j=2:nclasses
  k=I(j); replacementratioranks(i,k)=j-1;
 end
 replacementratioranks(i,i)=nclasses;
end


% Calculate the confusion matrix of result_for_original on training data.
% Count the number of training images over trials that are assigned to each seed region.
C=zeros(nseeds,nseeds);
for n=1:nseeds
 i=seedinds(n);
 ss=find(regiontraininglabels==i);
 for j=1:ntrials
  for k=1:nseeds
   l=sum(result_for_original(j,ss)==(k-1));
   C(n,k)=C(n,k)+l;
  end
 end
end

% Add extra links in pG in terms of the confusion matrix.
% For each seed region, sort the row in the confusion matrix.
% If C(i,i) is not the highest, then include the pairs (i,j) where C(i,j)>C(i,i).
% If C(i,i) is the highest, then include the pairs (i,j) where C(i,j)>=(C(i,i)*confusionratiothre).
nextralinks=0; extralinks=zeros(1,2);
for i=1:nseeds
 [Y,I]=sort(C(i,:),'descend');
 k=find(I==i);
 if (k>1)
  for l=1:(k-1)
   j=I(l); 
   nextralinks=nextralinks+1;
   extralinks(nextralinks,:)=[i j];
  end
 else
  ss=find(C(i,:)>=(C(i,i)*confusionratiothre));
  ss=ss(find(ss~=i));
  for l=1:length(ss)
   j=ss(l);
   nextralinks=nextralinks+1;
   extralinks(nextralinks,:)=[i j];
  end
 end
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Identify the singleton seed regions whose dominant classes likely do not appear in other seed regions.
% If a seed region is a singleton, then it is likely to find a considerable number of markers that separate the target region and each of its neighboring regions.
% This part is valid when the flag usemarker=1.
% Distinguish between discarded and issingleton.
% A discarded member does not appear in the merged seed regions.
% A singleton member appears in the merged seed region but does not come with any other member.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

issingleton=zeros(1,nseeds);


% Count the number of images assigned to each class according to consensus predictions.
nprednums=zeros(1,nclasses);
for n=1:nclasses
 nprednums(n)=sum((validpredictions==1)&(consensuslabels==n));
end

discard=zeros(1,nclasses);
discard(find(nprednums<nprednumthre))=1;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Construct a graph of seed regions with putative mergeable relations.
% Two seed regions i and j are adjacent if one of the following conditions hold.
% (1) j is within ntoprankthre seed regions of i in terms of reducedrankscores and reciprocally.
% (2) i and j are in the same cluster according to spectral clustering using the reducedsumscores as edge weights.
% (3) i and j have high confusion ratio.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% For each seed region n, there are three types of rank scores.
% (1) Sort replacementratios(n,:) by a decreasing order and report the ranks.
% (2) Sort localranksumspvals(n,:) by an increasing order and report the ranks.
% (3) Sort cocontributions(n,:) by a decreasing order and report the ranks.
% reducedrankscores(n,:) are the sum of the three types of scores.
% Also report the ranks of replacementratios, localranksumspvals, cocontributions in fullrankscores.
% Find the best partner in terms of reducedrankscores(n,:).
% An invalid member does not have best partners but can be the best partners of other members.

bestpartners=zeros(nclasses,nclasses);
fullrankscores=zeros(3,nclasses,nclasses);
reducedrankscores=zeros(nclasses,nclasses);
for n=1:nclasses
 %if (sum((validpredictions==1)&(consensuslabels==n))>=100)
 if (discard(n)==0)
  rrs=zeros(3,nclasses);
  vals=replacementratios(n,:);
  [Y,I]=sort(vals,'descend');
  I=I(find(I~=n));
  curind=1; l=0;
  for i=1:length(I)
   j=I(i); 
   if ((i==1)|((i>1)&(vals(I(i))<vals(I(i-1)))))
    curind=curind+l; 
    if ((i>1)&(vals(I(i))<vals(I(i-1))))
     l=0;
    end
   end   
   rrs(1,j)=curind;
   l=l+1;
  end
  vals=localranksumspvals(n,:);
  [Y,I]=sort(vals,'ascend');
  I=I(find(I~=n)); 
  curind=1; l=0;
  for i=1:length(I)
   j=I(i); 
   if ((i==1)|((i>1)&(vals(I(i))>vals(I(i-1)))))
    curind=curind+l; 
    if ((i>1)&(vals(I(i))>vals(I(i-1))))
     l=0;
    end
   end   
   rrs(2,j)=curind;
   l=l+1;
  end 
  vals=cocontributions(n,:);
  [Y,I]=sort(vals,'descend');
  I=I(find(I~=n));
  curind=1; l=0;
  for i=1:length(I)
   j=I(i); 
   if ((i==1)|((i>1)&(vals(I(i))<vals(I(i-1)))))
    curind=curind+l; 
    if ((i>1)&(vals(I(i))<vals(I(i-1))))
     l=0;
    end
   end   
   rrs(3,j)=curind;
   l=l+1;
  end
  rrs(:,n)=nclasses+1;
  vals=sum(rrs);
  minval=min(vals);
  subset=find(vals==minval);
  bestpartners(n,subset)=1;
  fullrankscores(:,:,n)=rrs;
  reducedrankscores(n,:)=vals;
 end
end


% Also calculate the joint scores combining three pairwise scores.
% For each pair, calculate the sum of replacementratios, cocontributions, and -log10(localranksumspvals+1/nperms)/log10(nperms).
reducedsumscores=zeros(nclasses,nclasses);
for n=1:nclasses
 if (discard(n)==0)
  for i=1:nclasses
   val1=replacementratios(n,i);
   val2=cocontributions(n,i);
   val3=localranksumspvals(n,i);
   val3=-1*log10(val3+1/nperms)/log10(nperms);
   val=val1+val2+val3;
   reducedsumscores(n,i)=val;
  end
 end
end


% For each seed region, find other seed regions which are among the top ntoprankthre partners, and the reducedrankscores<=reducedrankscorethre.
% Exclude discarded seeds when finding the top ranking partners.
topscorepartners=zeros(nclasses,nclasses);
for n=1:nclasses
 if (discard(n)==0)
  %rset=find(issingleton==0);
  rset=find(discard==0);
  vals=reducedrankscores(n,rset);
  uvals=unique(vals);
  [Y,I]=sort(uvals);
  svals=Y(1:ntoprankthre); %svals=svals(find(svals<=reducedrankscorethre));
  [e,f]=ismember(vals,svals);
  ss=find(f>0); ss=rset(ss);
  topscorepartners(n,ss)=1;
 end
end


% Build the graph of putatively mergeable relations of seed regions.
% Make no isolated nodes except the discarded nodes.
% Connect i and j if i and j are among the topscorepartners of each other.
% If there are isolated but non-discarded nodes, then find the partner(s) with the best symmetric reducedrankscores and connect them.

pG=zeros(nseeds,nseeds);
for i=1:(nseeds-1)
 for j=(i+1):nseeds
  flag=1;
  if (discard(i)==1)
   flag=0;
  end
  if (discard(j)==1)
   flag=0;
  end
  if (topscorepartners(i,j)==0)
   flag=0;
  end
  if (topscorepartners(j,i)==0)
   flag=0;
  end
  if (flag==1)
   pG(i,j)=1; pG(j,i)=1;
  end
 end
end

for i=1:nextralinks
 i1=extralinks(i,1); i2=extralinks(i,2);
 if ((discard(i1)==0)&(discard(i2)==0)&(pG(i1,i2)==0))
  pG(i1,i2)=1; pG(i2,i1)=1;
 end
end

for i=1:nseeds
 if ((discard(i)==0)&(sum(pG(i,:)==1)==0))
  v=reducedrankscores(i,:)+transpose(reducedrankscores(:,i));
  [Y,I]=sort(v);
  I=I(find(I~=i));
  j=I(1);
  pG(i,j)=1; pG(j,i)=1;
 end
end




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate large components by applying spectral clustering to a graph of symmetrized reducedsumscores.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

W=zeros(nseeds,nseeds);
for i=1:nseeds
 W(i,i)=reducedsumscores(i,i);
end
for i=1:(nseeds-1)
 for j=(i+1):nseeds
  val=reducedsumscores(i,j)+reducedsumscores(j,i);
  %val=reducedrankscores(i,j)+reducedrankscores(j,i);
  %val=exp(-val);
  W(i,j)=val; W(j,i)=val;
 end
end


% Find the seed pairs with high weights and force them to be on the same side in spectral clustering.
% Choose the seed pairs i and j such that W(i,j)>max(W(i,i),W(j,j)).

nforcedpairs=0; forcedpairs=zeros(1,2);
for i=1:(nseeds-1)
 for j=(i+1):nseeds
  val=W(i,j); val1=W(i,i); val2=W(j,j);
  if ((val>=val1)&(val>=val2))
   nforcedpairs=nforcedpairs+1;
   forcedpairs(nforcedpairs,:)=[i j];
  end
 end
end


% Iteratively partition seed regions using spectral clustering.
% Previous approaches have several problems.  Thus start again.
% In each iteration, calculate the eigen decomposition of three least eigen values.
% Discard the smallest eigen value (0), and weigh the next two eigen vectors by their eigen values.
% From the two dimension projections, identify and separate small outlier clusters.
% The outlier clusters are (1) located at extreme ends (left, right, top, down), (2) very close in both dimensions, (3) far from the next seed regions in either dimension.
% If there are outlier clusters, then give them separate labels from the remaining seed regions.
% If there are no outlier clusters, then check whether there is a clear binary partition along the first dimension.
% A clear binary partition denotes the largest gap between seed regions along the first dimension is substantially higher than the second largest gap.

maxl=1; slabels=ones(1,nseeds); flag=1;
ss=find(discard==1); slabels(ss)=0;

while (flag==1)

 linds=[];
 for i=1:maxl
  k=sum(slabels==i);
  if (k>sizethre)
   linds=[linds i]; 
  end
 end


 if (length(linds)>0)

  i=linds(1); ss=find(slabels==i);   
  ss=ss(find(discard(ss)==0)); 
  sW=W(ss,ss);
  tmpvec=sum(sW); sD=diag(tmpvec);
  sE=inv(sqrt(sD));
  sL=sE*(sD-sW)*sE;
  neigs=3;
  [eigvecs,eigvals]=eigs(sL,neigs,eps);
  eigvals=diag(eigvals); neigvals=length(eigvals);
  abseigvals=abs(eigvals);
  [Y,I]=sort(abseigvals);
  iinds=I(2:3); iind=I(2);  
  weigvecs=eigvecs;
  for j=1:neigs
   weigvecs(:,j)=eigvecs(:,j)*eigvals(j);
  end
  X=eigvecs(:,iinds);
  D=pdist(X,'euclidean'); D=squareform(D);

  ns=length(ss); nds=0; ds=zeros(1,3);
  for j=1:(ns-1)
   for k=(j+1):ns
    nds=nds+1; ds(nds,:)=[j k D(j,k)];
   end
  end
  [Y,I]=sort(ds(:,3)); ds=ds(I,:);

  [Y1_1,I1_1]=sort(transpose(X(:,1)));
  gapds_1=zeros(1,ns-1);
  for j=1:(ns-1)
   gapds_1(j)=Y1_1(j+1)-Y1_1(j);
  end
  [Y2_1,I2_1]=sort(gapds_1,'descend');
  
  [Y1_2,I1_2]=sort(transpose(X(:,2)));
  gapds_2=zeros(1,ns-1);
  for j=1:(ns-1)
   gapds_2(j)=Y1_2(j+1)-Y1_2(j);
  end
  [Y2_2,I2_2]=sort(gapds_2,'descend');


  % Find outlier small clusters along dimensions 1 or 2.
  % Incrementally find small clusters of sizes 2, 3 and 4 where max intra-cluster distance is much smaller than min inter-cluster distance.
  
   
  % Find outlier small clusters along dimensions 1 or 2.
  % In each dimension, apply k-means with k=2 and set the initial centroids as the two extreme ends.  Pick the small cluster.
  % Check whether the max intra-cluster distance is much smaller than the min inter-cluster distance.

  noutliers=0; outlierlabels=zeros(1,ns);

  x1=X(I1_1(1),:); x2=X(I1_1(ns),:);
  idx=kmeans(X,2,'Start',[x1;x2]);
  n1=sum(idx==1); n2=sum(idx==2);
  if (n1<n2)
   smallinds=transpose(find(idx==1));
   largeinds=transpose(find(idx==2));
  elseif (n2<n1)
   smallinds=transpose(find(idx==2));
   largeinds=transpose(find(idx==1));
  else
   smallinds=[];
   largeinds=[];
  end
  
  if ((length(smallinds)>0)&(length(smallinds)<=smallclustersizethre))
   vals=D(smallinds,smallinds);
   vals=triu(vals,1);
   vals=vals(:); vals=vals(find(vals>0));
   intrads=transpose(vals);
   vals=D(smallinds,largeinds);
   vals=triu(vals,1);
   vals=vals(:); vals=vals(find(vals>0));
   interds=transpose(vals);
   val1=max(intrads); val2=min(interds);
   if (val2>=(val1*jumpfoldthre))
    noutliers=noutliers+1;
    %outlierlabels(ss(smallinds))=noutliers;
    outlierlabels(smallinds)=noutliers;
   end
  end
  
  x1=X(I1_2(1),:); x2=X(I1_2(ns),:);
  idx=kmeans(X,2,'Start',[x1;x2]);
  n1=sum(idx==1); n2=sum(idx==2);
  if (n1<n2)
   smallinds=transpose(find(idx==1));
   largeinds=transpose(find(idx==2));
  elseif (n2<n1)
   smallinds=transpose(find(idx==2));
   largeinds=transpose(find(idx==1));
  else
   smallinds=[];
   largeinds=[];
  end
  
  if ((length(smallinds)>0)&(length(smallinds)<=smallclustersizethre)&(sum(outlierlabels(smallinds)>0)==0))
   vals=D(smallinds,smallinds);
   vals=triu(vals,1);
   vals=vals(:); vals=vals(find(vals>0));
   intrads=transpose(vals);
   vals=D(smallinds,largeinds);
   vals=triu(vals,1);
   vals=vals(:); vals=vals(find(vals>0));
   interds=transpose(vals);
   val1=max(intrads); val2=min(interds);
   if (val2>=(val1*jumpfoldthre))
    noutliers=noutliers+1;
    %outlierlabels(ss(smallinds))=noutliers;
    outlierlabels(smallinds)=noutliers;
   end
  end
   
  % If there are small outlier clusters, then split them from ss and proceed.
  if (noutliers>0)
   for j=1:noutliers
    tt=find(outlierlabels==j);
    tt=ss(tt);
    maxl=maxl+1; slabels(tt)=maxl;
   end
   tt=find(outlierlabels==0);
   tt=ss(tt);
   maxl=maxl+1; slabels(tt)=maxl;
  end

  % If there are no small outlier clusters, then check whether the largest gap in dimension 1 is substantially bigger than the second largest gap.
  % If so, then split ss by the two sides of the largest gap.
  if ((noutliers==0)&(Y2_1(1)>=(Y2_1(2)*gapfoldthre)))
   k1=find(gapds_1>=Y2_1(1));
   s1=I1_1(1:k1); s2=I1_1((k1+1):ns);
   s1=ss(s1); s2=ss(s2);
   maxl=maxl+1; slabels(s1)=maxl;
   maxl=maxl+1; slabels(s2)=maxl;
  end
  
  % Check if the minimum difference between data points at positive and negative sides of dimension 1 <= smallratiothre * the max difference of dimension 1.
  % If so, then split ss by the two sides of the largest gap.
  spanval1=max(X(:,1))-min(X(:,1));
  spanval2=min(X(find(X(:,1)>=0),1))-max(X(find(X(:,1)<0),1));
  ratioval=spanval2/spanval1;  

  if ((noutliers==0)&(Y2_1(1)<(Y2_1(2)*gapfoldthre))&(ratioval<=smallratiothre))
   k1=find(gapds_1>=Y2_1(1));
   s1=I1_1(1:k1); s2=I1_1((k1+1):ns);
   s1=ss(s1); s2=ss(s2);
   maxl=maxl+1; slabels(s1)=maxl;
   maxl=maxl+1; slabels(s2)=maxl;
  end
  

  % If there are no small outlier clusters, and the largest gap in dimension 1 is not substantially bigger than the second largest gap, then use the sign of dimension 1 to split ss.
  %if ((noutliers==0)&(Y2_1(1)<(Y2_1(2)*gapfoldthre)))
  if ((noutliers==0)&(Y2_1(1)<(Y2_1(2)*gapfoldthre))&(ratioval>smallratiothre))
   k1=find(gapds_1>=Y2_1(1));
   s1=find(X(:,1)>=0); s2=find(X(:,1)<0);
   signs=zeros(1,length(ss));
   signs(s1)=1; signs(s2)=-1;
   changesigns=zeros(1,length(ss));
   for j=1:length(ss)
    if (ismember(j,s1)==1)
     dintras=D(j,setdiff(s1,j)); dinters=D(j,s2);    
    else
     dintras=D(j,setdiff(s2,j)); dinters=D(j,s1);
    end
    dintras=sort(dintras); dinters=sort(dinters);
    if (mean(dinters)<mean(dintras))
     changesigns(j)=1;
    end
   end
   for j=1:length(ss)
    if (changesigns(j)==1)
     signs(j)=signs(j)*(-1);
    end
   end
   s1=find(signs==1); s2=find(signs==-1);
   s1=ss(s1); s2=ss(s2);
   maxl=maxl+1; slabels(s1)=maxl;
   maxl=maxl+1; slabels(s2)=maxl;
  end


  % Find the forcepairs which have identical signs according to spectral clustering.
  % Find the forcepairs which have opposite signs according to spectral clustering.
  % Enumerate all combinations of signs for members of contradictory forcedpairs.
  % Find the combination which minimize the number of contradictory pairs.

  ln=maxl; lp=maxl-1;
  pset=find(slabels==lp); nset=find(slabels==ln);
  
  forcedpairconsistent=-1*ones(1,nforcedpairs);
  for j=1:nforcedpairs
   j1=forcedpairs(j,1); j2=forcedpairs(j,2);
   if (slabels(j1)==slabels(j2))
    forcedpairconsistent(j)=1;
   elseif (slabels(j1)~=slabels(j2))
    forcedpairconsistent(j)=0;
   end
  end

  if (sum(forcedpairconsistent==0)>0)

   sel=zeros(1,nseeds); 
   for j=1:nforcedpairs
    j1=forcedpairs(j,1); j2=forcedpairs(j,2);
    if (forcedpairconsistent(j)==0)
     sel(j1)=1; sel(j2)=1;
    end
   end
   sel(find(discard==1))=0;  

   selinds=find(sel==1); nsel=length(selinds);
   configs=zeros(1,nsel); nconfigs=0; vec=zeros(1,nsel); flag2=1;
   while (flag2==1)
    nconfigs=nconfigs+1; configs(nconfigs,:)=vec;
    tt=find(configs(nconfigs,:)==0); configs(nconfigs,tt)=-1;
    j=nsel; carry=1;
    while ((j>=1)&(carry==1))
     vec(j)=vec(j)+1;
     if (vec(j)>=2)
      vec(j)=0;
     else
      carry=0;
     end
     j=j-1;
    end
    if (sum(vec~=0)==0)
     flag2=0;
    end
   end

   curconfig=zeros(1,nsel);
   for j=1:nsel
    k=selinds(j); l=slabels(k);
    if (l==lp)
     curconfig(j)=1;
    elseif (l==ln)
     curconfig(j)=-1;
    end
   end

   ndiffs=zeros(1,nconfigs); ncontradicts=zeros(1,nconfigs); 
   for j=1:nconfigs
    vec=configs(j,:);
    ndiffs(j)=sum(vec~=curconfig);
    tmpslabels=slabels;
    for k=1:nsel
     p=selinds(k);
     if (vec(k)==1)
      tmpslabels(p)=lp;
     elseif (vec(k)==-1)
      tmpslabels(p)=ln;
     end
    end
    for k=1:nforcedpairs
     if (tmpslabels(forcedpairs(k,1))~=tmpslabels(forcedpairs(k,2)))
      ncontradicts(j)=ncontradicts(j)+1;
     end
    end
   end

   minval1=min(ncontradicts);
   cands=find(ncontradicts<=minval1);
   minval2=min(ndiffs(cands));
   cands=cands(find(ndiffs(cands)<=minval2));
   scores=zeros(1,length(cands));
   for j=1:length(cands)
    vec=configs(cands(j),:);
    for k=1:nsel
     l=vec(k);
     if (l>0)
      val=median(W(selinds(k),find(slabels==lp)));
     elseif (l<0)
      val=median(W(selinds(k),find(slabels==ln)));
     end    
     scores(j)=scores(j)+val;
    end
   end 

   candind=find(scores>=max(scores));
   candind=cands(candind(1));
   vec=configs(candind,:);

   for j=1:nsel
    k=selinds(j); l=vec(j);
    if (l>0)
     slabels(k)=lp;
    elseif (l<0)
     slabels(k)=ln;
    end
   end
 
  end  

 else
  flag=0;
 end

end


ncs=0; cs={};
minl=min(slabels(find(slabels>0))); maxl=max(slabels);

for l=minl:maxl
 ss=find(slabels==l);
 n=length(ss);
 if (n>0)
  ncs=ncs+1;
  cs{ncs}.n=n;
  cs{ncs}.comps=ss;
 end
end


% Augment pG such that each component is connected.
% Incrementally connecting the edges with the top reducedrankscores.
for p=1:ncs
 ss=cs{p}.comps;
 sG=pG(ss,ss);
 [nscs,scs]=find_conn_comps(cs{p}.n,sG);
 nlinks=0; links=zeros(1,4);
 for i=1:(cs{p}.n-1)
  for j=(i+1):cs{p}.n
   nlinks=nlinks+1;
   links(nlinks,:)=[i j reducedrankscores(ss(i),ss(j))+reducedrankscores(ss(j),ss(i)) 0];
   if (sG(i,j)==1)
    links(nlinks,4)=1; 
   end
  end
 end
 [YY,II]=sort(links(:,3),'ascend');
 links=links(II,:);
 while (nscs>1)
  tt=find(links(:,4)==0); 
  i=links(tt(1),1); j=links(tt(1),2);
  links(tt(1),4)=1;
  sG(i,j)=1; sG(j,i)=1;
  [nscs,scs]=find_conn_comps(cs{p}.n,sG);
 end
 pG(ss,ss)=sG;
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Build up repulsive relations of seed regions within components.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Within each component, identify the node pairs which should not be connected.
% Identify those node pairs from localranksumspvals.
% If localranksumspvals(i,j)>=pthre or localranksumspvals(j,i)>=pthre, then (i,j) should not be connected.


% Identify the node pairs which should not be connected.
% Identify those node pairs from localranksumspvals.
% If localranksumspvals(i,j)>=pthre or localranksumspvals(j,i)>=pthre, then (i,j) should not be connected.
nrepulsepairs=0; repulsepairs=zeros(1,2);
for i1=1:(nclasses-1)
 for i2=(i1+1):nclasses
  if ((localranksumspvals(i1,i2)>=pvalthre)|(localranksumspvals(i2,i1)>=pvalthre))
   nrepulsepairs=nrepulsepairs+1;
   repulsepairs(nrepulsepairs,:)=[i1 i2];
  end
 end
end

% Within each component, construct a graph of repulsive relations.
rG=zeros(nclasses,nclasses);
for i=1:nrepulsepairs
 i1=repulsepairs(i,1); 
 i2=repulsepairs(i,2); 
 rG(i1,i2)=1; rG(i2,i1)=1;
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% In each component, partition the seed regions to satisfy the following conditions:
% (1)Members in the same partitioned group do not have repulsive relations.
% (2)Some shortest paths connecting each pair of members in the same partitioned group should be in the same partitioned group.
% (3)Minimize the number of singletons in the partitions.
% (4)Among the valid partitions satisfying conditions 1-3, report a small number of valid partitions with high enrichment scores and intra-component scores.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% In each component, enumerate all shortest paths.
ntotalpaths=0; totalpaths={};
for n=1:ncs
 subset=cs{n}.comps;
 tG=pG(subset,subset);
 [npaths,paths]=enumerate_shortest_paths(tG);
 for i=1:npaths
  path=paths{i}; path=subset(path);
  ntotalpaths=ntotalpaths+1;
  totalpaths{ntotalpaths}=path;
 end
end


% Within each component, report a small number of valid partitions with high enrichment scores and intra-component scores.
% Do not partition the component if there are no repulsive pairs within.

subpartitions={};

%bestpartition=zeros(1,nclasses);

for ind=1:ncs

 subset=cs{ind}.comps;
 nsubset=length(subset);

 subpartitions{ind}.nsubset=nsubset;
 subpartitions{ind}.subset=subset;

 % Do not partition the subset if there are no repulsive pairs.
 if (sum(sum(rG(subset,subset)==1))==0)
  
  subpartitions{ind}.partitions=zeros(1,nsubset);


 % Otherwise partition the subset.
 else

  % Enumerate all partitions from 1 to nsubset components.
  maxn=min(nsubset,5);
  npartitions=1; 
  partitions=zeros(1,nsubset);
  for nc=1:maxn
   vec=zeros(1,nsubset);
   flag=1;
   while (flag==1)
    i=0; sat=1;
    if (max(vec)~=(nc-1))
     sat=0;
    end
    while ((i<=(nc-1))&(sat==1))
     ss=find(vec==i); ss=subset(ss); 
     if (sum(sum(rG(ss,ss)==1))>0)
      sat=0;
     end
     i=i+1;
    end

    if (length(unique(vec))<(max(vec)+1))
     sat=0;
    end

    if (sat==1)
     npartitions=npartitions+1;
     partitions(npartitions,:)=vec;
    end
    i=nsubset; carry=1;
    while ((i>0)&(carry==1))
     vec(i)=vec(i)+1;
     if (vec(i)==nc)
      vec(i)=0;
     else
      carry=0;
     end
     i=i-1;
    end
    if (sum(vec>0)==0)
     flag=0;
    end
   end
  end

  % Collapse all equivalent partitions by permuting subcomponent indices.
  nupartitions=1; upartitions=partitions(1,:);
  for n=2:npartitions
   vec=partitions(n,:);
   i=1; flag=1;
   while ((i<=nupartitions)&(flag==1))
    vec2=upartitions(i,:); k=max(vec2);
    if (max(vec)==k)
     j=0; flag2=1;
     while ((j<=k)&(flag2==1))
      ss=find(vec2==j);
      if (length(unique(vec(ss)))>1)
       flag2=0;
      end
      j=j+1;
     end
     if (flag2==1)
      flag=0;
     end
    end
    i=i+1;
   end
   if (flag==1)
    nupartitions=nupartitions+1;
    upartitions(nupartitions,:)=vec;
   end
  end

  % Impose additional constraints in terms of repulsion.
  % If two nodes i and j are in the same subcomponent, but on all shortest paths connecting them include repulsive node pairs, then the partition is not valid.
  % If two nodes i and j are in the same subcomponent, but on all shortest paths connecting them include nodes belonging to other components, then the partition is not valid.
  % If some component indices in the middle are missing in a partition, then it is not valid.

  isvalid=zeros(1,nupartitions);
  for n=1:nupartitions
   vec=upartitions(n,:);
   flag=1; 
   l=max(vec)-min(vec)+1;
   if (l>length(unique(vec)))
    flag=0;
   end
   i=1;
   while ((i<=nsubset)&(flag==1))
    j=i+1;
    while ((j<=nsubset)&(flag==1))
     if (vec(i)==vec(j))
      k=1; flag2=0;
      while ((k<=ntotalpaths)&(flag2==0))
       path=totalpaths{k};
       if (((path(1)==subset(i))&(path(length(path))==subset(j)))|((path(1)==subset(j))&(path(length(path))==subset(i))))
        flag2_1=0; flag2_2=0;          
        if (sum(sum(rG(path,path)==1))==0)
         flag2_1=1;
        end        
        [c,d]=ismember(path,subset);
        if ((sum(d==0)==0)&(sum(vec(d)~=vec(i))==0))       
         flag2_2=1;
        end          
        if ((flag2_1==1)&(flag2_2==1))
         flag2=1;
        end
       end
       k=k+1;
      end
      if (flag2==0)
       flag=0;
      end
     end
     j=j+1;
    end
   i=i+1;
   end
   if (flag==1)
    isvalid(n)=1;
   end
  end

  % Find valid partitions.
  nvalidpartitions=sum(isvalid==1);
  validpartitions=upartitions(find(isvalid==1),:);
  

  % Evaluate the score of each valid partition.
  % The idea is to reward the partitions where intra-component entries are concentrated in the top-ranking entries.
  % Sort non-diagonal entries of the scores with an increasing order. 
  % Build a random walk.  Increment with 1 if the entry is intra-component.
  % Compare the random walk with the control line of diagonal straight line.
  % Do not normalize the random walk because the number of top-ranking intra-component entries matters.
  % For instance, if one valid partition has 10 intra-component entries and they cover the top 10 entries, and another valid partition has 8 intra-component entries and they cover the top 8 entries, then the former is superior to the latter since the former covers more top-ranking entries.
  % The score is the area difference of the two curves.
  % Find the valid partitions that minimize the number of singletons.
  MM1=reducedrankscores(subset,subset);
  nentries1=0; entries1=zeros(1,3);
  for i=1:nsubset
   for j=1:nsubset
    if (i~=j)
     nentries1=nentries1+1; entries1(nentries1,:)=[i j MM1(i,j)];
    end
   end
  end
  [YY,II]=sort(entries1(:,3),'ascend');
  entries1=entries1(II,:); 
  rws1=zeros(nvalidpartitions,nentries1);
  cts1=zeros(nvalidpartitions,nentries1);
  hits1=zeros(nvalidpartitions,nentries1);
  rwscores1=zeros(1,nvalidpartitions);
  for n=1:nvalidpartitions
   vec=validpartitions(n,:);
   k=0; lastind=0;
   for p=1:nentries1
    i=entries1(p,1); j=entries1(p,2);
    if (vec(i)==vec(j))
     k=k+1; lastind=p;
     hits1(n,p)=1;
    end
   end
   if (k>0)
    rw=zeros(1,nentries1);
    ct=zeros(1,nentries1);
    for p=1:nentries1
     if (p>1)
      rw(p)=rw(p-1);
      ct(p)=ct(p-1);
     end
     i=entries1(p,1); j=entries1(p,2);
     if (vec(i)==vec(j))
      rw(p)=rw(p)+1;
     end
     ct(p)=ct(p)+k/nentries1;
    end
    rws1(n,:)=rw; cts1(n,:)=ct;
    a1=0; a2=0;
    a1=0.5*rw(1)*1; a0=0.5*ct(1)*1;
    for i=2:nentries1
     y1=rw(i-1); y2=rw(i);
     val=0.5*(y1+y2)*1; a1=a1+val;
     y1=ct(i-1); y2=ct(i);
     val=0.5*(y1+y2)*1; a0=a0+val;
    end
    rwscores1(n)=a1-a0;
   end
  end

  % Also apply the same method to reducedsumscores and calcualte the random walk scores.
  MM2=reducedsumscores(subset,subset);
  nentries2=0; entries2=zeros(1,3);
  for i=1:nsubset
   for j=1:nsubset
    if (i~=j)
     nentries2=nentries2+1; entries2(nentries2,:)=[i j MM2(i,j)];
    end
   end
  end
  [YY2,II2]=sort(entries2(:,3),'descend');
  entries2=entries2(II2,:); 
  rws2=zeros(nvalidpartitions,nentries2);
  cts2=zeros(nvalidpartitions,nentries2);
  hits2=zeros(nvalidpartitions,nentries2);
  rwscores2=zeros(1,nvalidpartitions);
  for n=1:nvalidpartitions
   vec=validpartitions(n,:);
   k=0; lastind=0;
   for p=1:nentries2
    i=entries2(p,1); j=entries2(p,2);
    if (vec(i)==vec(j))
     k=k+1; lastind=p;
     hits2(n,p)=1;
    end
   end
   if (k>0)
    rw2=zeros(1,nentries2);
    ct2=zeros(1,nentries2);
    for p=1:nentries2
     if (p>1)
      rw2(p)=rw2(p-1);
      ct2(p)=ct2(p-1);
     end
     i=entries2(p,1); j=entries2(p,2);
     if (vec(i)==vec(j))
      rw2(p)=rw2(p)+1;
     end
     ct2(p)=ct2(p)+k/nentries2;
    end
    rws2(n,:)=rw2; cts2(n,:)=ct2;
    a1=0; a2=0;
    a1=0.5*rw2(1)*1; a0=0.5*ct2(1)*1;
    for i=2:nentries2
     y1=rw2(i-1); y2=rw2(i);
     val=0.5*(y1+y2)*1; a1=a1+val;
     y1=ct2(i-1); y2=ct2(i);
     val=0.5*(y1+y2)*1; a0=a0+val;
    end
    rwscores2(n)=a1-a0;
   end
  end

  % rwscores3 is the sum of rwscores and rwscores2.
  rwscores3=rwscores1+rwscores2;


  % For each valid partition, calculate the sum of reducedrankscores of intra-component pairs.  
  intrareducedrankscoressums=zeros(1,nvalidpartitions);
  for n=1:nvalidpartitions
   vec=validpartitions(n,:);
   k=max(vec);
   for i=0:k
    ss=find(vec==i);
    vals=reducedrankscores(subset(ss),subset(ss));
    val=sum(sum(vals))-sum(diag(vals));
    intrareducedrankscoressums(n)=intrareducedrankscoressums(n)+val;
   end
  end

  % For each valid partition, calculate the sum of reducedsumscores of intra-component pairs.
  intrareducedsumscoressums=zeros(1,nvalidpartitions);
  for n=1:nvalidpartitions
   vec=validpartitions(n,:);
   k=max(vec);
   for i=0:k
    ss=find(vec==i);
    vals=reducedsumscores(subset(ss),subset(ss));
    val=sum(sum(vals))-sum(diag(vals));
    intrareducedsumscoressums(n)=intrareducedsumscoressums(n)+val;
   end
  end

  % Count the number of isolated members for each partition.
  nisolated=zeros(1,nvalidpartitions);
  for n=1:nvalidpartitions
   vec=validpartitions(n,:);
   k=max(vec); cnts=zeros(1,k+1);
   for i=0:k
    cnts(i+1)=sum(vec==i);
   end
   nisolated(n)=sum(cnts==1);
  end

  % Count the number of subcomponents for each partition.
  npcs=zeros(1,nvalidpartitions);
  for n=1:nvalidpartitions
   vec=validpartitions(n,:);
   npcs(n)=length(unique(vec));
  end

  % Extract unique component size distributions.
  nvecs=0; vecs=zeros(1,nsubset);
  for n=1:nvalidpartitions
   vec=validpartitions(n,:);
   k=max(vec);
   v=zeros(1,nsubset);
   for i=0:k
    l=sum(vec==i);
    if (l>0)
     v(l)=v(l)+1;
    end
   end
   if (ismember(v,vecs,'rows')==0)
    nvecs=nvecs+1; vecs(nvecs,:)=v;
   end
  end

  % Find the unique component size distribution index for each validpartition.
  cinds=zeros(1,nvalidpartitions);
  for n=1:nvalidpartitions
   vec=validpartitions(n,:);
   k=max(vec);
   v=zeros(1,nsubset);
   for i=0:k
    l=sum(vec==i);
    if (l>0)
     v(l)=v(l)+1;
    end
   end
   [a,b]=ismember(v,vecs,'rows');
   cinds(n)=b;
  end


  % Find the valid partitions with the smallest number of isolated members.
  isselected=zeros(1,nvalidpartitions);
  k=min(nisolated);
  isselected(find(nisolated==k))=1;

  % Among the selected partitions keep the ones with the smallest number of subcomponents.
  k=min(npcs(find(isselected==1)));
  isselected(find(npcs>k))=0;
  
  % Among the selected partitions subdivide them by the unique distributions of the sizes of the components.
  % For each component size distribution, report the partitions with any of the following (1)the highest rwscores3, (2)the highest intrareducedrankscoressums, (3)the lowest intrareducedsumscoressums.
  uinds=unique(cinds(find(isselected==1)));
  isselected2=zeros(1,nvalidpartitions);
  for i=1:length(uinds)
   j=uinds(i);
   ss=find((isselected==1)&(cinds==j));
   s1=ss(find(rwscores3(ss)>=max(rwscores3(ss))));
   s2=ss(find(intrareducedrankscoressums(ss)<=min(intrareducedrankscoressums(ss))));
   s3=ss(find(intrareducedsumscoressums(ss)>=max(intrareducedsumscoressums(ss))));
   isselected2(s1)=1; isselected2(s2)=1; isselected2(s3)=1;
  end
  isselected=isselected2;

  ss=find(isselected==1);
  subpartitions{ind}.partitions=validpartitions(ss,:);

  
 end

end



%%%%%%%%%%%%%%%%%%%%%%%%%
% Modify subpartitions.
%%%%%%%%%%%%%%%%%%%%%%%%%


nnewsubpartitions=0;
newsubpartitions={};


issingleton=zeros(1,nseeds);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Post process subpartitions for the scenario that markers of the data are used.
% Used for processing CIFAR10 data.
% For each marker, calculate the pdiff scores by comparing the data of each pair of seed regions.
% These pdiff scores indicate whether a marker has consistently high (or low) values in one seed region relative to another seed region.
% A marker is informative about a seed region if its pdiff scores relative to all other seed regions are high.
% For instance, if a seed region has dominant label of bird, then a marker of peacock has high values in this seed region and low values in other non-bird seed regions.
% If a seed region possesses many more informative markers than other seed regions, then this seed region is likely a singleton.
% For instance, if only one seed region has truck label, then it possesses informative markers of trucks but other seed regions do not have them.
% Calculate the pairwise seedmarkerpdiffs for each marker.
% For each seed region, minimize seedmarkerpdiffs over all other seed regions and obtain seedmarkerminpdiffs for each marker.
% For each marker, check whether it is an informative marker of a seed region with the following conditions.
% (1) The seedmarkerminpdiffs >= pdiffthre.
% (2) The median data values >= medvalthre.
% Count the number of informative markers for each seed region.
% If a seed region has the number of informative markers >= a threshold, then assign it as a singleton.
% For each subpartition, apply the following procedures.
% (1) Take out singletons.
% (2) If a subpartition subcomponent has only one seed region but is not a singleton, then discard this subcomponent.
% (3) Use the pdiff scores of seed regions within the component to identify the seed regions which are distinct from other seed regions of the component.  Discard the distinct seed regions if they are not singletons.
% (4) Redo subpartition after removing the aforementioned seed regions.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Consider the scenario to use markers.

if (usemarker==1)

 % Identify singletons according to the markers with high values specific to one seed region.
 
 % For each marker, calculate the pdiff scores by comparing the data of each pair of seed regions.
 seedmarkerpdiffs=zeros(nseeds,nseeds,ndim);
 for k=1:ndim 
  for i=1:nseeds
   ss1=find(regiontraininglabels==seedinds(i));
   vals1=data(ss1,k);
   for j=1:nseeds
    if (i~=j)
     ss2=find(regiontraininglabels==seedinds(j));
     vals2=data(ss2,k);
     [medscore,pdiff]=evaluate_score_significance(vals2,vals1,20,2000,-10,10);
     seedmarkerpdiffs(i,j,k)=pdiff;
    end
   end
  end
 end

 % For each marker, obtain the minimum pdiff score over all other seed regions.
 seedmarkerminpdiffs=zeros(ndim,nseeds);
 for k=1:ndim
  for i=1:nseeds
   %vec=seedmarkerscores(i,:,k);
   vec=seedmarkerpdiffs(i,:,k);
   vec(i)=2; val=min(vec);
   seedmarkerminpdiffs(k,i)=val;
  end
 end

 % Find the markers specific to one seed region.
 % For each marker, check whether the row corresponding to the seed region has high scores in all entries.
 % For each seed region, find the markers satisfying the following conditions:
 % (1) The seedmarkerminpdiffs >= pdiffthre.
 % (2) The median data values >= medvalthre.
 seedmarkerindicators=zeros(ndim,nseeds);
 for k=1:ndim
  for i=1:nseeds
   if (seedmarkerminpdiffs(k,i)>=pdiffthre)
    ss=find(regiontraininglabels==seedinds(i));
    vals=data(ss,k);
    val=median(vals);
    if (val>=medvalthre)
     seedmarkerindicators(k,i)=1;
    end
   end
  end
 end

 seedmarkerindicators2=zeros(ndim,nseeds);
 for k=1:ndim
  valthre=quantile(transpose(data(:,k)),0.95);
  for i=1:nseeds
   if (seedmarkerminpdiffs(k,i)>=pdiffthre)
    ss=find(regiontraininglabels==seedinds(i));
    vals=data(ss,k);
    val=median(vals);
    if (val>=valthre)   
     seedmarkerindicators2(k,i)=1;
    end
   end
  end
 end


 % For each seed region, calculate the median value of members of each marker.
 seedmarkervals=zeros(nseeds,ndim);
 for i=1:nseeds
  ss=find(regiontraininglabels==seedinds(i));
  for j=1:ndim
   vals=data(ss,j);
   seedmarkervals(i,j)=median(vals(:));
  end
 end
 
 vec=sum(seedmarkerindicators==1);
 ss=find(vec>=ninformativemarkersthre);
 issingleton(ss)=1;

 vec2=sum(seedmarkerindicators2==1);
 ss2=find(vec2>=ninformativemarkersthre2);
 issingleton(ss2)=1;
 

 % For each subpartition, identify and remove some seed regions and redo the subpartition.
  
 for ind=1:ncs

  nsubset=subpartitions{ind}.nsubset;
  subset=subpartitions{ind}.subset;
  validpartitions=subpartitions{ind}.partitions;
  nvalidpartitions=length(validpartitions(:,1));
  newvalidpartitions=validpartitions;

  % Find the seed regions in this component to be isolated.
  % An isolated seed region can be a singleton or a seed region which is very distinct from other members.
  % If a subpartitioned subcomponent has one seed region, it is also isolated.
  
  isolated=zeros(1,nsubset);

  % Identify singletons.
  ss=find(issingleton(subset)==1);
  isolated(ss)=1;
  
  % Identify the seed regions which form one-member subcomponents.
  vec=zeros(1,nsubset);  
  for n=1:nvalidpartitions
   vec2=validpartitions(n,:);
   k=max(vec2);
   for i=0:k
    ss=find(vec2==i);
    if (length(ss)==1)
     vec(ss)=vec(ss)+1;
    end
   end
  end
  ss=find(vec==nvalidpartitions);
  isolated(ss)=1;

  % Identify the seed regions which are distinct from other seed regions in the component.
  % A seed region is distinct from other members in the component if it has more informative markers confined to the component.
  minvals=zeros(ndim,nsubset);
  for k=1:ndim
   vals=seedmarkerpdiffs(subset,subset,k);
   for i=1:nsubset
    vec=vals(i,:);
    vec(i)=2;
    val=min(vec);
    minvals(k,i)=val;
   end
  end
  cnts=sum(minvals>=pdiffthre2);  
  [Y,I]=sort(cnts,'descend');
  d=Y(1)-Y(2);
  if ((d>=cntdiffthre)&(Y(2)<100))
   i=I(1); isolated(i)=1;
  end

    
  % Modify subpartitions if there are isolated seed regions.
  % For each valid partition, check whether the isolated seed region pair with another seed region to form a subcomponent.
  % If so, then reassign this single partner seed region to another subcomponent.
  % Avoid the subcomponents containing repulsive pairs with the partner.
  % Among the remaining subcomponents find the one with the lowest average of reducedrankscores.  If there are multiple such subcomponents, then assign to the one with the smallest size.
  newvalidpartitions=validpartitions;
  nnewvalidpartitions=nvalidpartitions;

  if (sum(isolated==1)>0)

   % In newvalidpartitions1, assign isolated seed regions to -1.       
   newvalidpartitions1=validpartitions;
   nnewvalidpartitions1=nvalidpartitions;
   for i=1:nsubset
    if (isolated(i)==1)
     newvalidpartitions1(:,i)=-1;
    end
   end

   % Find the seed regions which become single after removing isolated seed regions.
   nopartners=zeros(nnewvalidpartitions1,nsubset);
   for i=1:nnewvalidpartitions1
    vec=newvalidpartitions1(i,:);
    k=max(vec);
    for j=0:k
     ss=find(vec==j);
     if (length(ss)==1)
      nopartners(i,ss)=1;
     end
    end
   end

   aggnopartners=zeros(1,nsubset);
   for i=1:nsubset
    aggnopartners(i)=sum(nopartners(:,i)==1);    
   end

   
   cands=find(aggnopartners>0);
   ncands=length(cands);


   % Keep the newvalidpartitions1 which do not generate single-member subcomponents after removing isolated seed regions.
   nnewvalidpartitions2=0;
   newvalidpartitions2=zeros(1,nsubset);
   for i=1:nnewvalidpartitions1
    vec=newvalidpartitions1(i,:);
    if (sum(nopartners(i,:)==1)==0)
     nnewvalidpartitions2=nnewvalidpartitions2+1;
     newvalidpartitions2(nnewvalidpartitions2,:)=vec;
    end
   end

   % If there are no such partitions, then reassign the candidate seed regions to the subcomponents with no repulsive relations and min average rankscores.
   if (nnewvalidpartitions2==0)    
    srG=rG(subset,subset);
    ssG=reducedrankscores(subset,subset);                
    newvalidpartitions2=zeros(1,nsubset);
    nnewvalidpartitions2=0;
    for i=1:nnewvalidpartitions1
     vec=newvalidpartitions1(i,:);
     vec2=vec; vec2(cands)=-2;
     for j=1:ncands
      k=cands(j); l=vec(k);
      if ((l>=0)&(sum(vec==l)==1))
       scores=nseeds*10*ones(1,max(vec)+1); 
       cnts=zeros(1,max(vec)+1); 
       for m=0:max(vec)
	if (m~=l)
 	 tt=find(vec==m); cnts(m+1)=length(tt);
         if (sum(srG(k,tt)==1)==0)
	  vals=ssG(k,tt); val=mean(vals);
          scores(m+1)=val; 
         end
        end
       end
       minval=min(scores);
       if (minval<(nseeds*10))
        ss=find(scores<=minval);
        m=min(cnts(ss));
        ss=ss(find(cnts(ss)==m));
        m=ss(1); vec2(k)=m-1;
       end
      end 
     end
     if (sum(vec2==-2)==0)
      nnewvalidpartitions2=nnewvalidpartitions2+1;
      newvalidpartitions2(nnewvalidpartitions2,:)=vec2;
     end
    end  
   end      

   % If no such partition exists, then discard the candidate seed regions.
   if (nnewvalidpartitions2==0)
    nnewvalidpartitions2=nnewvalidpartitions1;
    newvalidpartitions2=newvalidpartitions1;
    newvalidpartitions2(:,cands)=-1;
   end

   % Simplify newvalidpartitions2.
   % For each partition, make labels consecutive numbers, and place the larger components the smaller numbers.
   % Extract unique partitions.
   nnewvalidpartitions3=nnewvalidpartitions2;
   newvalidpartitions3=newvalidpartitions2;
   for i=1:nnewvalidpartitions3
    vec=newvalidpartitions2(i,:);
    uvals=unique(vec); uvals=uvals(find(uvals>=0)); nuvals=length(uvals);
    cnts=zeros(1,nuvals);
    for j=1:nuvals
     cnts(j)=sum(vec==uvals(j));
    end
    [Y,I]=sort(cnts,'descend');
    vec2=vec;
    for j=1:nuvals
     k=uvals(I(j)); ss=find(vec==k); vec2(ss)=j-1;
    end
    newvalidpartitions3(i,:)=vec2;
   end
   newvalidpartitions3=unique(newvalidpartitions3,'rows');
   nnewvalidpartitions3=length(newvalidpartitions3(:,1));

   newvalidpartitions=newvalidpartitions3;
   nnewvalidpartitions=nnewvalidpartitions3;


   % Discard the seed regions which are assigned to -1 in all newvalidpartitions.
   if (nnewvalidpartitions==1)
    vec=double(newvalidpartitions<0);
   else
    vec=sum(newvalidpartitions<0);
   end
   ss=find(vec==nnewvalidpartitions);
   ss=subset(ss);
   ss=ss(find(issingleton(ss)==0));
   discard(ss)=1;

  end   


  % Re-generate valid partitions to cope with singletons and discarded seed regions.
  % Remove discarded seed regions from the subset.
  % Make singletons standlone subcomponents.
  newsubset=subset; 
  newsubset=newsubset(find(discard(subset)==0));
  nnewsubset=length(newsubset);
  [a,sinds]=ismember(newsubset,subset);
  nnewvalidpartitions2=0;
  newvalidpartitions2=zeros(1,nnewsubset);

  for n=1:nnewvalidpartitions
   vec=newvalidpartitions(n,sinds);
   k=max(vec);
   for i=1:nnewsubset
    j=newsubset(i);
    if (issingleton(j)==1)
     k=k+1; vec(i)=k;
    end
   end
   if (sum(vec<0)==0)
    nnewvalidpartitions2=nnewvalidpartitions2+1;
    newvalidpartitions2(nnewvalidpartitions2,:)=vec;
   end
  end

  nnewsubpartitions=nnewsubpartitions+1;
  newsubpartitions{nnewsubpartitions}.nsubset=nnewsubset;
  newsubpartitions{nnewsubpartitions}.subset=newsubset;
  newsubpartitions{nnewsubpartitions}.validpartitions=newvalidpartitions2;

 end

end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Post process subpartitions for the scenarios that markers of the data are not used.
% Used for processing MNIST and MNIST_Aug data.
% Identify the singletons which are clustered together with other seed regions in the previous steps.
% (1) For each subcomponent from the partitions, find the member seed regions which have poor reducedrankscores with any other member seed region.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Consider the scenario to not use markers.

if (usemarker==0)

 for ind=1:ncs

  nsubset=subpartitions{ind}.nsubset;
  subset=subpartitions{ind}.subset;
  validpartitions=subpartitions{ind}.partitions;
  nvalidpartitions=length(validpartitions(:,1));
  newvalidpartitions=validpartitions;


  % Find the singletons which are distinct from any other subcomponent members in all validpartitions.
  % For each subcomponent, enumerate all valid subsubpartitions.
  % Choose the one to minimize the number of isolated components and maximize scores.

  for q=1:nvalidpartitions   
   maxl=max(validpartitions(q,:));
   for l=0:maxl
    subsubset=find(validpartitions(q,:)==l);
    nsubsubset=length(subsubset);
    sG=reducedrankscores(subset(subsubset),subset(subsubset));
    for i=1:nsubsubset
     for j=1:nsubsubset
      if (i==j)
       sG(i,j)=1;
      else
       if (sG(i,j)<=reducedrankscorethre)
	sG(i,j)=1;
       else
	sG(i,j)=0;
       end
      end
     end
    end

    if ((nsubsubset>0)&(sum(sum(sG==0))>0))    
     maxn=nsubsubset;
     npartitions=0;
     partitions=zeros(1,nsubsubset);
     for nc=1:maxn
      vec=zeros(1,nsubsubset);
      flag=1;
      while (flag==1)
       i=0; sat=1;
       if (max(vec)~=(nc-1))
        sat=0;
       end
       while ((i<=(nc-1))&(sat==1))
        ss=find(vec==i); 
        if (sum(sum(sG(ss,ss)==0))>0)
         sat=0;
        end
        i=i+1;
       end

       if (length(unique(vec))<(max(vec)+1))
	sat=0;
       end

       if (sat==1)
        npartitions=npartitions+1;
        partitions(npartitions,:)=vec;
       end
       i=nsubsubset; carry=1;
       while ((i>0)&(carry==1))
        vec(i)=vec(i)+1;
        if (vec(i)==nc)
         vec(i)=0;
        else
         carry=0;
        end
        i=i-1;
       end
       if (sum(vec>0)==0)
        flag=0;
       end
      end
     end
         
     % Collapse all equivalent partitions by permuting subcomponent indices.
     nupartitions=1; upartitions=partitions(1,:);
     for n=2:npartitions
      vec=partitions(n,:);
      i=1; flag=1;
      while ((i<=nupartitions)&(flag==1))
       vec2=upartitions(i,:); k=max(vec2);
       if (max(vec)==k)
        j=0; flag2=1;
        while ((j<=k)&(flag2==1))
         ss=find(vec2==j);
         if (length(unique(vec(ss)))>1)
          flag2=0;
         end
         j=j+1;
        end
        if (flag2==1)
         flag=0;
        end
       end
       i=i+1;
      end
      if (flag==1)
       nupartitions=nupartitions+1;
       upartitions(nupartitions,:)=vec;
      end
     end
     
     % Calculate the sum of intra-subsubcomponent reducedsumscores of all valid partitions.
     scores=zeros(1,nupartitions);
     for n=1:nupartitions
      vec=upartitions(n,:);
      submaxl=max(vec);
      for l=0:submaxl
       ss=find(vec==l);
       ss=subset(subsubset(ss));
       val=sum(sum(reducedsumscores(ss,ss)))-sum(diag(reducedsumscores(ss,ss)));
       scores(n)=scores(n)+val;
      end
     end

     % Find the upartition with the max score.
     k=find(scores>=max(scores)); k=k(1);
     vec=upartitions(k,:);
     m=max(newvalidpartitions(q,:));
     vec=vec+m+1; 
     newvalidpartitions(q,subsubset)=vec;

    end
   end
  end

  newnewvalidpartitions=newvalidpartitions;
  for q=1:nvalidpartitions
   ulabels=unique(newvalidpartitions(q,:));
   for l=1:length(ulabels)
    ss=find(newvalidpartitions(q,:)==ulabels(l));
    newnewvalidpartitions(q,ss)=l-1;
   end
  end

  nnewsubpartitions=nnewsubpartitions+1;
  newsubpartitions{nnewsubpartitions}.nsubset=nsubset;
  newsubpartitions{nnewsubpartitions}.subset=subset;
  newsubpartitions{nnewsubpartitions}.validpartitions=newnewvalidpartitions;

 end


 % Identify singletons.
 % A singleton forms a single subcomponent in all newsubpartitions.
 for ind=1:ncs
  nsubset=newsubpartitions{ind}.nsubset;
  subset=newsubpartitions{ind}.subset;
  validpartitions=newsubpartitions{ind}.validpartitions;
  nvalidpartitions=length(validpartitions(:,1));
  singleflags=zeros(nvalidpartitions,nsubset);
  for n=1:nvalidpartitions
   for l=0:max(validpartitions(n,:))
    ss=find(validpartitions(n,:)==l);
    if (length(ss)==1)
     singleflags(n,ss)=1;
    end
   end
  end
  accsingleflags=zeros(1,nsubset);
  for i=1:nsubset
   accsingleflags(i)=sum(singleflags(:,i)==1);
  end
  accsingleflags=double(accsingleflags==nvalidpartitions);
  issingleton(subset(find(accsingleflags==1)))=1;
 end


 % Filter out the spurious singletons.
 % A spurious singleton satisfies the following conditions.
 % (1) It splits out in generating newsubpartitions.
 % (2) It has small reducedrankscores with at least another seed region in both directions (reducedrankscores(i,:) and reducedrankscores(:,i)).
 isspurioussingleton=zeros(1,nseeds);
 for n=1:nseeds
  if (issingleton(n)==1)
   i=1; j=0;
   while ((i<=ncs)&(j==0))
    if (ismember(n,subpartitions{i}.subset)==1)
     j=i;
    end
    i=i+1;
   end
   k=find(subpartitions{j}.subset==n);
   i=1; l=0;
   while ((i<=length(subpartitions{j}.partitions(:,1)))&(l==0))
    m=subpartitions{j}.partitions(i,k);
    if (sum(subpartitions{j}.partitions(i,:)==m)>1)
     l=i;
    end
    i=i+1;
   end
   if (l>0)
    v1=reducedrankscores(n,:);
    v2=transpose(reducedrankscores(:,n));
    if (sum((v1<=reducedrankscorethre)&(v2<=reducedrankscorethre))>0)
     isspurioussingleton(n)=1;
    end
   end
  end
 end

 % Discard the spurious singletons and update them in newsubpartitions.
 sinds=find(isspurioussingleton==1);
 discard(sinds)=1;
 for ind=1:nnewsubpartitions
  nsubset=newsubpartitions{ind}.nsubset;
  subset=newsubpartitions{ind}.subset;
  validpartitions=newsubpartitions{ind}.validpartitions;
  if (sum(isspurioussingleton(subset)==1)>0)
   ss=find(isspurioussingleton(subset)==0);
   newsubset=subset(ss);
   nnewsubset=length(newsubset);
   newvalidpartitions=validpartitions(:,ss);
   vecs=unique(newvalidpartitions,'rows');
   if (length(newvalidpartitions(:,1))>length(vecs(:,1)))
    newvalidpartitions=vecs;
   end
   for i=1:length(newvalidpartitions(:,1))
    vec=newvalidpartitions(i,:);
    vec2=vec;
    k=0; maxl=max(vec);
    for j=0:maxl
     if (sum(vec==j)>0)
      tt=find(vec==j); vec2(tt)=k; k=k+1;
     end
    end
    newvalidpartitions(i,:)=vec2;
   end
   newsubpartitions{ind}.nsubset=nnewsubset;
   newsubpartitions{ind}.subset=newsubset;
   newsubpartitions{ind}.validpartitions=newvalidpartitions;
  end
 end

end

 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate one or multiple merged seed classes outcomes from newsubpartitions.
% Enumerate all combinations of newsubpartitions for all components.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Count the number of valid partitions in each newsubpartition.
vcnts=zeros(1,ncs);
for ind=1:ncs
 k=length(newsubpartitions{ind}.validpartitions(:,1));
 vcnts(ind)=k;
end


% Enumerate all possible states of newsubpartitions.
vstates=zeros(1,ncs); nvstates=0; vstate=zeros(1,ncs); flag=1;
while (flag==1)
 nvstates=nvstates+1;
 vstates(nvstates,:)=vstate;
 i=ncs; carry=1;
 while ((i>=1)&(carry==1))
  vstate(i)=vstate(i)+1;
  if (vstate(i)==vcnts(i))
   vstate(i)=0; 
  else
   carry=0;
  end
  i=i-1;
 end
 if (sum(vstate>0)==0)
  flag=0;
 end
end


% Generate all merged classes outcomes.
nmergeoutcomes=nvstates;
mergedclasslabels=zeros(nmergeoutcomes,nclasses);
for p=1:nmergeoutcomes
 vstate=vstates(p,:);
 classlabels=zeros(1,nseeds);
 ncclasses=0;
 for ind=1:ncs
  nsubset=newsubpartitions{ind}.nsubset;
  subset=newsubpartitions{ind}.subset;
  validpartitions=newsubpartitions{ind}.validpartitions;
  m=vstate(ind)+1;
  validpartition=validpartitions(m,:);
  minl=min(validpartition); maxl=max(validpartition); 
  for i=minl:maxl
   if (sum(validpartition==i)>0)
    ss=find(validpartition==i);
    ss=subset(ss); 
    ncclasses=ncclasses+1;
    classlabels(ss)=ncclasses;
   end
  end
 end
 mergedclasslabels(p,:)=classlabels;
end


