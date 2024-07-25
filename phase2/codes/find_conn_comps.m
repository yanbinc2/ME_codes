% Find all connected components of a graph.
% G is an undirected graph.
% Do not use the recursive function to label nodes.

function [nconncomps, conncomps] = find_conn_comps(nnodes, G)

% Label nodes in the graph until all nodes are labeled.
labeled=zeros(1,nnodes); flag=1;

% Label all the singletons.
nlabels=0;
for i=1:nnodes
 tmp=find(G(i,:)>0);
 tmp=setdiff(tmp,i);
 if (length(tmp)==0)
  nlabels=nlabels+1; labeled(i)=nlabels;
 end
end


% Iteratively label all nodes.

while (flag==1)

 % Check if all nodes are labeled.
 n=sum(labeled==0);

 % If yes, then return.
 if (n==0)
  flag=0;

 % Otherwise pick up an unlabeled node and iteratively label its neighbors.
 else
  ss=find(labeled==0);
  selectedinds=ss(1); nlabels=nlabels+1; flag2=1;
  while (flag2==1)
   labeled(selectedinds)=nlabels;
   subset=find(labeled==nlabels);
   vec=zeros(1,nnodes); vec(subset)=1;
   vec=vec*G;
   selectedinds=find(vec>0);
   selectedinds=setdiff(selectedinds,subset);
   if (length(selectedinds)==0)
    flag2=0;
   end
  end
  
 end
end


% Find the connected components from the labels.
nconncomps=nlabels; clear conncomps;
for i=1:nconncomps
 conncomps{i}.n=0;
 conncomps{i}.comps=[];
end

for i=1:nnodes
 j=labeled(i);
 conncomps{j}.n=conncomps{j}.n+1;
 conncomps{j}.comps(conncomps{j}.n)=i;
end

% Sort the connected components by size.
tmp=zeros(1,nconncomps);
for i=1:nconncomps
 tmp(i)=conncomps{i}.n;
end
[Y,I]=sort(tmp,'descend'); clear tmp2;
for i=1:nconncomps
 tmp2{i}=conncomps{I(i)}; 
end
conncomps=tmp2;

% Discard the connected components which do not have type 1 or type 2 nodes.
discard=zeros(1,nconncomps);
for i=1:nconncomps
 if (conncomps{i}.n<=0)
  discard(i)=1;
 end
end
tmp=find(discard==0);
for i=1:length(tmp)
 tmp2{i}=conncomps{tmp(i)};
end
nconncomps=length(tmp); conncomps=tmp2;

clear tmp tmp2 Y I discard;




  

