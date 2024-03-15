%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Enumerate shortest paths between each node pair in a undirected graph.
% Incrementally augment paths, and calculate shortest distances between node pairs.
% Stop augmenting paths when the shortest path distances between terminal nodes are known.
% Suppose the graph is connected.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [npaths, paths] = enumerate_shortest_paths(G);

nnodes=length(G(1,:));

D=-1*ones(nnodes,nnodes);

[a,b]=find(G>0);
c=find(a<b); a=a(c); b=b(c);
npaths=0; paths={};
for i=1:length(c)
 i1=a(i); i2=b(i);
 D(i1,i2)=1; D(i2,i1)=1;
 npaths=npaths+1; paths{npaths}=[i1 i2];
end
for i=1:nnodes
 D(i,i)=0;
end

active=ones(1,npaths);

flag=1;
while (flag==1)

 nnewpaths=0; newpaths={};
 newD=D; 
 newpathmat=zeros(1,nnodes);
 for n=1:npaths
  if (active(n)==1)
   path=paths{n};
   i1=path(1); i2=path(length(path));
   s1=find(G(i1,:)>0);
   s1=setdiff(s1,path);
   s2=find(G(i2,:)>0);
   s2=setdiff(s2,path);
   if (length(s1)>0)
    for i=1:length(s1)
     j=s1(i); vec=zeros(1,nnodes); vec([j path])=1;
     if ((D(j,i2)<0)&(ismember(vec,newpathmat,'rows')==0))
      nnewpaths=nnewpaths+1;
      newpaths{nnewpaths}=[j path];
      newD(j,i2)=length(path)+1;
      newD(i2,j)=length(path)+1;
      newpathmat(nnewpaths,:)=vec;
     end
    end
   end
   if (length(s2)>0)
    for i=1:length(s2)
     j=s2(i); vec=zeros(1,nnodes); vec([path j])=1;
     if ((D(i1,j)<0)&(ismember(vec,newpathmat,'rows')==0))
      nnewpaths=nnewpaths+1;
      newpaths{nnewpaths}=[path j];
      newD(i1,j)=length(path)+1;
      newD(j,i1)=length(path)+1;
      newpathmat(nnewpaths,:)=vec;
     end
    end
   end
  end
 end
 
 active=zeros(1,npaths);
 for n=1:nnewpaths
  npaths=npaths+1; paths{npaths}=newpaths{n};
  active(npaths)=1;
 end
 D=newD;

 if (sum(sum(D<0))==0)
  flag=0;
 end

end

