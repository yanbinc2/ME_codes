%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Demarcate the center zone of data points in a Euclidean space.
% Input is the coordinates of data points.
% Output is the data point indices belonging to the center zone.
% Calculate the distribution of pairwise distances.
% Find the mode of the distance distribution.
% Check whether the mode frequency is substantially higher than the background frequencies.
% If there is no mode, then there is no center zone.
% Use the mode as cutoff to generate a graph.
% Find the largest connected component and treat it as a center zone.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [nhighvals, centerzonemembers] = demarcate_center_zones(X)

% Get the distribution of pairwise distances.
[nxnodes,nxdim]=size(X);
D=pdist(X,'euclidean');
D=squareform(D);
sD=triu(D);
vals=sD(find(sD>0));
vals=transpose(vals);
maxval=max(vals);
dl=maxval/100; ll=round(maxval/dl);
intvals=0:dl:(ll*dl);
hh=hist(vals,intvals);
nn=sum(hh);

% Find the peaks in the distance distribution.
% If the max value does not appear in the peaks, then reduce the min peak width to 1.
[pks,locs]=findpeaks(hh,'MinPeakHeight',nn*0.005,'MinPeakWidth',3);
maxhh=max(hh);
if (ismember(maxhh,pks)==0)
 [pks,locs]=findpeaks(hh,'MinPeakHeight',nn*0.005,'MinPeakWidth',1);
end

% If the distance distribution is not concentrated in a few peaks, then there is no center zone.
[YY,II]=sort(hh,'descend');

nhighvals=sum(YY>=(YY(1)*0.5));


% Find the peak at the smallest distance.
% Use this peak distance as threshold to generate an undirected graph.
dthre=intvals(min(locs));
tG=double(D<=dthre);
[ncs,cs]=find_conn_comps(nxnodes,tG);

centerzonemembers=cs{1}.comps;






