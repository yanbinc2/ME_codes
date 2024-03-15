% Implement rejection sampling from an empirical distribution of 1D data.
% Inputs: 1D observation data, data value range, number of intervals in the value range, size of the random sample.
% Outputs: randomly sampled data drawn from the empirical distribution.
% Obtain the empirical PDF.
% Randomly draw numbers from a uniform distribution over the range.
% Accept a number a with probability Pr(X<=a) and Pr(X<=a) from empirical distribution.
% Continue until generating the sample of the designated size.


function rsampledpoints = rejection_sampling(empdatapoints, valrange, nintervals, nrsamplesize)

% Calculate the PDF of empirical data.
% Subdivide the valrange into nintervals intervals.  Calculate the PDF for each interval value.
minx=min(valrange); maxx=max(valrange); dx=(maxx-minx)/nintervals;
intervals=minx:dx:maxx;
pdfvals=zeros(1,nintervals+1);
for n=1:nintervals
 val1=intervals(n); val2=intervals(n+1);
 k=sum((empdatapoints>=val1)&(empdatapoints<=val2));
 pdfvals(n)=k/length(empdatapoints);
end

maxp=max(pdfvals);

% Apply rejection sampling to generate random sampled points.
rsampledpoints=[];
while (length(rsampledpoints)<nrsamplesize)
 rvals=rand(1,nrsamplesize);
 rvals=(maxx-minx)*rvals+minx;
 %rvals=rand(1,nrsamplesize)*dx+minx;
 qs=round((rvals-minx)/dx)+1;
 qs(find(qs>(nintervals+1)))=nintervals+1;
 p0s=pdfvals(qs); 
 p1s=rand(1,nrsamplesize)*maxp;  
 ss=find(p1s<=p0s);
 rsampledpoints=[rsampledpoints rvals(ss)];
end

rsampledpoints=rsampledpoints(1:nrsamplesize);

% Handle the case when empdatapoints are concentrated in a value.
if (length(unique(empdatapoints))==1)
 val=unique(empdatapoints);
 rsampledpoints=val*ones(1,nrsamplesize);
end




