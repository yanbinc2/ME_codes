% Evaluate the median score of a group of genes, and the significance of their distribution from a background distribution.
% Direction is determined by the mean score.
% Inputs: background scores, the data scores.
% Outputs: median score, significance of deviation.

function [medscore, pdiff] = evaluate_score_significance(bgscores, datascores, nintervals, nrsamplesize, minvalinput, maxvalinput)

n0=length(bgscores); n1=length(datascores);

% Calculate the median score.
X=datascores; medscore=median(X);

% Calculate the significance of deviation.
% Randomly sample bgscores ntrials times with the same dimension as datascores.
% Skip when encounter invalid entries.

if (medscore>=0)
 dir=1;
else
 dir=-1;
end 

dir=1;

% Generate nrsamplesize random samples from X and Y distributions.
Y=bgscores;

minval=minvalinput;
maxval=maxvalinput;

valrange=[minval maxval];
rXs=rejection_sampling(X,valrange,nintervals,nrsamplesize);
rYs=rejection_sampling(Y,valrange,nintervals,nrsamplesize);

if (dir>0)
 val1=sum(rXs>rYs); val2=sum(rXs<rYs);
elseif (dir<0)
 val1=sum(rXs<rYs); val2=sum(rXs>rYs);
else
 val1=0; val2=0;
end

if (dir>0)
 val1=sum(rXs>(rYs+0.05)); val2=sum(rXs<(rYs-0.05));
elseif (dir<0)
 val1=sum(rXs<(rYs-0.05)); val2=sum(rXs>(rYs+0.05));
else
 val1=0; val2=0;
end


pdiff=(val1-val2)/nrsamplesize;


