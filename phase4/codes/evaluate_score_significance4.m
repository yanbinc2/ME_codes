% Evaluate the median score of a group of genes, and the significance of their distribution from a background distribution.
% Direction is determined by the mean score.
% Inputs: background scores, the data scores.
% Outputs: median score, significance of deviation.
% Define X as a random variable of correlation coefficients drawn from data.
% Define Y as a random variable of correlation coefficients drawn from the background distribution.
% Define Z=X-Y if X has positive deviation from background, and Z=Y-X vice versa.
% The significance score is Pr(Z>0)-Pr(Z<0).
% Difference from evaluate_score_significance.m: (1)Apply rejection sampling to draw random samples of the same size from X and Y distributions.  Evaluate Pr(Z>0) accordingly.
% Difference from evaluate_score_significance2.m: (1)Set dir=1, (2)Set minval and maxval to -1 and +1.
% Difference from evaluate_score_significance3.m in TCGA and single cell data analysis: (1)Specify minval and maxval in input.


function [medscore, pdiff] = evaluate_score_significance4(bgscores, datascores, nintervals, nrsamplesize, minvalinput, maxvalinput)

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
minval=quantile(Y,0.05); maxval=quantile(Y,0.95);

minval=quantile(Y,0.001); maxval=quantile(Y,0.999);
minval=-0.4; maxval=0.4;

minval=-1; maxval=1;

%minval1=quantile(X,0.001); maxval1=quantile(X,0.999);

%minval=-10; maxval=10;


minval=minvalinput;
maxval=maxvalinput;


valrange=[minval maxval];
rXs=rejection_sampling(X,valrange,nintervals,nrsamplesize);
rYs=rejection_sampling(Y,valrange,nintervals,nrsamplesize);


% Debug
%h1=hist(datascores,-1:0.01:1);
%h2=hist(bgscores,-1:0.01:1);
%r1=hist(rXs,-1:0.01:1);
%r2=hist(rYs,-1:0.01:1);
%clf; hold on;
%plot(-1:0.01:1,h1/sum(h1),'b');
%plot(-1:0.01:1,h2/sum(h2),'r');
%plot(-1:0.01:1,r1/sum(r1),'b-.');
%plot(-1:0.01:1,r2/sum(r2),'r-.');


% Debug
%fprintf('%d %d\n',sum(rXs>(rYs+0.01)),sum(rXs<(rYs-0.01)));


% Debug
%fprintf('%.4f %.4f\n',minval,maxval);


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



% Debug
%fprintf('%d %d\n',val1,val2);

pdiff=(val1-val2)/nrsamplesize;


