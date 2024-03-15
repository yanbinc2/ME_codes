% Implement k nearest neighbors algorithm to classify all images in the data.
% The distances between training and all images are pre-computed.
% The distance matrix is sparse.  Zero entries denote that the distances are too large to be considered.
% Speed up by sorting columns in trainingD.

function predlabels = kNN(nclasses, ntraining, traininginds, traininglabels, nimages, trainingD, nk)

% Find the k nearest neighbors of the training images.
% Label an image with the majority vote.

template=trainingD;
maxval=max(trainingD(:));
ss=find(trainingD>0);
template(ss)=maxval-template(ss);

[Y,I]=sort(template,'descend');
sinds=I(1:nk,:);
stlabels=traininglabels(sinds);
cnts=zeros(nclasses,nimages);
for n=1:nclasses
 submat=double(stlabels==n);
 vec=sum(submat);
 cnts(n,:)=vec;
end
clear template;

[M,I2]=max(cnts);
predlabels=I2;

