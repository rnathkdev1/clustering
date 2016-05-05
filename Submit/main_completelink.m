%% HEIRARCHICAL COMPLETE LINKAGE WITH EUCLIDEAN DISTANCES
clc;
clear;

%% Loading the Required Data
XY=load('data.txt');

%Finding the euclidean distances between datapoints
distMat=pdist2(XY,XY,'euclidean');
m=length(distMat);

%This creates the heirarchy
heirMat=1:m;
heirMat=heirMat';

%% Begin clustering
[Z,c]=cluster_completeLink(heirMat,distMat,m);
Z_=linkage(XY,'complete','euclidean');
disp('Euclidean Distance:');
testTruth(c);
%% Plotting
figure(1)
subplot(1,2,1);
dendrogram(Z);
title('Output from My implementation')
subplot(1,2,2)
dendrogram(Z_);
title('Output from MATLAB builtin implementation')
suptitle('HEIRARCHICAL COMPLETE LINKAGE WITH EUCLIDEAN DISTANCES');

figure(3)
colormap winter;
scatter(XY(:,1),XY(:,2),20,c);
title('Clustered Results with Euclidean Distances');

%% HEIRARCHICAL COMPLETE LINKAGE WITH COSINE DISTANCES

clear;
%% Loading the Required Data
XY=load('data.txt');

%Finding the euclidean distances between datapoints
distMat=pdist2(XY,XY,'cosine');

m=length(distMat);
distMat(1:m+1:end)=0;
%This creates the heirarchy
heirMat=1:m;
heirMat=heirMat';

%% Begin clustering
[Z,c]=cluster_completeLink(heirMat,distMat,m);
Z_=linkage(XY,'complete','cosine');
disp('Cosine Distance:');
testTruth(c);

%% Plotting
figure(2)
dendrogram(Z);
title('HEIRARCHICAL COMPLETE LINKAGE WITH COSINE DISTANCES');
subplot(1,2,1);
dendrogram(Z);
title('Output from My implementation')
subplot(1,2,2)
dendrogram(Z_);
title('Output from MATLAB builtin implementation')
suptitle('HEIRARCHICAL COMPLETE LINKAGE WITH COSINE DISTANCES');

figure(4)
colormap winter;
scatter(XY(:,1),XY(:,2),20,c);
title('Clustered Results with Cosine Distances');