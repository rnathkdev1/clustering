%% HEIRARCHICAL AVERAGE LINKAGE WITH EUCLIDEAN DISTANCES
clc;
clear;
%% Loading the Required Data
XY=load('data.txt');

%Finding the euclidean distances between datapoints
distMat=pdist2(XY,XY,'euclidean');

%This is just a testing done to the code with a known heirarchy
% distMat=[NaN 10 12 8 7;
%          10 NaN 4 4 14;
%          12 4 NaN 6 16';
%          8 4 6 NaN 12;
%          7 14 16 12 NaN];
m=length(distMat);

%This creates the heirarchy
heirMat=1:m;
heirMat=heirMat';

%% Begin clustering
[Z,c]=cluster_averageLink(heirMat,distMat,m);
Z_=linkage(XY,'average','euclidean');
disp('Euclidean Distance:');
testTruth(c);
%% Plotting results
figure(1)
subplot(1,2,1);
dendrogram(Z);
title('Output from my implementation')
subplot(1,2,2)
dendrogram(Z_);
title('Output from MATLAB builtin implementation')
suptitle('HEIRARCHICAL AVERAGE LINKAGE WITH EUCLIDEAN DISTANCES');

figure(3)
colormap winter;
scatter(XY(:,1),XY(:,2),20,c);
title('Clustered Results with Euclidean Distances');

%% HEIRARCHICAL AVERAGE LINKAGE WITH COSINE DISTANCES

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
[Z,c]=cluster_averageLink(heirMat,distMat,m);
Z_=linkage(XY,'average','cosine');
disp('Cosine Distance:');
testTruth(c);

%% Plotting results
figure(2)
subplot(1,2,1);
dendrogram(Z);
title('Output from my implementation')
subplot(1,2,2)
dendrogram(Z_);
title('Output from MATLAB builtin implementation')
suptitle('HEIRARCHICAL AVERAGE LINKAGE WITH COSINE DISTANCES');

figure(4)
colormap winter;
scatter(XY(:,1),XY(:,2),20,c);
title('Clustered Results with Cosine Distances');