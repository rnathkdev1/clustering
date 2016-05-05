%% HEIRARCHICAL SINGLE LINKAGE WITH EUCLIDEAN DISTANCES

clc;
clear;
%% Loading the Required Data
XY=load('data.txt');

%Finding the euclidean distances between datapoints
distMat=pdist2(XY,XY,'euclidean');

%This is just a testing done to the code with a known heirarchy
% distMat=[NaN  662 877 255 412 996;
%          662 NaN 295 468 268 400;
%          877 295 NaN 754 564 138;
%          255 468 754 NaN 219 869;
%          412 268 564 219 NaN 669;
%          996 400 138 869 669 NaN];

     
m=length(distMat);

%This creates the heirarchy
heirMat=1:m;
heirMat=heirMat';
% Y=[662 877 255 412 996 295 468 268 400 754 564 138 219 869 669];

%% Begin clustering
[Z,c]=cluster_singleLink(heirMat,distMat,m);
Z_=linkage(XY,'single','euclidean');
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
suptitle('HEIRARCHICAL SINGLE LINKAGE WITH EUCLIDEAN DISTANCES');


figure(3)
colormap winter;
scatter(XY(:,1),XY(:,2),20,c);
title('Clustered Results with Euclidean Distances');

%% HEIRARCHICAL SINGLE LINKAGE WITH COSINE DISTANCES

clear;
%% Loading the Required Data
XY=load('data.txt');

%Finding the euclidean distances between datapoints
distMat=pdist2(XY,XY,'cosine');
m=length(distMat);
distMat(1:m+1:end)=0;


%This is just a testing done to the code with a known heirarchy
% distMat=[NaN  662 877 255 412 996;
%          662 NaN 295 468 268 400;
%          877 295 NaN 754 564 138;
%          255 468 754 NaN 219 869;
%          412 268 564 219 NaN 669;
%          996 400 138 869 669 NaN];
% 
% Y=[662 877 255 412 996 295 468 268 400 754 564 138 219 869 669];

%This creates the heirarchy
heirMat=1:m;
heirMat=heirMat';

%% Begin clustering
[Z,c]=cluster_singleLink(heirMat,distMat,m);
Z_=linkage(XY,'single','cosine');
disp('Cosine Distance:');
testTruth(c);

%% Plot results
figure(2)
subplot(1,2,1);
dendrogram(Z);
title('Output from My implementation')
subplot(1,2,2)
dendrogram(Z_);
title('Output from MATLAB builtin implementation')
suptitle('HEIRARCHICAL SINGLE LINKAGE WITH COSINE DISTANCES');

figure(4)
colormap winter;
scatter(XY(:,1),XY(:,2),20,c);
title('Clustered Results with Cosine Distances');