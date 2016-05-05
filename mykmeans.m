clc; clear;
%% Using the K means algorithm by Euclidean Distances
XY=load('data.txt');
K=2;
% Choose K random centers
I=randperm(length(XY),K);
newCentroids=zeros(K,2);
centroids=XY(I,:);
kmeans(XY,2);
shiftCentroid=0;
while(1)
    % Calculating distances from each point to the centroids
    distMat=pdist2(XY,centroids,'euclidean');
    
    % Calculating the closest centroid
    [~,Label]=min(distMat,[],2);
    
    % Recomputing the centroids of the new set
    for i=1:K
        newCentroids(i,:)=mean(XY(Label==i));
    end
    
    %Calculating shift of centroid
    shiftCentroid=mean(sqrt(sum((centroids-newCentroids).^2,2)));
    
%     figure(1)
%     desc=strcat('Shift in the centroid is ',num2str(shiftCentroid));
%     colormap winter
%     scatter(XY(:,1),XY(:,2),20,Label);
%     hold on;
%     scatter(centroids(:,1),centroids(:,2),'red');
%     title(desc);
%     hold off;
    
    %Convergence condition
    if shiftCentroid<0.0001
        break;
    end
    centroids=newCentroids;
end

%% Plotting the results  
figure(2)
colormap winter;
scatter(XY(:,1),XY(:,2),20,Label);
title('Clustered Results with Euclidean Distances');
    
disp('Euclidean Distance:');
testTruth(Label');

%% For Cosine Distances, K Means 

clear;
%% Using the K means algorithm by Cosine Distances
XY=load('data.txt');
K=2;

% Choose K random centers
I=[15 30];
newCentroids=zeros(K,2);
centroids=XY(I,:);
shiftCentroid=0;

flag=0;
while(1)
    % Calculating distances from each point to the centroids
    distMat=pdist2(XY,centroids,'cosine');
    
    % Calculating the closest centroid
    [~,Label]=min(distMat,[],2);
    
%     figure(3)
%     desc=strcat('Shift in the centroid is ',num2str(shiftCentroid));
%     colormap winter
%     scatter(XY(:,1),XY(:,2),20,Label);
%     hold on;
%     scatter(centroids(:,1),centroids(:,2),'red');
%     title(desc);
%     hold off;

    % Recomputing the centroids of the new set
    for i=1:K
        members=find(Label==i);
        if length(members)>0
            newCentroids(i,:)=mean(XY(Label==i));
        end
        
    end
    
    %Calculating shift of centroid
    thisshiftCentroid=mean(sqrt(sum((centroids-newCentroids).^2,2)));
    if shiftCentroid==thisshiftCentroid
        flag=flag+1;
    else flag=0;
    end
    
    if flag==4 break;
    end
    
    %Convergence condition
    if shiftCentroid<0.001
        break;
    end
    
    centroids=newCentroids;
end

%% Plotting the results  
figure(4)
colormap winter;
scatter(XY(:,1),XY(:,2),20,Label);
title('Clustered Results with Cosine Distances');
    
disp('Cosine Distance:');
testTruth(Label');