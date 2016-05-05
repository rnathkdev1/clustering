function [Z,c]=cluster_averageLink(heirMat,distMat,m)

repeatMat=ones(m,1);
Z=[];
clusterInfo=1:m;
divide=ones(1,m);
clusterInfo=mat2cell(clusterInfo,1,divide);

for i=1:m
    distMat(distMat==0)=NaN;
    distVect=distMat(:);
    
    
    [leastDist,I]=min(distVect,[],'omitnan');
    
    distMat(isnan(distMat))=0;
    [col,row]=ind2sub(size(distMat),I);
    
    
    % average distance implies average heirarchical link
    % Accessing heirarchy matrix to determine num of elements for
    % taking the average
    w1=repeatMat(row);
    w2=repeatMat(col);
    sumw=w1+w2;
    distMat(row,:)=(w1*distMat(row,:)+w2*distMat(col,:))/sumw;
    
    if i<m-1
        clusterInfo{row}=cat(2,clusterInfo{row},clusterInfo{col});
        clusterInfo{col}=0;
    end
    
    distMat(row,row)=0;
    distMat(:,row)=distMat(row,:);
    distMat(:,col)=0;
    distMat(col,:)=0;
    
    % Generating the heirarchy
    Z=cat(1,Z,[heirMat(row),heirMat(col),leastDist]);
    repeatMat(row)=repeatMat(row)+1;
    repeatMat(col)=repeatMat(col)+1;
    heirMat(row)=m+i;
end

delIndex=[];
for i=1:m
    if isequal(clusterInfo{i},[0])
        delIndex=cat(1,delIndex,i);
    end
end

clusterInfo(delIndex)=[];
c=ones(1,m);
c(clusterInfo{1})=2;

Z(end,:)=[];
end

