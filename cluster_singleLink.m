function [Z,c]=cluster_singleLink(heirMat,distMat,m)

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
   
    % Min distance implies heirarchical single link
    distMat(row,:)=min(distMat(row,:),distMat(col,:));
    
    if i<m-1
        clusterInfo{row}=cat(2,clusterInfo{row},clusterInfo{col});
        clusterInfo{col}=0;
    end
    
    distMat(:,row)=distMat(row,:);
    
    distMat(:,col)=0;
    distMat(col,:)=0;

    % Generating the heirarchy
    Z=cat(1,Z,[heirMat(row),heirMat(col),leastDist]);
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

