%% This script uses 2 SVMS and gives an accuracy of 96%
clc; clear;
% 1-walking 5-standing 6-laying

data=load('UCI HAR Dataset/train/X_train.txt');
labels=load('UCI HAR Dataset/train/y_train.txt');

test_data=load('UCI HAR Dataset/test/X_test.txt');
test_labels=load('UCI HAR Dataset/test/y_test.txt');
test_data=cat(1,test_data(test_labels==1,:),test_data(test_labels==5,:),test_data(test_labels==6,:));

data=cat(1,data(labels==1,:),data(labels==5,:),data(labels==6,:));
label_req=cat(1,labels(labels==1,:),labels(labels==5,:),labels(labels==6,:));

%% Performing PCA to the data
[coeff1]=pca(data,'NumComponents',2);
coeff1=data*coeff1;

test_label_req=cat(1,test_labels(test_labels==1,:),test_labels(test_labels==5,:),test_labels(test_labels==6,:));
coeff_test=pca(test_data,'NumComponents',2);
coeff_test=test_data*coeff_test;

% Displaying the training data
figure;
scatter(coeff1(:,1),coeff1(:,2),20,label_req);
figure;
scatter(coeff_test(:,1),coeff_test(:,2),20,test_label_req);

%% Training SVM to recognize walking
label_svm1=label_req;
label_svm1(label_svm1~=1)=0;
disp('Training SVM for walking...');
figure;
SVMModel_1 = svmtrain(coeff1,label_svm1,'ShowPlot',true,'kernel_function','polynomial','polyorder',3);
title('Training SVM for walking...or NOT walking');

% Extracting non walking labels
nextlabel=label_req(label_svm1==0);
coeff5=coeff1(label_svm1==0,:);

%% Training SVM to recognize standing IF NOT WALKING
label_svm5=nextlabel;
label_svm5(label_svm5~=5)=0;
disp('Training SVM for standing...');
figure;
SVMModel_5 = svmtrain(coeff5,label_svm5,'ShowPlot',true,'kernel_function','polynomial','polyorder',3);
title('If NOT walking training SVM for standing...');
% Extracting non standing non walking
nextlabel=label_req(label_req==6);
coeff6=coeff1(label_req==6,:);

%% Testing SVM 

% Testing for walking
label=svmclassify(SVMModel_1,coeff_test,'ShowPlot',true);
actualLabel1=test_label_req;
actualLabel1(actualLabel1~=1)=0;
% accuracy1=sum(actualLabel1==label);

% checking the indices of the classified labels that were zero
indexfor5=label==0;
coeff_test5=coeff_test(indexfor5,:);
newLabel5=test_label_req(indexfor5);

% If not walking then testing for standing
label5=svmclassify(SVMModel_5,coeff_test5,'ShowPlot',true);
actualLabel5=newLabel5;
actualLabel5(actualLabel5~=5)=6;

% Assigning this decision back to the original labels that were classified as zero
label(indexfor5)=label5;
label(label==0)=6;

%% Checking for accuracy

accuracy=sum(label==test_label_req)*100/length(test_label_req);
fprintf('Accuracy is %f percent\n',accuracy);

%% Generating Confusion Matrix
label(label==5)=2;
label(label==6)=3;
test_label_req(test_label_req==5)=2;
test_label_req(test_label_req==6)=3;
confusion=zeros(3,3);
for i=1:length(label)
    j=label(i);
    k=test_label_req(i);
    confusion(j,k)=confusion(j,k)+1;
end

confusion

%% Implementing K means to cluster the data 
idx=kmeans(data,3);

%Arranging data to clusters

idx(idx==idx(1))=9;
idx(idx==idx(end))=0;

idx(not(or(logical(idx==0), logical(idx==9))))=5;
idx(idx==9)=1;
idx(idx==0)=6;

accuracy_kmeans=sum(idx==label_req)*100/length(label_req);
fprintf('The accuracy using K means clustering is %f percent\n',accuracy_kmeans);
covariance=cov(coeff1)