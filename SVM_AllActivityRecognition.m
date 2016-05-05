%% This script uses 5 SVMS to classify 6 groups USING 2 PRINCIPAL COMPONENTS

clc; clear;

% 1 WALKING
% 2 WALKING_UPSTAIRS
% 3 WALKING_DOWNSTAIRS
% 4 SITTING
% 5 STANDING
% 6 LAYING

data=load('UCI HAR Dataset/train/X_train.txt');
labels=load('UCI HAR Dataset/train/y_train.txt');

test_data=load('UCI HAR Dataset/test/X_test.txt');
test_labels=load('UCI HAR Dataset/test/y_test.txt');

%% Performing PCA to the data and normalizing
[coeff]=pca(data,'NumComponents',2);
coeff=data*coeff;

coeff_test=pca(test_data,'NumComponents',2);
coeff_test=test_data*coeff_test;
%% Training to recognize Walking or not walking

label_svm_1=labels;
label_svm_1(label_svm_1==4)=0; label_svm_1(label_svm_1==5)=0; label_svm_1(label_svm_1==6)=0;
label_svm_1(label_svm_1==1)=1; label_svm_1(label_svm_1==2)=1; label_svm_1(label_svm_1==3)=1;

disp('Training SVM for walking...or not walking...');
figure;
SVMModel_split = svmtrain(coeff,label_svm_1,'ShowPlot',true,'kernel_function','polynomial','polyorder',1);


%% Training to recognize walking types:

% Extracting the walking data
label_walk=cat(1,labels(labels==1),labels(labels==2),labels(labels==3));
coeff_walk=cat(1,coeff(labels==1,:),coeff(labels==2,:),coeff(labels==3,:));

figure;
scatter(coeff_walk(:,1),coeff_walk(:,2),20,label_walk);
title('Walking data');

% Training SVM to recognize walking in level
label_svm_walklevel=label_walk;
label_svm_walklevel(label_svm_walklevel==2)=0; label_svm_walklevel(label_svm_walklevel==3)=0;

disp('Training SVM for walking in level...');
figure;
SVMModel_walklevel = svmtrain(coeff_walk,label_svm_walklevel,'ShowPlot',true,'kernel_function','polynomial','polyorder',1);
title('Training SVM for walking in level...');

% Extracting the data for walking up
label_walk_up=label_walk;
label_walk_up(label_walk_up==1)=0; label_walk_up(label_walk_up==3)=0;

disp('Training SVM for walking up...');
figure;
SVMModel_walkup=svmtrain(coeff_walk,label_walk_up,'ShowPlot',true,'kernel_function','polynomial','polyorder',1);
title('Training SVM for walking up...');

% Extracting the data for walking down
label_walk_down=label_walk;
label_walk_down(label_walk_down==1)=0; label_walk_down(label_walk_down==2)=0;

disp('Training SVM for walking down...');
figure;
SVMModel_walkdown=svmtrain(coeff_walk,label_walk_down,'ShowPlot',true,'kernel_function','polynomial','polyorder',1);
title('Training SVM for walking down...');

%% Training to recognize moving types

% Extracting the non walking data
label_nonwalk=cat(1,labels(labels==4),labels(labels==5),labels(labels==6));
coeff_nonwalk=cat(1,coeff(labels==4,:),coeff(labels==5,:),coeff(labels==6,:));

figure;
scatter(coeff_nonwalk(:,1),coeff_nonwalk(:,2),20,label_nonwalk);
title('Non Walking data');

% Training SVM to recognize laying and non laying
label_svm_laying=label_nonwalk;
label_svm_laying(label_svm_laying==4)=0; label_svm_laying(label_svm_laying==5)=0;

disp('Training SVM for laying');
figure;
SVMModel_laying = svmtrain(coeff_nonwalk,label_svm_laying,'ShowPlot',true,'kernel_function','polynomial','polyorder',2);
title('Training SVM for laying');

% Extracting the sitting 
label_sit=label_nonwalk;
label_sit(label_sit==5)=0; label_sit(label_sit==6)=0;

disp('Training SVM for sitting');
figure;
SVMModel_sit=svmtrain(coeff_nonwalk,label_sit,'ShowPlot',true,'kernel_function','polynomial','polyorder',1);
title('Training SVM for sitting');

% Extracting the standing
label_stand=label_nonwalk;
label_stand(label_stand==4)=0; label_stand(label_stand==6)=0;

disp('Training SVM for sitting');
figure;
SVMModel_stand=svmtrain(coeff_nonwalk,label_stand,'ShowPlot',true,'kernel_function','polynomial','polyorder',1);
title('Training SVM for sitting');


%----------------END OF TRAINING------------------%
%% Begin Testing of data

% Testing for walking or not
[label,confidence_split]=svmclassify(SVMModel_split,coeff_test,'ShowPlot',true);
truth_label_walk=test_labels(label==1);
truth_label_nonwalk=test_labels(label~=1);
label(label==1)=9;

indexforwalk=label==9;
coeff_walktest=coeff_test(label==9,:);

indexfornonwalk=label==0;
coeff_nonwalk=coeff_test(indexfornonwalk,:);

%Checking for walk on a level
[label_level,conf_level]=svmclassify(SVMModel_walklevel,coeff_walktest,'ShowPlot',true);
conf_level=abs(conf_level);
conf_level(label_level==0)=0;

% Classifying walk up 
[label_up,conf_up]=svmclassify(SVMModel_walkup,coeff_walktest,'ShowPlot',true);
conf_up=abs(conf_up);
conf_up(label_up==0)=0;

% Classifying walk down
[label_down,conf_down]=svmclassify(SVMModel_walkdown,coeff_walktest,'ShowPlot',true);
conf_down=abs(conf_down);
conf_down(label_down==0)=0;

label_struct_walk=[label_level, label_up, label_down, truth_label_walk];
conf_struct_walk=[conf_level, conf_up, conf_down, truth_label_walk];
[~,predicted_truth_1]=max(conf_struct_walk(:,1:3),[],2);
accuracy1=sum(predicted_truth_1==truth_label_walk)/length(truth_label_walk);

%% Classifying the non walk

% Testing for laying or non laying
[label_laying, conf_laying]=svmclassify(SVMModel_laying,coeff_nonwalk,'ShowPlot',true);
conf_laying=abs(conf_laying);
conf_laying(label_laying==0)=0;

%Testing for sit 
[label_sit,conf_sit]=svmclassify(SVMModel_sit,coeff_nonwalk,'ShowPlot',true);
conf_sit=abs(conf_sit);
conf_sit(label_sit==0)=0;

% Testing for standing
[label_stand, conf_stand]=svmclassify(SVMModel_stand,coeff_nonwalk,'ShowPlot',true);
conf_stand=abs(conf_stand);
conf_stand(label_stand==0)=0;

label_struct_nonwalk=[label_sit, label_stand, label_laying truth_label_nonwalk];
conf_struct_nonwalk=[conf_sit, conf_stand, conf_laying, truth_label_nonwalk];

[~,predicted_truth_2]=max(conf_struct_nonwalk(:,1:3),[],2);
predicted_truth_2=predicted_truth_2+3;
accuracy2=sum(predicted_truth_2==truth_label_nonwalk)/length(truth_label_nonwalk);

%----------------END OF TESTING---------------%

%% Checking accuracy and evaluating confusion
label=[predicted_truth_1;predicted_truth_2];
test_labels=[truth_label_walk; truth_label_nonwalk];
accuracy=sum(label==test_labels)*100/length(label);
fprintf('The accuracy of this SVM series is %f percent\n',accuracy);

confusion=zeros(6,6);
for i=1:length(label)
    j=label(i);
    k=test_labels(i);
    confusion(j,k)=confusion(j,k)+1;
end
confusion
covariance=cov(coeff)
