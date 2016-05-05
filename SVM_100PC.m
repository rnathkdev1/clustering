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

%% Performing PCA to the data
[coeff]=pca(data','NumComponents',100);
normalparam=repmat(max(coeff),[length(coeff) 1]);
coeff=coeff./normalparam;
%% Training to recognize Walking or not walking

label_svm_1=labels;
label_svm_1(label_svm_1==4)=0; label_svm_1(label_svm_1==5)=0; label_svm_1(label_svm_1==6)=0;
label_svm_1(label_svm_1==1)=1; label_svm_1(label_svm_1==2)=1; label_svm_1(label_svm_1==3)=1;

disp('Training SVM for walking...or not walking...');
SVMModel_split = fitcsvm(coeff,label_svm_1,'KernelFunction','Polynomial','PolynomialOrder',1);


%% Training to recognize walking types:

% Extracting the walking data
label_walk=cat(1,labels(labels==1),labels(labels==2),labels(labels==3));
coeff_walk=cat(1,coeff(labels==1,:),coeff(labels==2,:),coeff(labels==3,:));

% Training SVM to recognize walking in level
label_svm_walklevel=label_walk;
label_svm_walklevel(label_svm_walklevel==2)=0; label_svm_walklevel(label_svm_walklevel==3)=0;

disp('Training SVM for walking in level...or on an incline...');
SVMModel_walklevel = fitcsvm(coeff_walk,label_svm_walklevel,'KernelFunction','Polynomial','PolynomialOrder',2);

% Extracting the data for walking inclines
label_walk_incline=cat(1,label_walk(label_walk==2),label_walk(label_walk==3));
coeff_walkincline=cat(1,coeff_walk(label_walk==2,:),coeff_walk(label_walk==3,:));

disp('Training SVM for walking up...or down...');
SVMModel_walkincline=fitcsvm(coeff_walkincline,label_walk_incline,'KernelFunction','Polynomial','PolynomialOrder',2);

%% Training to recognize moving types

% Extracting the non walking data
label_nonwalk=cat(1,labels(labels==4),labels(labels==5),labels(labels==6));
coeff_nonwalk=cat(1,coeff(labels==4,:),coeff(labels==5,:),coeff(labels==6,:));

% Training SVM to recognize laying and non laying
label_svm_laying=label_nonwalk;
label_svm_laying(label_svm_laying==4)=0; label_svm_laying(label_svm_laying==5)=0;

disp('Training SVM for laying...or not...');
SVMModel_laying = fitcsvm(coeff_nonwalk,label_svm_laying,'KernelFunction','Polynomial','PolynomialOrder',2);

% Extracting the sitting and standing data
label_sitstand=cat(1,label_nonwalk(label_nonwalk==4),label_nonwalk(label_nonwalk==5));
coeff_sitstand=cat(1,coeff_nonwalk(label_nonwalk==4,:),coeff_nonwalk(label_nonwalk==5,:));

disp('Training SVM for sitting...or standing...');
SVMModel_sitstand=fitcsvm(coeff_sitstand,label_sitstand,'KernelFunction','Polynomial','PolynomialOrder',1);

%---------------------------------------END OF TRAINING-----------------------------------------------%
%% Begin Testing of data

test_data=load('UCI HAR Dataset/test/X_test.txt');
test_labels=load('UCI HAR Dataset/test/y_test.txt');

coeff_test=pca(test_data','NumComponents',100);
normalparam=repmat(max(coeff_test),[length(coeff_test) 1]);
coeff_test=coeff_test./normalparam;

% Testing for walking or not
label=predict(SVMModel_split,coeff_test);
label(label==1)=9;

indexforwalk=label==9;
coeff_walktest=coeff_test(label==9,:);

%Checking for walk on a level or not
label_level=predict(SVMModel_walklevel,coeff_walktest);
label_level(label_level==0)=9;

% Assigning the decisions back
label(indexforwalk)=label_level;

% Accessing unassigned labels again
indexforwalkup=label==9;
coeff_walkup=coeff_test(label==9,:);

% Classifying walk up or down
label_updown=predict(SVMModel_walkincline,coeff_walkup);

% Assigning decisions back
label(indexforwalkup)=label_updown;

%% Classifying the non walk

indexfornonwalk=label==0;
coeff_nonwalk=coeff_test(indexfornonwalk,:);

% Testing for laying or non laying
label_laying=predict(SVMModel_laying,coeff_nonwalk);

%Assigning labels back
label(indexfornonwalk)=label_laying;

% Extracting unassigned labels
indexforsitstand=label==0;
coeff_sitstand=coeff_test(indexforsitstand,:);

%Testing for sit or stand
label_sitstand=predict(SVMModel_sitstand,coeff_sitstand);

%Assigning labels back
label(indexforsitstand)=label_sitstand;

%----------------END OF TESTING---------------%

%% Checking accuracy and evaluating confusion

accuracy=sum(label==test_labels)*100/length(label);
fprintf('The accuracy of this SVM series is %f percent\n',accuracy);

confusion=zeros(6,6);
for i=1:length(label)
    j=label(i);
    k=test_labels(i);
    confusion(j,k)=confusion(j,k)+1;
end
confusion