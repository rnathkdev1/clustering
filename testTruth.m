function testTruth(c)

labels=load('labels.txt');
accuracy1=sum(labels==c');
accuracy2=sum(labels==3-c');
accuracy=max(accuracy1,accuracy2);
percent=accuracy*100/length(c);
fprintf('Accuracy with the ground truth for this case is %f percent\n',percent);

disp('Indices in one cluster are')
I=find(c==1);
I
disp('Indices in other cluster are')
I=find(c==2);
I
end