function [x_train_new,y_train_new]=pure_new(x_train_new,y_train_new)
% ����ֻ����������ѵ������

index=find(y_train_new(:,end)==1);
x_train_new=x_train_new(index,:);
y_train_new=y_train_new(index,:);
end