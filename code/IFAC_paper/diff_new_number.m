function [x_train_new1,y_train_new1,sample_train_new1]=diff_new_number(x_train_new,y_train_new,new_number1)
% ���ɲ�ͬ������������ѵ����

% ������ѵ������������
sample_train_new=cell(1,size(y_train_new,2));
for i=1:size(y_train_new,1)
    sample_train_new{y_train_new(i,:)==1}(end+1,1)=i;
end

% ��������������
class=size(y_train_new,2);
feature=size(x_train_new,2);

% ��ʼ����ѵ����
x_train_new1=zeros(new_number1*class,feature);
y_train_new1=zeros(new_number1*class,class);
sample_train_new1=sample_train_new;

% ��ȡ��������
index1=1;
for i=1:class
    sample_train_new1{i}(new_number1+1:end)=[];
    x_train_new1(index1:index1+new_number1-1,:)=x_train_new(sample_train_new1{i},:);
    y_train_new1(index1:index1+new_number1-1,:)=y_train_new(sample_train_new1{i},:);
    index1=index1+new_number1;
end

% ������������
sample_train_new1=cell(1,class);
for i=1:size(y_train_new1,1)
    sample_train_new1{y_train_new1(i,:)==1}(end+1,1)=i;
end
end