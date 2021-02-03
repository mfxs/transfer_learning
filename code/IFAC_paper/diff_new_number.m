function [x_train_new1,y_train_new1,sample_train_new1]=diff_new_number(x_train_new,y_train_new,new_number1)
% 生成不同样本数量的新训练集

% 生成新训练样本的索引
sample_train_new=cell(1,size(y_train_new,2));
for i=1:size(y_train_new,1)
    sample_train_new{y_train_new(i,:)==1}(end+1,1)=i;
end

% 计算类别和特征数
class=size(y_train_new,2);
feature=size(x_train_new,2);

% 初始化新训练集
x_train_new1=zeros(new_number1*class,feature);
y_train_new1=zeros(new_number1*class,class);
sample_train_new1=sample_train_new;

% 提取部分样本
index1=1;
for i=1:class
    sample_train_new1{i}(new_number1+1:end)=[];
    x_train_new1(index1:index1+new_number1-1,:)=x_train_new(sample_train_new1{i},:);
    y_train_new1(index1:index1+new_number1-1,:)=y_train_new(sample_train_new1{i},:);
    index1=index1+new_number1;
end

% 重新生成索引
sample_train_new1=cell(1,class);
for i=1:size(y_train_new1,1)
    sample_train_new1{y_train_new1(i,:)==1}(end+1,1)=i;
end
end