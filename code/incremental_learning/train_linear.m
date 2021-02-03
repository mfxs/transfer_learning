function [alpha,bias,supportvector]=train_linear(x_train,y_train,C)
% 线性模型训练

% model=fitcsvm(x_train_real,y_train_real(:,1),'kernelfunction','linear','BoxConstraint',C);
% alpha{1}=model.Alpha.*model.SupportVectorLabels;
% bias(1)=model.Bias;
% supportvector{1}=model.SupportVectors;

% 计算类别数
class=size(y_train,2);

% 初始化拉格朗日乘子,偏置和支持向量
alpha=cell(1,class);
bias=zeros(1,class);
supportvector=cell(1,class);

% 训练
for i=1:class
    disp(strcat('class',num2str(i)));
    model=fitcsvm(x_train,y_train(:,i),'kernelfunction','linear','BoxConstraint',C);
    alpha{i}=model.Alpha.*model.SupportVectorLabels;
    bias(i)=model.Bias;
    supportvector{i}=model.SupportVectors;
end
end