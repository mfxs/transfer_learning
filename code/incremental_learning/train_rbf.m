function [alpha,bias,supportvector]=train_rbf(x_train,y_train,C)
% RBF核模型训练

% 计算类别数
class=size(y_train,2);

% 初始化拉格朗日乘子,偏置和支持向量
alpha=cell(1,class);
bias=zeros(1,class);
supportvector=cell(1,class);

% 训练
for i=1:class
    disp(strcat('class',num2str(i)));
    model=fitcsvm(x_train,y_train(:,i),'kernelfunction','rbf','BoxConstraint',C,'kernelscale',100*sqrt(2));
    alpha{i}=model.Alpha.*model.SupportVectorLabels;
    bias(i)=model.Bias;
    supportvector{i}=model.SupportVectors;
end
end