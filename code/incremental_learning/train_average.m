function [alpha,bias,supportvector]=train_average(x_train,y_train,C)
% 平均核模型训练

% 计算类别数
class=size(y_train,2);

% 初始化拉格朗日乘子,偏置和支持向量
alpha=cell(1,class);
bias=zeros(1,class);
supportvector=cell(1,class);

% 训练
for i=1:class
    disp(strcat('class',num2str(i)));
    model=fitcsvm(x_train,y_train(:,i),'kernelfunction','kernel','BoxConstraint',C);
    alpha{i}=model.Alpha.*model.SupportVectorLabels;
    bias(i)=model.Bias;
    supportvector{i}=model.SupportVectors;
end
end