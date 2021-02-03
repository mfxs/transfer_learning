function [alpha_best,sv_best,bias_best,C_best,hist]=train_linear(x_train,y_train,x_validation,y_validation,C)
% 线性模型训练

% 计算类别数
class=size(y_train,2);

% 初始化拉格朗日乘子、支持向量和偏置
alpha=cell(1,class);
sv=cell(1,class);
bias=zeros(1,class);

% 训练
accuracy_best=0;
hist=[];
ktrain=[(1:size(x_train,1))' x_train*x_train'];
for c=C
%     采用一对多的方式进行训练
    disp(['C=',num2str(c)]);
    options=['-s 0 -t 4 -c ',num2str(c)];
    for i=1:class
        model=svmtrain(y_train(:,i),ktrain,options);
        alpha{i}=model.sv_coef;
        sv{i}=x_train(model.SVs,:);
        bias(i)=-model.rho;
    end
    
%     记录不同参数下训练和验证的准确率
    disp('Predict on train set.');
    [~,temp]=pred(x_train,y_train,alpha,bias,sv,'linear');
    disp('Predict on validation set.');
    [~,accuracy]=pred(x_validation,y_validation,alpha,bias,sv,'linear');
    hist(1,end+1)=temp(1);
    hist(2,end)=accuracy(1);
    
%     寻找验证准确率最高的参数    
    if accuracy(1)>accuracy_best
        alpha_best=alpha;
        sv_best=sv;
        bias_best=bias;
        C_best=c;
        accuracy_best=accuracy(1);
    end
end
end