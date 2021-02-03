function [alpha_best,sv_best,bias_best,C_best,bandwidth_best,hist]=train_average(x_train,y_train,x_validation,y_validation,C,bandwidth)
% 平均核模型训练

% 计算类别数
class=size(y_train,2);

% 初始化拉格朗日乘子、支持向量和偏置
alpha=cell(1,class);
sv=cell(1,class);
bias=zeros(1,class);

% 训练
accuracy_best=0;
hist=cell(1,2);
hist{1}=zeros(length(C),length(bandwidth));
hist{2}=zeros(length(C),length(bandwidth));
for c=C
    for sigma=bandwidth
%         采用一对多的方式进行训练
        disp(['C=',num2str(c),',bandwidth=',num2str(sigma)]);
        ktrain=[(1:size(x_train,1))' average_kernel(x_train,x_train,sigma)];
        options=['-s 0 -t 4 -c ',num2str(C)];
        for i=1:class
            model=svmtrain(y_train(:,i),ktrain,options);
            alpha{i}=model.sv_coef;
            sv{i}=x_train(model.SVs,:);
            bias(i)=-model.rho;
        end
        
%         记录不同参数下训练和验证的准确率
        disp('Predict on train set.');
        [~,temp]=pred(x_train,y_train,alpha,bias,sv,'average',sigma);
        disp('Predict on validation set.');
        [~,accuracy]=pred(x_validation,y_validation,alpha,bias,sv,'average',sigma);
        hist{1}(c==C,sigma==bandwidth)=temp(1);
        hist{2}(c==C,sigma==bandwidth)=accuracy(1);
        
%         寻找验证准确率最高的参数
        if accuracy(1)>accuracy_best
            alpha_best=alpha;
            sv_best=sv;
            bias_best=bias;
            C_best=c;
            bandwidth_best=sigma;
            accuracy_best=accuracy(1);
        end
    end
end
end