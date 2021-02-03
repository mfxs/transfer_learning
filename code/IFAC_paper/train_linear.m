function [alpha_best,sv_best,bias_best,C_best,hist]=train_linear(x_train,y_train,x_validation,y_validation,C)
% ����ģ��ѵ��

% ���������
class=size(y_train,2);

% ��ʼ���������ճ��ӡ�֧��������ƫ��
alpha=cell(1,class);
sv=cell(1,class);
bias=zeros(1,class);

% ѵ��
accuracy_best=0;
hist=[];
ktrain=[(1:size(x_train,1))' x_train*x_train'];
for c=C
%     ����һ�Զ�ķ�ʽ����ѵ��
    disp(['C=',num2str(c)]);
    options=['-s 0 -t 4 -c ',num2str(c)];
    for i=1:class
        model=svmtrain(y_train(:,i),ktrain,options);
        alpha{i}=model.sv_coef;
        sv{i}=x_train(model.SVs,:);
        bias(i)=-model.rho;
    end
    
%     ��¼��ͬ������ѵ������֤��׼ȷ��
    disp('Predict on train set.');
    [~,temp]=pred(x_train,y_train,alpha,bias,sv,'linear');
    disp('Predict on validation set.');
    [~,accuracy]=pred(x_validation,y_validation,alpha,bias,sv,'linear');
    hist(1,end+1)=temp(1);
    hist(2,end)=accuracy(1);
    
%     Ѱ����֤׼ȷ����ߵĲ���    
    if accuracy(1)>accuracy_best
        alpha_best=alpha;
        sv_best=sv;
        bias_best=bias;
        C_best=c;
        accuracy_best=accuracy(1);
    end
end
end