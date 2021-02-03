function [alpha_best,sv_best,bias_best,C_best,bandwidth_best,hist]=train_average(x_train,y_train,x_validation,y_validation,C,bandwidth)
% ƽ����ģ��ѵ��

% ���������
class=size(y_train,2);

% ��ʼ���������ճ��ӡ�֧��������ƫ��
alpha=cell(1,class);
sv=cell(1,class);
bias=zeros(1,class);

% ѵ��
accuracy_best=0;
hist=cell(1,2);
hist{1}=zeros(length(C),length(bandwidth));
hist{2}=zeros(length(C),length(bandwidth));
for c=C
    for sigma=bandwidth
%         ����һ�Զ�ķ�ʽ����ѵ��
        disp(['C=',num2str(c),',bandwidth=',num2str(sigma)]);
        ktrain=[(1:size(x_train,1))' average_kernel(x_train,x_train,sigma)];
        options=['-s 0 -t 4 -c ',num2str(C)];
        for i=1:class
            model=svmtrain(y_train(:,i),ktrain,options);
            alpha{i}=model.sv_coef;
            sv{i}=x_train(model.SVs,:);
            bias(i)=-model.rho;
        end
        
%         ��¼��ͬ������ѵ������֤��׼ȷ��
        disp('Predict on train set.');
        [~,temp]=pred(x_train,y_train,alpha,bias,sv,'average',sigma);
        disp('Predict on validation set.');
        [~,accuracy]=pred(x_validation,y_validation,alpha,bias,sv,'average',sigma);
        hist{1}(c==C,sigma==bandwidth)=temp(1);
        hist{2}(c==C,sigma==bandwidth)=accuracy(1);
        
%         Ѱ����֤׼ȷ����ߵĲ���
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