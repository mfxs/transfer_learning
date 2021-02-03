function [alpha_best,supportvector_best,bias_best,C_best,time_train,time_validation]=skl(x_train,y_train,train_index,x_validation,y_validation,sample_validation,range_C,s)
% ����ѧϰ

% ���������
class=size(y_train,2);

% ��ʼ���������ճ���,ƫ�ú�֧������
alpha=cell(1,class);
bias=zeros(1,class);
supportvector=cell(1,class);

time_train=zeros(length(range_C),1);
time_validation=zeros(length(range_C),1);
k=1;
% �Ż��ͷ�����
accuracy_best=0;
for C=range_C
    disp(strcat('C=',num2str(C)));
    tstart=tic;
    for i=1:class
%         ѵ��
        disp(strcat('class',num2str(i)));
        tic
        switch s
            case 'linear'
                model=fitcsvm(x_train(train_index{i},:),y_train(train_index{i},i),'kernelfunction',s,'BoxConstraint',C);
            case 'rbf'
                model=fitcsvm(x_train(train_index{i},:),y_train(train_index{i},i),'kernelfunction',s,'BoxConstraint',C,'kernelscale',100*sqrt(2));
        end
        toc
        alpha{i}=model.Alpha.*model.SupportVectorLabels;
        bias(i)=model.Bias;
        supportvector{i}=model.SupportVectors;
    end
    time_train(k)=toc(tstart);
%     ����֤���ϲ���
    tstart=tic;
    switch s
        case 'linear'
            accuracy=test({alpha,supportvector},bias,x_validation,y_validation,sample_validation,0,'linear_model');
        case 'rbf'
            accuracy=test({alpha,supportvector},bias,x_validation,y_validation,sample_validation,0,'kernel_model');
    end
    time_validation(k)=toc(tstart);
    disp(strcat('��֤����׼ȷ��Ϊ',num2str(100*accuracy(3,1)),'%'));
    if accuracy(3,1)>accuracy_best
        C_best=C;
        alpha_best=alpha;
        bias_best=bias;
        supportvector_best=supportvector;
        accuracy_best=accuracy(3,1);
    end
    k=k+1;
end
time_train=mean(time_train);
time_validation=mean(time_validation);
end