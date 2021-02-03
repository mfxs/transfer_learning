function [alpha,bias,supportvector]=train_average(x_train,y_train,C)
% ƽ����ģ��ѵ��

% ���������
class=size(y_train,2);

% ��ʼ���������ճ���,ƫ�ú�֧������
alpha=cell(1,class);
bias=zeros(1,class);
supportvector=cell(1,class);

% ѵ��
for i=1:class
    disp(strcat('class',num2str(i)));
    model=fitcsvm(x_train,y_train(:,i),'kernelfunction','kernel','BoxConstraint',C);
    alpha{i}=model.Alpha.*model.SupportVectorLabels;
    bias(i)=model.Bias;
    supportvector{i}=model.SupportVectors;
end
end