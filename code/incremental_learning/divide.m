function [x,y]=divide(x_train,y_train,n)
% ��������ʽѵ����

% ��ʼ������ʽѵ����
x=cell(1,length(n));
y=cell(1,length(n));

% ����ÿ������������
train_num=length(find(y_train(:,1)==1));
if sum(n)~=train_num
    disp('����Ļ��֡�');
else
%     ���ݸ�������ѵ����
    temp=1;
    for i=1:length(n)
        index=[];
        for j=1:size(y_train,2)
            index=[index (temp+train_num*(j-1):temp+train_num*(j-1)+n(i)-1)];
        end
        x{i}=x_train(index,:);
        y{i}=y_train(index,:);
        temp=temp+n(i);
    end
end
end