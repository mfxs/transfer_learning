function [x,y]=divide(x_train,y_train,n)
% 划分增量式训练集

% 初始化增量式训练集
x=cell(1,length(n));
y=cell(1,length(n));

% 计算每个类别的样本数
train_num=length(find(y_train(:,1)==1));
if sum(n)~=train_num
    disp('错误的划分。');
else
%     根据个数划分训练集
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