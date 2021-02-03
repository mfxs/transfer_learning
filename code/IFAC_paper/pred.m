function [y_pred,accuracy,correct]=pred(x_test,y_test,alpha,bias,supportvector,s,bandwidth,kernel_weight,InfoKernel)
% 测试

% 初始化预测矩阵
y_pred=-1*ones(size(y_test,1),length(bias));

% 计算每一个二分类器的预测结果
for i=1:length(bias)
    fprintf('.');
    switch s 
        case 'mkl'
            Kt=mklkernel(x_test,InfoKernel,ones(size(kernel_weight)),supportvector{i},kernel_weight);
        case 'linear'
            Kt=x_test*supportvector{i}';
        case 'rbf'
            Kt=rbf_kernel(x_test,supportvector{i},bandwidth);
        case 'average'
            Kt=average_kernel(x_test,supportvector{i},bandwidth);
%         case 'ovo'
%             Kt=mklkernel(x_test,InfoKernel,ones(size(kernel_weight(i,:))),supportvector{i},kernel_weight(i,:));
    end
    y_pred(:,i)=Kt*alpha{i}+bias(i);
end
fprintf('\n');

% % 一对一情况
% if strcmp(s,'ovo')
%     class=size(y_test,2);
%     vote=zeros(size(y_test));
%     k=1;
%     for i=1:class
%         for j=i+1:class
%             for m=1:size(y_test,1)
%                 if y_pred(m,k)>0
%                     vote(m,i)=vote(m,i)+1;
%                 elseif y_pred(m,k)<0
%                     vote(m,j)=vote(m,j)+1;
%                 end
%             end
%             k=k+1;
%         end
%     end
%     [~,y_real]=max(y_test,[],2);
%     [~,y_pred]=max(vote,[],2);
%     accuracy=length(find(y_real==y_pred))/length(y_real);
%     return
% end

% 计算真实标签和预测正确个数
y_real=label_transform(y_test);
correct=zeros(1,size(y_test,2));
top=1;
for i=1:length(y_real)
    [~,pos]=sort(y_pred(i,:));
    if ismember(y_real(i),pos(end+1-top:end))
        correct(y_real(i))=correct(y_real(i))+1;
    end
end

% 计算准确率
accuracy(1)=sum(correct)/length(y_real);
accuracy(2)=sum(correct(1:end-1))/length(find(y_test(:,end)~=1));
accuracy(3)=correct(end)/length(find(y_test(:,end)==1));
end