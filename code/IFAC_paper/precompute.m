function [M,A1,A2,B1,B2]=precompute(x_train_new,y_train_new,s,alpha,sv,C,bandwidth,kernel_weight,InfoKernel) %x_train_new和y_train_new为新的训练数据
% 预计算

% 样本个数
m=size(x_train_new,1);
switch s
    case 'mkl'
        M=[mklkernel(x_train_new,InfoKernel,ones(size(kernel_weight)),x_train_new,kernel_weight)+eye(m)/C ones(m,1);ones(1,m) 0];
    case 'linear'
        M=[x_train_new*x_train_new'+eye(m)/C ones(m,1);ones(1,m) 0];
    case 'rbf'
        M=[rbf_kernel(x_train_new,x_train_new,bandwidth)+eye(m)/C ones(m,1);ones(1,m) 0];
    case 'average'
        M=[average_kernel(x_train_new,x_train_new,bandwidth)+eye(m)/C ones(m,1);ones(1,m) 0];
end
M=M^-1;

% 类别数
n=size(y_train_new,2);
temp1=M*[y_train_new;zeros(1,n)];
A1=temp1(1:m,:);
B1=temp1(end,:);
B1=B1';
temp2=zeros(m,n-1);
for i=1:n-1
    switch s
        case 'mkl'
            temp=mklkernel(x_train_new,InfoKernel,ones(size(kernel_weight)),sv{i},kernel_weight);
        case 'linear'
            temp=x_train_new*sv{i}';
        case 'rbf'
            temp=rbf_kernel(x_train_new,sv{i},bandwidth);
        case 'average'
            temp=average_kernel(x_train_new,sv{i},bandwidth);
    end
    for j=1:size(temp,2)
        temp2(:,i)=temp2(:,i)+alpha{i}(j)*temp(:,j);
    end
end
temp3=M*[temp2;zeros(1,n-1)];
A2=temp3(1:m,:);
B2=temp3(end,:);
B2=B2';
end