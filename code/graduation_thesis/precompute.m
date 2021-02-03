function [M,A1,A2,B1,B2]=precompute(x_train_new,y_train_new,s,W,C,InfoKernel,Weight,kernel_weight) %x_train_new和y_train_new为新的训练数据
% 预计算

% 样本个数
m=size(x_train_new,1);
switch s
    case 'multi_kernel'
        M=[mklkernel(x_train_new,InfoKernel,Weight,x_train_new,kernel_weight)+eye(m)/C ones(m,1);ones(1,m) 0];
    case 'average_kernel'
        M=[kernel(x_train_new,x_train_new)+eye(m)/C ones(m,1);ones(1,m) 0];
    case 'linear_model'
        M=[x_train_new*x_train_new'+eye(m)/C ones(m,1);ones(1,m) 0];
    case 'kernel_model'
        M=[kernel_rbf(x_train_new,x_train_new)+eye(m)/C ones(m,1);ones(1,m) 0];
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
        case 'multi_kernel'
            temp=mklkernel(x_train_new,InfoKernel,Weight,W{2}{i},kernel_weight);
        case 'average_kernel'
            temp=kernel(x_train_new,W{2}{i});
        case 'linear_model'
            temp=x_train_new*W{2}{i}';
        case 'kernel_model'
            temp=kernel_rbf(x_train_new,W{2}{i});
    end
    for j=1:size(temp,2)
        temp2(:,i)=temp2(:,i)+W{1}{i}(j)*temp(:,j);
    end
end
temp3=M*[temp2;zeros(1,n-1)];
A2=temp3(1:m,:);
B2=temp3(end,:);
B2=B2';
end