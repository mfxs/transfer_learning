function [alpha,bias,supportvector]=updation(x_new,y_new,alpha,supportvector,s,C,kernel_weight,InfoKernel)
% ����ģ��

% ���������������������
m=size(x_new,1);
n=size(y_new,2);

% ����M
switch s
    case 'mkl'
        M=[mklkernel(x_new,InfoKernel,ones(size(kernel_weight)),x_new,kernel_weight)+eye(m)/C ones(m,1);ones(1,m) 0];
    case 'linear'
        M=[x_new*x_new'+eye(m)/C ones(m,1);ones(1,m) 0];
end
M=M^-1;

% ����temp1��temp2
temp1=zeros(size(y_new));
for i=1:n
    switch s
        case 'mkl'
            Kt=mklkernel(x_new,InfoKernel,ones(size(kernel_weight)),supportvector{i},kernel_weight);
        case 'linear'
            Kt=x_new*supportvector{i}';
    end
    temp1(:,i)=Kt*alpha{i};
end
temp2=M*[y_new-temp1;ones(1,n)];

% �����������ճ��Ӻ�ƫ��
A=temp2(1:end-1,:);
bias=temp2(end,:);

% ����ģ��
for i=1:n
    alpha{i}=[alpha{i};A(:,i)];
    supportvector{i}=[supportvector{i};x_new];
end
end