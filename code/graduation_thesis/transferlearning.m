function [W,B,time_transfer]=transferlearning(x_train_new,y_train_new,s,alpha,supportvector,C,kernel_weight,Weight,InfoKernel)
% Ǩ��ѧϰ

tstart=tic;
% ���õ�������
T=1000;
W={alpha,supportvector};

% Ԥ����
switch s
    case 'multi_kernel'
        [M,A1,A2,B1,B2]=precompute(x_train_new,y_train_new,s,W,C,InfoKernel,Weight,kernel_weight);
    otherwise
        [M,A1,A2,B1,B2]=precompute(x_train_new,y_train_new,s,W,C);
end

% ���ģ���������ϵ��
b=PSD(M,A1,A2,T,y_train_new);

% �����º�ģ��
[W,B]=MULTIpLE(W,x_train_new,A1,A2,B1,B2,b);
time_transfer=toc(tstart);
end