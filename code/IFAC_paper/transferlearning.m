function [W,b,time_transfer]=transferlearning(x_train_new,y_train_new,s,alpha,sv,C,bandwidth,kernel_weight,InfoKernel)
% Ǩ��ѧϰ

tstart=tic;
% ���õ�������
T=1000;

% Ԥ����
[M,A1,A2,B1,B2]=precompute(x_train_new,y_train_new,s,alpha,sv,C,bandwidth,kernel_weight,InfoKernel);

% ���ģ���������ϵ��
b=PSD(M,A1,A2,T,y_train_new);

% �����º�ģ��
W=MULTIpLE(alpha,sv,x_train_new,A1,A2,B1,B2,b);
time_transfer=toc(tstart);
end