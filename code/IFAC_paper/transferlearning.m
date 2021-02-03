function [W,b,time_transfer]=transferlearning(x_train_new,y_train_new,s,alpha,sv,C,bandwidth,kernel_weight,InfoKernel)
% 迁移学习

tstart=tic;
% 设置迭代次数
T=1000;

% 预计算
[M,A1,A2,B1,B2]=precompute(x_train_new,y_train_new,s,alpha,sv,C,bandwidth,kernel_weight,InfoKernel);

% 求解模型线性组合系数
b=PSD(M,A1,A2,T,y_train_new);

% 求解更新后模型
W=MULTIpLE(alpha,sv,x_train_new,A1,A2,B1,B2,b);
time_transfer=toc(tstart);
end