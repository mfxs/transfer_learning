function [W,B,time_transfer]=transferlearning(x_train_new,y_train_new,s,alpha,supportvector,C,kernel_weight,Weight,InfoKernel)
% 迁移学习

tstart=tic;
% 设置迭代次数
T=1000;
W={alpha,supportvector};

% 预计算
switch s
    case 'multi_kernel'
        [M,A1,A2,B1,B2]=precompute(x_train_new,y_train_new,s,W,C,InfoKernel,Weight,kernel_weight);
    otherwise
        [M,A1,A2,B1,B2]=precompute(x_train_new,y_train_new,s,W,C);
end

% 求解模型线性组合系数
b=PSD(M,A1,A2,T,y_train_new);

% 求解更新后模型
[W,B]=MULTIpLE(W,x_train_new,A1,A2,B1,B2,b);
time_transfer=toc(tstart);
end