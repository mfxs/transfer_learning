function [W,bias,Weight,kernel_weight,time_notransfer]=notransferlearning(x_train_new,y_train_new,train_new_index,kernel_weight,alpha,bias,supportvector,Weight,kernel,kerneloptionvec,variableveccell,options,C)
% ²»Ç¨ÒÆÑ§Ï°

tstart=tic;
verbose=0;
W={alpha,supportvector};

[Weight(end+1,:),InfoKernel]=UnitTraceNormalization(x_train_new(train_new_index,:),kernel,kerneloptionvec,variableveccell);
K=mklkernel(x_train_new(train_new_index,:),InfoKernel,Weight(end,:));

% ÑµÁ·
[kernel_weight(end+1,:),W{1}{end+1},bias(end+1),supportvector_index] = mklsvm(K,y_train_new(train_new_index,end),C,options,verbose);
W{2}{end+1}=x_train_new(train_new_index(supportvector_index),:);
time_notransfer=toc(tstart);
end