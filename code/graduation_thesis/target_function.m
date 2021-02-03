function f=target_function(x,alpha,supportvector,bias,x_validation,y_validation,sample_validation,InfoKernel,Weight,kernel_weight)
% 遗传算法优化目标函数

accuracy=test({alpha,supportvector},bias,x_validation,y_validation,sample_validation,0,'common',InfoKernel,x*Weight,x*kernel_weight);
f=-accuracy(3,1);
end