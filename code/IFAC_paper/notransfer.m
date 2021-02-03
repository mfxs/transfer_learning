function [alpha,sv,bias]=notransfer(x_train_new,y_train_new,alpha,sv,bias,C,kernel_weight,InfoKernel)
% 构建新类别和原有类别的分类器

options=['-s 0 -t 4 -c ',num2str(C)];
ktrain=mklkernel(x_train_new,InfoKernel,ones(size(kernel_weight)),x_train_new,kernel_weight);
ktrain=[(1:size(x_train_new,1))' ktrain];
model=svmtrain(y_train_new(:,end),ktrain,options);
alpha{end+1}=model.sv_coef;
sv{end+1}=x_train_new(model.SVs,:);
bias(end+1)=-model.rho;
end