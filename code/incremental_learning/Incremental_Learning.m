% 参数设置
bandwidth=10*[1 2 5 7 10];
C=10000;

% 数据划分
[x_train,y_train,x_test,y_test]=train_test();

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 

% 多核模型
[alpha_mkl,bias_mkl,supportvector_mkl,kernel_weight_mkl,InfoKernel_mkl]=mmkl(x_train,y_train,bandwidth,C);
[y_pred_mkl,accuracy_mkl]=pred(x_test,y_test,alpha_mkl,bias_mkl,supportvector_mkl,'common',kernel_weight_mkl,InfoKernel_mkl);

% 线性模型
[alpha_lin,bias_lin,supportvector_lin]=train_linear(x_train,y_train,C);
[y_pred_lin,accuracy_lin]=pred(x_test,y_test,alpha_lin,bias_lin,supportvector_lin,'linear');

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 

% 增量式数据集划分
n=[20,40];
[x,y]=divide(x_train,y_train,n);
accuracy=zeros(1,length(n));

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 

% 训练初始多核模型
[alpha_inc,bias_inc,supportvector_inc,kernel_weight_inc,InfoKernel_inc]=mmkl(x{1},y{1},bandwidth,C);
[y_pred,accuracy(1)]=pred(x_test,y_test,alpha_inc,bias_inc,supportvector_inc,'common',kernel_weight_inc,InfoKernel_inc);

% 增量式训练多核模型
for i=2:length(n)
    disp(strcat('section',num2str(i)));
    [alpha_inc,bias_inc,supportvector_inc]=updation(x{i},y{i},alpha_inc,supportvector_inc,'mkl',C,kernel_weight_inc,InfoKernel_inc);
    [y_pred,accuracy(i)]=pred(x_test,y_test,alpha_inc,bias_inc,supportvector_inc,'common',kernel_weight_inc,InfoKernel_inc);
    disp(strcat('accuracy:',num2str(accuracy)));
end

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 

% % 训练初始线性模型
% [alpha_inc,bias_inc,supportvector_inc]=train_linear(x{1},y{1},C);
% [y_pred,accuracy(1)]=pred(x_test,y_test,alpha_inc,bias_inc,supportvector_inc,'linear');
% 
% % 增量式训练线性模型
% for i=2:length(n)
%     disp(strcat('section',num2str(i)));
%     [alpha_inc,bias_inc,supportvector_inc]=updation(x{i},y{i},alpha_inc,supportvector_inc,'linear',C);
%     [y_pred,accuracy(i)]=pred(x_test,y_test,alpha_inc,bias_inc,supportvector_inc,'linear');
%     disp(strcat('accuracy:',num2str(accuracy)));
% end