%% 导入工具箱路径
addpath('D:\mfxs\SimpleMKL','D:\mfxs\libsvm-3.23\matlab');

%% 加载数据
disp('loading...');
S=load('data.mat');
disp('loading done');

%% 超参数范围设置
global bandwidth_mkl;
bandwidth_mkl=10.^(1:4);
C=10.^(-1:6);
correct=zeros(4,13);

%% 实验十次
for time=1:10
    %% 数据划分
    [x_train,y_train,x_val,y_val,x_train_new,y_train_new,x_test,y_test]=split2(S);

    %% 原始模型训练
    disp('---------------------------');
    disp('训练原始模型');
    disp('---------------------------');
    [alpha,bias,sv,kernel_weight,InfoKernel,C_best,hist]=train_mkl(x_train,y_train,x_val,y_val,C,bandwidth_mkl);
    [~,~,temp]=pred(x_test,y_test,alpha,bias,sv,'mkl',[],kernel_weight,InfoKernel);
    correct(1,:)=correct(1,:)+temp;

    %% 模型迁移
    disp('---------------------------');
    disp('模型迁移');
    disp('---------------------------');
    W=cell(1,4);
    W{1}={alpha,sv,bias};
    for i=1:3
        disp(['迁移第',num2str(i),'类']);
        W{i+1}=transferlearning(x_train_new{i},y_train_new{i},'mkl',W{i}{1},W{i}{2},C_best,[],kernel_weight,InfoKernel);
        [~,~,temp]=pred(x_test,y_test,W{i+1}{1},W{i+1}{3},W{i+1}{2},'mkl',[],kernel_weight,InfoKernel);
        correct(i+1,:)=correct(i+1,:)+temp;
    end
end
correct=correct/time;

%% 画图
for i=1:4
    subplot(2,2,i);
    bar(correct(i,:),'y','edgecolor','none');
    hold on
    bar(correct(i,1:end-1),'r','edgecolor','none');
    hold on
    bar(correct(i,1:end-2),'b','edgecolor','none');
    hold on
    bar(correct(i,1:end-3),'g','edgecolor','none');
    grid on;
    axis([0 14 0 20]);
    xlabel('Class');
    ylabel('Correct Number');
%     legend('Correct Prediction');
    if i==1
        title('N-class classifier');
    else
        title(['N+',num2str(i-1),'-class classifier']);
    end
end
suptitle('Correct predicted sample');