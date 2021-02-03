function [kernel_weight_best,alpha_best,bias_best,supportvector_best,Weight,InfoKernel,kernel,kerneloptionvec,variableveccell,options,C_best,time_train,time_validation]=mkl(x_train,y_train,train_index,x_validation,y_validation,sample_validation,bandwidth,range_C,verbose)
% 多核学习

% 参数设置
options.algo='svmclass';
options.stopvariation=0;
options.stopKKT=0;
options.stopdualitygap=1;
options.seuildiffsigma=0.1;
options.seuildiffconstraint=0.1;
options.seuildualitygap=0.5;
options.goldensearch_deltmax=1e-1;
options.numericalprecision=1e-8;
options.lambdareg = 1e-8;
options.firstbasevariable='first';
options.nbitermax=500;
options.seuil=0;
options.seuilitermax=10;
options.miniter=0;
options.verbosesvm=0;
options.efficientkernel=0;

% 核数量
kernel_number=30;
kernel=cell(1,kernel_number);
kerneloptionvec=cell(1,kernel_number);
variableveccell=cell(1,kernel_number);
kernel_length=[20,80,320,1280,40,160,640,2560,300,300,300,300,298,300,298,298,1000,1000,1000,1000,991,995,995,993,28,112,448,37,148,592];

% 生成核参数
index=1;
for i=1:kernel_number
    variableveccell{i}=index:index-1+kernel_length(i);
    index=index+kernel_length(i);
    kernel{i}='gaussian';
    kerneloptionvec{i}=bandwidth;
end

% 计算类别数以及转化索引到训练样本
class=size(y_train,2);
x=cell(1,class);
y=cell(1,class);
for i=1:class
    x{i}=x_train(train_index{i},:);
    y{i}=y_train(train_index{i},i);
end

% 初始化核系数,拉格朗日乘子,偏置和支持向量
kernel_weight=zeros(class,kernel_number*length(bandwidth));
alpha=cell(1,class);
bias=zeros(1,class);
supportvector=cell(1,class);
Weight=zeros(class,kernel_number*length(bandwidth));
K=cell(1,class);

% 训练原始模型
tstart=tic;
[Weight(1,:),InfoKernel]=UnitTraceNormalization(x{1},kernel,kerneloptionvec,variableveccell);
K{1}=mklkernel(x{1},InfoKernel,Weight(1,:));
% parfor i=2:class
for i=2:class
    [Weight(i,:),~]=UnitTraceNormalization(x{i},kernel,kerneloptionvec,variableveccell);
    K{i}=mklkernel(x{i},InfoKernel,Weight(i,:));
end
t=toc(tstart);

time_train=zeros(length(range_C),1);
time_validation=zeros(length(range_C),1);
k=1;
% 优化惩罚参数
accuracy_best=0;
for C=range_C
    disp(strcat('C=',num2str(C)));
    tstart=tic;
%     parfor i=1:class
    for i=1:class
%         训练
        disp(strcat('class',num2str(i)));
        [kernel_weight(i,:),alpha{i},bias(i),supportvector_index] = mklsvm(K{i},y{i},C,options,verbose);
        supportvector{i}=x{i}(supportvector_index,:);
    end
    time_train(k)=toc(tstart);
%     在验证集上测试
    tstart=tic;
    accuracy=test({alpha,supportvector},bias,x_validation,y_validation,sample_validation,0,'respective',InfoKernel,Weight,kernel_weight);
    time_validation(k)=toc(tstart);
    disp(strcat('验证集上准确率为',num2str(100*accuracy(3,1)),'%'));
    if accuracy(3,1)>accuracy_best
        C_best=C;
        kernel_weight_best=kernel_weight;
        alpha_best=alpha;
        bias_best=bias;
        supportvector_best=supportvector;
        accuracy_best=accuracy(3,1);
    end
    k=k+1;
end
time_train=mean(time_train)+t;
time_validation=mean(time_validation);
end